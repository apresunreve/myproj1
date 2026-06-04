from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class AttentionExperimentConfig:
    name: str
    latent_dim: int | None = None
    disable_cache: bool = False


class Qwen2_5_VLMLAAttention(nn.Module):
    """MLA-style pilot attention for Qwen2.5-VL.

    This pilot keeps the external attention contract unchanged and routes K/V
    through a shared latent representation. The default initialization preserves
    baseline behavior exactly, which makes it a safe inference rewrite target.
    """

    def __init__(self, original_attn, latent_dim: int | None = None) -> None:
        super().__init__()
        self.config = original_attn.config
        self.layer_idx = original_attn.layer_idx
        self.hidden_size = original_attn.hidden_size
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim
        self.num_key_value_heads = original_attn.num_key_value_heads
        self.num_key_value_groups = original_attn.num_key_value_groups
        self.is_causal = original_attn.is_causal
        self.attention_dropout = original_attn.attention_dropout
        self.rope_parameters = original_attn.rope_parameters
        self.scaling = original_attn.scaling
        self.layer_type = getattr(original_attn, "layer_type", None)
        self.sliding_window = getattr(original_attn, "sliding_window", None)

        self.q_proj = original_attn.q_proj
        self.o_proj = original_attn.o_proj
        self.kv_hidden_size = self.num_key_value_heads * self.head_dim
        proj_device = original_attn.k_proj.weight.device
        proj_dtype = original_attn.k_proj.weight.dtype
        linear_kwargs = {"device": proj_device, "dtype": proj_dtype}
        # For the pilot, preserve the original K/V information exactly via a
        # shared latent vector, then split it back into K and V projections.
        self.latent_dim = latent_dim or (2 * self.kv_hidden_size)
        if self.latent_dim != 2 * self.kv_hidden_size:
            raise ValueError(
                "The current MLA pilot supports only latent_dim == 2 * kv_hidden_size "
                "so the rewrite remains inference-safe before any retraining."
            )
        self.kv_latent_proj = nn.Linear(self.hidden_size, self.latent_dim, bias=True, **linear_kwargs)
        self.k_up_proj = nn.Linear(self.latent_dim, self.kv_hidden_size, bias=False, **linear_kwargs)
        self.v_up_proj = nn.Linear(self.latent_dim, self.kv_hidden_size, bias=False, **linear_kwargs)
        self._copy_from_original(original_attn)

    def _copy_from_original(self, original_attn) -> None:
        with torch.no_grad():
            self.kv_latent_proj.weight.zero_()
            self.kv_latent_proj.bias.zero_()
            self.kv_latent_proj.weight[: self.kv_hidden_size].copy_(original_attn.k_proj.weight)
            self.kv_latent_proj.weight[self.kv_hidden_size :].copy_(original_attn.v_proj.weight)
            if original_attn.k_proj.bias is not None:
                self.kv_latent_proj.bias[: self.kv_hidden_size].copy_(original_attn.k_proj.bias)
            if original_attn.v_proj.bias is not None:
                self.kv_latent_proj.bias[self.kv_hidden_size :].copy_(original_attn.v_proj.bias)

            self.k_up_proj.weight.zero_()
            self.v_up_proj.weight.zero_()
            eye = torch.eye(self.kv_hidden_size, dtype=self.k_up_proj.weight.dtype, device=self.k_up_proj.weight.device)
            self.k_up_proj.weight[:, : self.kv_hidden_size].copy_(eye)
            self.v_up_proj.weight[:, self.kv_hidden_size :].copy_(eye)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            ALL_ATTENTION_FUNCTIONS,
            apply_multimodal_rotary_pos_emb,
            eager_attention_forward,
        )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        latent_kv = self.kv_latent_proj(hidden_states)
        key_states = self.k_up_proj(latent_kv)
        value_states = self.v_up_proj(latent_kv)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.config.rope_parameters["mrope_section"]
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            position_ids=position_ids,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def apply_qwen25vl_mla_rewrite(model, *, latent_dim: int | None = None):
    layers = model.model.language_model.layers
    for layer in layers:
        if isinstance(layer.self_attn, Qwen2_5_VLMLAAttention):
            continue
        layer.self_attn = Qwen2_5_VLMLAAttention(layer.self_attn, latent_dim=latent_dim)
    return model
