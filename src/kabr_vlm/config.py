from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    train_file: Path
    val_file: Path
    video_root: Path | None = None
    task_mode: str = "animal_recognition"
    input_type: str = "clip"
    max_frames_per_clip: int = 8
    max_train_samples: int | None = None
    max_val_samples: int | None = None


class ModelConfig(BaseModel):
    base_model_id: str = Field(alias="name")
    processor_id: str | None = None
    adapter_path: Path | None = None
    attention_experiment: str = "baseline"
    max_frames: int = 32
    precision: str = "bf16"
    prompt_mode: str = "label"
    output_format: str = "plain"

    @property
    def name(self) -> str:
        return self.base_model_id

    @property
    def resolved_processor_id(self) -> str:
        return self.processor_id or self.base_model_id


class TrainingConfig(BaseModel):
    output_dir: Path
    method: str = "lora"
    epochs: int = 1
    batch_size: int = 1
    eval_batch_size: int = 1
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 8
    weight_decay: float = 0.0
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    save_strategy: str = "steps"
    save_steps: int = 250
    eval_strategy: str = "epoch"
    save_total_limit: int = 2
    logging_steps: int = 10
    seed: int = 42
    dry_run: bool = True
    feature_batch_size: int = 2
    classifier_hidden_dim: int = 256
    classifier_dropout: float = 0.1
    classifier_epochs: int = 25
    classifier_learning_rate: float = 1e-3
    classifier_weight_decay: float = 1e-4
    classifier_type: str = "linear"
    distill_temperature: float = 2.0
    distill_alpha: float = 0.5
    # Phase-2 class-collapse-aware training.
    # class_weighting: none | balanced | inv_sqrt
    #   balanced  -> weight_c = N / (num_classes * count_c)        (sklearn convention)
    #   inv_sqrt  -> weight_c = sqrt(max_count / count_c)          (gentler, our default for imbalance)
    #   focal     -> focal(gamma) on per-example loss
    # class_sampler: uniform | class_balanced
    class_weighting: str = "none"
    focal_gamma: float = 2.0
    class_sampler: str = "uniform"


class AppConfig(BaseModel):
    project_name: str
    seed: int = 42
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return AppConfig.model_validate(raw)
