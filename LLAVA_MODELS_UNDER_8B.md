# LLaVA Models <= 8B For Future Planning

Last checked: 2026-04-10

This is a planning list of **official or major Hugging Face-packaged LLaVA-family models that are 8B or smaller**.

Scope note:
- This is **not** a list of every community fork on Hugging Face.
- It focuses on the main families we are most likely to benchmark or smoke test in this repo.

## llava-hf

### LLaVA-OneVision

| Model ID | Approx Size | Notes |
|---|---:|---|
| `llava-hf/llava-onevision-qwen2-0.5b-si-hf` | 0.9B | single-image variant |
| `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | 0.9B | OneVision multi-image/video-oriented variant |
| `llava-hf/llava-onevision-qwen2-7b-si-hf` | 8B | single-image variant |
| `llava-hf/llava-onevision-qwen2-7b-ov-hf` | 8B | OneVision multi-image/video-oriented variant |

### LLaVA-NeXT / LLaVA-1.6

| Model ID | Approx Size | Notes |
|---|---:|---|
| `llava-hf/llava-v1.6-vicuna-7b-hf` | 7B | classic NeXT family |
| `llava-hf/llava-v1.6-mistral-7b-hf` | 8B | Mistral-backed NeXT variant |

### LLaVA-1.5

| Model ID | Approx Size | Notes |
|---|---:|---|
| `llava-hf/llava-1.5-7b-hf` | 7B | older but still common baseline |
| `llava-hf/bakLlava-v1-hf` | 8B | BakLLaVA variant in the same HF family collection |

### LLaVA Video

| Model ID | Approx Size | Notes |
|---|---:|---|
| `llava-hf/LLaVA-Next-Video-7B-Qwen2-hf` | 8B | Qwen2-backed video variant |
| `llava-hf/LLaVA-NeXT-Video-7B-hf` | 7B | another sub-8B video-family checkpoint |

## Deep-VLM / LLaVA-OneVision-1.5

| Model ID | Approx Size | Notes |
|---|---:|---|
| `Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf` | 4B | newer OneVision-1.5 line |
| `Deep-VLM/LLaVA-OneVision-1.5-8B-Instruct-hf` | 8B | larger but still within our <=8B planning cutoff |

## Likely Priority Candidates For This Repo

If we revisit LLaVA-family experiments here, these are the most practical first choices:

1. `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`
2. `llava-hf/llava-onevision-qwen2-7b-ov-hf`
3. `Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf`
4. `llava-hf/llava-v1.6-mistral-7b-hf`

## Source Pointers

- `llava-hf` collections page:
  - https://huggingface.co/llava-hf/collections
- `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`:
  - https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf
- `llava-hf/LLaVA-Next-Video-7B-Qwen2-hf`:
  - https://huggingface.co/llava-hf/LLaVA-Next-Video-7B-Qwen2-hf
- `Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf`:
  - https://huggingface.co/Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf
- `Deep-VLM/LLaVA-OneVision-1.5-8B-Instruct-hf`:
  - https://huggingface.co/Deep-VLM/LLaVA-OneVision-1.5-8B-Instruct-hf
