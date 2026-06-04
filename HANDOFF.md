# Session Handoff — Wildlife Behavior VLM Project

*For the next person/session picking up this project.*
*Last updated: 2026-05-03.*

This document is operational. If you want the paper content, see
[`PAPER_DRAFT_SUMMARY.md`](PAPER_DRAFT_SUMMARY.md). If you want the
phase-by-phase plan, see [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md).

---

## 1. Where things are right now

- **Phase 1 (baseline table)**: ✅ complete
- **Phase 2 (Contribution B — class-collapse-aware training)**: ✅ complete,
  B2 inv-sqrt class-weighted CE wins (+4.7 macro F1 vs B1 baseline)
- **Phase 3 (Contribution A — joint cross-dataset LoRA)**: 🟡 A1 done and
  evaluated; A3 (= A1 + B2 combined) trained, evals queued in DB
- **Phase 4 (scale check, SFT ablation)**: not started
- **Phase 5 (figures)**: F1/F2/F4 done at
  `artifacts/figures/`; F3/F5 not started

## 2. Repository map (only the essentials)

```
/scratch/general/vast/j.yan/myproj1/
├── PAPER_DRAFT_SUMMARY.md      ← paper-ready content
├── HANDOFF.md                  ← you are here
├── PROJECT_SUMMARY.md          ← older paper-style summary (still useful)
├── EXPERIMENT_PLAN.md          ← phase plan
├── EXPERIMENT_IDEAS.md         ← original brainstorm
├── RELATED_WORK_BEHAVIOR_VLMS.md
├── current_clip_results.csv    ← original KABR/bbl CSV
├── chpc_manager/               ← project-local task queue
│   ├── config.yaml             ← points to vlm_behavior.db
│   ├── manager_daemon.py       ← USER LAUNCHES THIS
│   ├── task_executor.py
│   ├── task_db.py
│   ├── cli.py
│   └── gpu_specs.py            ← granite/notchpeak node memory lookup
├── configs/                    ← LoRA + smoke YAMLs
├── scripts/                    ← prep + wrapper scripts
│   ├── build_joint_train_jsonl.py
│   ├── prepare_baboonland_behavior_jsonl.py
│   ├── eval_cross_dataset_vlm.py
│   ├── run_vlm_lora_train_task.sh    ← DB-driven LoRA wrapper
│   ├── run_vlm_config_train.sh        ← joint training entry (trusts config)
│   ├── run_vlm_eval_task.sh
│   ├── analysis/build_figures.py
│   └── ...
├── src/kabr_vlm/
│   ├── training/qwen_tasks.py  ← has class-aware Trainer subclass (Phase 2)
│   ├── data/{kabr,baboonland}_frames.py
│   └── vlm_baselines.py        ← generator loaders, label matching
├── data/processed/
│   ├── kabr_behavior_{train,val_cleaned}.jsonl
│   ├── baboonland_behavior_{train,val}.jsonl
│   └── joint_kabr_bbl_behavior_train.jsonl
├── artifacts/
│   ├── runs/baboonland/...     ← in-domain bbl LoRA adapters (6 models)
│   ├── runs/joint/...          ← Phase-3 joint adapters (A1, A3)
│   ├── baboonland_eval/        ← bbl val metrics per model/method
│   ├── transfer_eval/          ← cross-dataset eval metrics
│   ├── joint_eval/             ← Phase-3 joint adapter metrics
│   ├── figures/                ← F1, F2, F4 paper figures (PNG)
│   └── x3d_metrics/, latency_benchmark/, ...
├── artifacts_2/artifacts/      ← earlier metrics imported from another machine
├── baboonland_files/           ← raw clip manifests
└── data/                       ← processed training JSONLs

External:
/scratch/general/nfs1/u1320844/baboon2/charades/   ← raw baboonland frames
/scratch/general/nfs1/u1320844/KABR_files2/        ← raw KABR frames
/scratch/general/nfs1/u1320844/manual_train/       ← X3D/I3D SlowFast outputs
/scratch/general/vast/j.yan/SlowFast/              ← SlowFast clone with kabr_clips.py
/uufs/chpc.utah.edu/common/home/u1320844/task_dbs/ ← project task DB
```

## 3. Task DB

- **DB path**: `/uufs/chpc.utah.edu/common/home/u1320844/task_dbs/vlm_behavior.db`
- **Config**: `chpc_manager/config.yaml`
- **Task types registered in `/scratch/general/vast/j.yan/NAS/task_types.yaml`**:
  - `vlm_lora_train` — calls `run_vlm_lora_train_task.sh`
  - `vlm_cross_eval` — calls `run_vlm_eval_task.sh`
  - `slowfast_clip_train` — calls `run_slowfast_clip_train_task.sh`

### Useful commands

```bash
cd /scratch/general/vast/j.yan/myproj1/chpc_manager
PY=/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python

# List all tasks
$PY cli.py --config config.yaml list

# Status
$PY cli.py --config config.yaml status

# Add a task (example: queue an eval)
$PY cli.py --config config.yaml add vlm_cross_eval \
  '{"val_jsonl":"/scratch/.../val.jsonl","target_dataset":"baboonland",
    "model_id":"Qwen/Qwen2.5-VL-3B-Instruct",
    "adapter_path":"/scratch/.../checkpoint",
    "output_jsonl":"/scratch/.../predictions.jsonl"}'

# Launch the manager daemon (user does this; not automatic)
python manager_daemon.py --config config.yaml &
```

### Important DB gotchas

- **DB integrity has been corrupted once.** If `sqlite3 ... "PRAGMA
  integrity_check"` returns errors, run `REINDEX` and reset stuck
  `running:<jobid>` rows to `idle` (a SLURM job is always referenced by
  its actual current state via `squeue`).
- **The `task_executor.py` in our chpc_manager has a critical fix.**
  It supports `--key value` args for `command_type: bash`. If you ever
  re-copy from `/scratch/general/vast/j.yan/myproj/chpc_manager`, you
  *will* break our wrappers. The repo-local version is the canonical one.

## 4. GPU memory routing

`chpc_manager/gpu_specs.py` lists every known GPU node on granite,
notchpeak, lonepeak, kingspeak with effective memory. The daemon excludes
nodes with `< gpu_memory_gb` (set in task_types.yaml) for each task type.

- `vlm_lora_train`: `gpu_memory_gb: 40` → routes away from 18-24 GB nodes
  (grn065, grn081-086, MIG slices, etc.)
- `vlm_cross_eval`: `gpu_memory_gb: 21` → can run on smaller cards
- `slowfast_clip_train`: `gpu_memory_gb: 40`

If a job OOMs on a node not yet in `gpu_specs.py`, add it.

## 5. Where the data lives

### Raw datasets
- KABR raw frames: `/scratch/general/nfs1/u1320844/KABR_files2/`
- Baboonland raw frames: `/scratch/general/nfs1/u1320844/baboon2/charades/`

### Derived clip manifests
- KABR: `/scratch/general/nfs1/u1320844/KABR_clip_dataset_cleaned_2026-04-07/annotation/clip_manifests/`
- Baboonland: `/scratch/general/vast/j.yan/myproj1/baboonland_files/annotation/clip_manifests/`

### Processed VLM training JSONLs
- `data/processed/kabr_behavior_train.jsonl` (11003 rows)
- `data/processed/kabr_behavior_val_cleaned.jsonl` (3761 rows)
- `data/processed/baboonland_behavior_train.jsonl` (20130 rows)
- `data/processed/baboonland_behavior_val.jsonl` (6198 rows)
- `data/processed/joint_kabr_bbl_behavior_train.jsonl` (31133 rows)
  *(joint val is just baboonland val since that's the harder set)*

### Adapter checkpoints (the artifacts that matter for the paper)
- Per-model bbl LoRA: `artifacts/runs/baboonland/{model}_behavior_lora/checkpoint/`
- KABR LoRA (from earlier session): `artifacts_2/artifacts/runs/qwen25vl3b_behavior_lora_full/checkpoint/`
- Joint Phase-3: `artifacts/runs/joint/qwen25vl3b_joint_kabr_bbl_lora_{a1,a3_invsqrt}/checkpoint/`

## 6. Eval / scoring conventions

`scripts/eval_cross_dataset_vlm.py`:
- Reads a processed val JSONL
- Loads a base model (+ optional LoRA adapter)
- Generates text completions (16 new tokens, do_sample=False)
- Extracts a label using `match_candidate` (normalized substring) against
  the target dataset's label list, with dataset-specific fuzzy fallbacks
  (`extract_baboonland_label`, `extract_kabr_label`)
- Writes per-clip predictions to `<output>.jsonl`
- Writes metrics (acc, macro F1, per-class F1, pred + gold distributions)
  to `<output>.jsonl.metrics.json`

**Important: the completion-file path expected by the DB is
`<output_jsonl>.metrics.json`** (concat, not suffix-replace). Don't
revert this.

## 7. Phase-2 class-aware Trainer

`src/kabr_vlm/training/qwen_tasks.py:1131+` has a subclassed
`_ClassAwareTrainer` that activates when either:
- `training.class_weighting in {"balanced", "inv_sqrt", "focal"}`, or
- `training.class_sampler == "class_balanced"`

It computes class counts from `train_dataset` labels, builds a
per-example weight tensor, optionally a `WeightedRandomSampler`, and for
`focal` applies a per-batch scale `(1 - exp(-loss))**γ`.

To use: just add to the YAML's `training:` block:
```yaml
class_weighting: inv_sqrt
class_sampler: uniform  # or class_balanced
focal_gamma: 2.0        # if class_weighting: focal
```

## 8. Phase-3 joint training pipeline

`scripts/build_joint_train_jsonl.py` interleaves KABR + bbl processed
JSONLs (seed-controlled shuffle). Output:
`data/processed/joint_kabr_bbl_behavior_train.jsonl`.

`scripts/run_vlm_config_train.sh` is the launcher that **trusts the
config's data paths** (doesn't override them). Use `--dataset joint`
when calling `run_vlm_lora_train_task.sh` to route through this.

The dataset-specific launchers (`run_vlm_behavior_full.sh`,
`run_vlm_baboonland_full.sh`) hard-code train_file. If you re-use those
for joint training you will silently train on the wrong data.

## 9. Open issues to be aware of

- **InternVL2.5-2B LoRA**: `modeling_internvl_chat.py` has an
  image-token mismatch when fed multi-image clips. Forward pass fails
  with `'bool' object has no attribute 'sum'`. Needs a custom prompt
  renderer that emits the right `<IMG_CONTEXT>` count for 8 images.
  Defer or skip.
- **MIG slice nodes on granite**: grn025, grn026 expose H200 MIG
  partitions as `gpu:1`. Already in the exclude list in
  `chpc_manager/gpu_specs.py`. If you ever see a sub-40-GB OOM with a
  `gpu_memory_gb: 40` task, that's the first place to look.
- **CSV `current_clip_results.csv` is stale.** It has a placeholder
  `InternVL2-2B,KABR-Clip,0.3273,0.0616,zero-shot` row that's identical
  to MobileVLM's — almost certainly copy-pasted by an earlier run.
  Disregard. Canonical source of truth: `PAPER_DRAFT_SUMMARY.md` §4.

## 10. The next things to do

In recommended order:

1. **Wait for A3 evals to land** (tasks 21, 22 in vlm_behavior.db).
   Refresh tables in `PAPER_DRAFT_SUMMARY.md` §4.5 once they do.
2. **Re-run figures**:
   `/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python
   scripts/analysis/build_figures.py` (adds the A3 column to the
   per-behavior heatmap and pred-distribution plot).
3. **Optional: Qwen3B full-SFT on KABR** as a single-row reviewer-defense
   ablation. Existing config:
   `configs/qwen25vl3b_behavior_full_sft_full.yaml` (uses full-param SFT
   path in qwen_tasks.py if `training.method: full`).
4. **Start paper writing.** All six claims have data behind them; the
   abstract paragraph in `PAPER_DRAFT_SUMMARY.md` §10 is a starting
   point.
5. **(If reviewers ask)** Add MammAlps zero-shot or A2 (V1 label-mapped
   joint training) as response experiments.

## 11. Contact / pitch material for advisor

See:
- [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) §9 — pitch arc for advisor
- The Slack-draft + spoken-pitch versions in earlier session transcripts
  (one-pager friendly framing)
- [`PAPER_DRAFT_SUMMARY.md`](PAPER_DRAFT_SUMMARY.md) §10 — abstract
  paragraph

The framing that lands well: *"the X3D-NAS baselines I ran exposed that
VLMs already beat X3D on accuracy, so the NAS target is now closing the
VLM gap at specialist efficiency — and here are two training contributions
the field doesn't have yet (class-collapse-aware loss + cross-dataset
joint LoRA), demonstrated end-to-end."*
