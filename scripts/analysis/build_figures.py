"""Build F1 (per-behavior heatmap), F2 (pred-distribution class-collapse),
and F4 (accuracy vs macro-F1 scatter) from the metrics JSONs on disk.

Run with the miniconda python (has matplotlib+seaborn):
  /uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python \
      scripts/analysis/build_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(context="paper", style="whitegrid")

REPO = Path("/scratch/general/vast/j.yan/myproj1")
FIG_DIR = REPO / "artifacts" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BABOONLAND_CLASSES = [
    "Walking/Running", "Sitting/Standing", "Fighting/Playing", "Self-Grooming",
    "Being Groomed", "Grooming Somebody", "Mutual Grooming", "Infant-Carrying",
    "Foraging", "Drinking", "Mounting", "Sleeping", "Occluded",
]


def load_new(path):
    d = json.loads(Path(path).read_text())
    pc = {k: v["f1"] for k, v in d.get("per_class_f1", {}).items()}
    return {
        "acc": d.get("accuracy", 0.0),
        "f1": d.get("macro_f1", 0.0),
        "pc": pc,
        "pred_dist": d.get("pred_distribution", {}),
        "gold_dist": d.get("gold_distribution", {}),
        "n": d.get("n_total", 0),
    }


def load_legacy(path):
    d = json.loads(Path(path).read_text())
    raw = d.get("per_class", {})
    pc = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                pc[k] = v.get("f1", 0.0)
    return {"acc": d.get("accuracy", 0.0), "f1": d.get("macro_f1", 0.0), "pc": pc}


BBL_ROWS_NEW = [
    ("Qwen7B ZS",
     REPO / "artifacts_2/artifacts/baboonland_eval/qwen25vl7b_baboonland_behavior_clip_val_full.metrics.json"),
    ("LLaVA-OV-7B ZS",
     REPO / "artifacts_2/artifacts/baboonland_eval/llava_onevision_qwen2_7b_baboonland_behavior_clip_val_full.metrics.json"),
    ("Qwen3B ZS",
     REPO / "artifacts_2/artifacts/baboonland_eval/qwen25vl3b_baboonland_behavior_clip_val_full.metrics.json"),
    ("LLaVA-OV-0.5B ZS",
     REPO / "artifacts/baboonland_eval/llava_ov_0_5b_baboonland_zero_shot.metrics.json"),
    ("MobileVLM ZS",
     REPO / "artifacts/baboonland_eval/mobilevlm_v2_1_7b_baboonland_zero_shot.metrics.json"),
    ("Qwen7B LoRA",
     REPO / "artifacts_2/artifacts/transfer_eval/qwen25vl7b_baboonland_behavior_lora_on_baboonland.metrics.json"),
    ("Qwen3B LoRA",
     REPO / "artifacts_2/artifacts/transfer_eval/qwen25vl3b_baboonland_behavior_lora_on_baboonland.metrics.json"),
    ("LLaVA-OV-7B LoRA",
     REPO / "artifacts/baboonland_eval/llava_ov_7b_baboonland_lora.metrics.json"),
    ("LLaVA-OV-0.5B LoRA",
     REPO / "artifacts/baboonland_eval/llava_ov_0_5b_baboonland_lora.metrics.json"),
    ("MobileVLM LoRA",
     REPO / "artifacts/baboonland_eval/mobilevlm_v2_1_7b_baboonland_lora.metrics.json"),
]


def fig_per_behavior_heatmap():
    names = []
    matrix = []
    for name, path in BBL_ROWS_NEW:
        if not path.exists():
            continue
        row = load_new(path)
        names.append(name)
        matrix.append([row["pc"].get(cls, 0.0) for cls in BABOONLAND_CLASSES])
    arr = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.35 * len(names)))
    sns.heatmap(
        arr,
        xticklabels=BABOONLAND_CLASSES,
        yticklabels=names,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "per-class F1"},
        ax=ax,
    )
    ax.set_title("F1 per behavior — Baboonland val (6198 clips, 13 classes)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    plt.tight_layout()
    out = FIG_DIR / "F1_baboonland_per_behavior_heatmap.png"
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def fig_pred_distribution():
    targets = [
        ("MobileVLM ZS",
         REPO / "artifacts/baboonland_eval/mobilevlm_v2_1_7b_baboonland_zero_shot.metrics.json"),
        ("LLaVA-OV-0.5B ZS",
         REPO / "artifacts/baboonland_eval/llava_ov_0_5b_baboonland_zero_shot.metrics.json"),
        ("LLaVA-OV-0.5B LoRA",
         REPO / "artifacts/baboonland_eval/llava_ov_0_5b_baboonland_lora.metrics.json"),
        ("MobileVLM LoRA",
         REPO / "artifacts/baboonland_eval/mobilevlm_v2_1_7b_baboonland_lora.metrics.json"),
    ]
    valid = [(n, p) for n, p in targets if p.exists()]
    n_rows = len(valid)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(10, 2.2 * n_rows), sharex=True
    )
    if n_rows == 1:
        axes = [axes]

    gold_ref = None
    for ax, (name, path) in zip(axes, valid):
        d = load_new(path)
        gold = d["gold_dist"]
        pred = d["pred_dist"]
        if gold_ref is None:
            gold_ref = gold
        n = d["n"]
        xs = np.arange(len(BABOONLAND_CLASSES))
        gvec = np.array([gold.get(c, 0) / n for c in BABOONLAND_CLASSES])
        pvec = np.array([pred.get(c, 0) / n for c in BABOONLAND_CLASSES])
        w = 0.4
        ax.bar(xs - w / 2, gvec, width=w, color="#4c78a8", label="gold")
        ax.bar(xs + w / 2, pvec, width=w, color="#e45756", label="pred")
        ax.set_ylabel("share")
        ax.set_title(f"{name} · acc={d['acc']:.3f} · macroF1={d['f1']:.3f}")
        ax.set_xticks(xs)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xticklabels(BABOONLAND_CLASSES, rotation=35, ha="right")
    fig.suptitle("F2 Class-collapse signature — baboonland predicted vs gold distribution", y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "F2_baboonland_pred_distribution.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_acc_vs_f1():
    # Collect every available row on baboonland + KABR
    points = []

    # Baboonland new-format
    for name, path in BBL_ROWS_NEW:
        if not path.exists():
            continue
        d = load_new(path)
        method = "LoRA" if "LoRA" in name else "zero-shot"
        points.append(("Baboonland", name.replace(" LoRA", "").replace(" ZS", ""),
                       method, d["acc"], d["f1"]))

    # KABR legacy metrics
    a2 = REPO / "artifacts_2/artifacts"
    kabr_sources = [
        ("Qwen7B", f"{a2}/full_eval/qwen25vl7b_behavior_clip_val_full.metrics.json", "zero-shot"),
        ("Qwen3B", f"{a2}/full_eval/qwen25vl3b_behavior_clip_val_full.metrics.json", "zero-shot"),
        ("Gemma3-4B", f"{a2}/full_eval/gemma3_4b_behavior_clip_val_full.metrics.json", "zero-shot"),
        ("LLaVA-OV-7B", f"{a2}/full_eval/llava_onevision_qwen2_7b_behavior_clip_val_full.metrics.json", "zero-shot"),
        ("Idefics3-8B", f"{a2}/full_eval/idefics3_8b_behavior_clip_val_full.metrics.json", "zero-shot"),
        ("SmolVLM-256M", f"{a2}/full_eval/smol_behavior_clip_val_full.metrics.json", "zero-shot"),
    ]
    for name, path, method in kabr_sources:
        if Path(path).exists():
            d = load_legacy(path)
            points.append(("KABR", name, method, d["acc"], d["f1"]))

    # KABR LoRA numbers from current_clip_results.csv
    csv_path = REPO / "current_clip_results.csv"
    if csv_path.exists():
        lines = csv_path.read_text().splitlines()[1:]
        for line in lines:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            model, dataset, acc, f1, method = parts[:5]
            if "kabr" not in dataset.lower():
                continue
            if "lora" not in method.lower():
                continue
            try:
                points.append(("KABR", model, "LoRA", float(acc), float(f1)))
            except ValueError:
                pass

    # X3D-L on KABR (supervised)
    points.append(("KABR", "X3D-L", "supervised", 0.7968, 0.5199))
    points.append(("KABR", "I3D-L", "supervised", 0.8144, 0.0))  # F1 unknown → 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    palette = {"zero-shot": "#4c78a8", "LoRA": "#e45756", "supervised": "#54a24b"}
    markers = {"zero-shot": "o", "LoRA": "s", "supervised": "^"}

    for ax, ds_name in zip(axes, ["KABR", "Baboonland"]):
        subset = [p for p in points if p[0] == ds_name]
        for method in ["zero-shot", "LoRA", "supervised"]:
            pts = [(p[3], p[4], p[1]) for p in subset if p[2] == method]
            if not pts:
                continue
            xs = [x for x, _, _ in pts]
            ys = [y for _, y, _ in pts]
            labels = [l for _, _, l in pts]
            ax.scatter(xs, ys, s=60, c=palette[method], marker=markers[method], label=method, alpha=0.85, edgecolor="black", linewidth=0.3)
            for x, y, lab in pts:
                ax.annotate(lab, (x, y), fontsize=7, alpha=0.9, xytext=(3, 3), textcoords="offset points")
        # y=x guide
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 1.0)
        ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.4, linewidth=0.8)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 0.7)
        ax.set_xlabel("accuracy")
        ax.set_title(ds_name)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("macro F1")
    fig.suptitle("F4 Accuracy vs macro-F1 — class collapse lives under the diagonal", y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "F4_acc_vs_macro_f1.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    fig_per_behavior_heatmap()
    fig_pred_distribution()
    fig_acc_vs_f1()


if __name__ == "__main__":
    main()
