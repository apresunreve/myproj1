from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.make_current_status_plan_tables_ppt import (
    GOLD,
    GRAY,
    NAVY,
    TEAL,
    WHITE,
    add_footer,
    add_table,
    add_title,
    bullet_box,
    card,
    set_bg,
)


OUT = Path("artifacts/presentations/comprehensive_behavior_results_2026-05-20.pptx")


def title_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, RGBColor(248, 250, 252))
    title = slide.shapes.add_textbox(Inches(0.7), Inches(0.9), Inches(9.1), Inches(1.0))
    p = title.text_frame.paragraphs[0]
    p.text = "Comprehensive Behavior Results"
    p.font.size = Pt(38)
    p.font.bold = True
    p.font.color.rgb = NAVY
    subtitle = slide.shapes.add_textbox(Inches(0.75), Inches(1.82), Inches(8.8), Inches(0.6))
    sp = subtitle.text_frame.paragraphs[0]
    sp.text = "KABR, Baboonland, transfer, joint training, class-collapse mitigation, and small VLMs"
    sp.font.size = Pt(18)
    sp.font.color.rgb = GRAY
    rows = [
        ["Source files re-read", "Role"],
        ["CURRENT_BEHAVIOR_RESULTS.md", "Detailed KABR behavior ladder"],
        ["CURRENT_BEHAVIOR_SUMMARY.md", "Dataset construction and reliable KABR setup"],
        ["CURRENT_RESULTS_SUMMARY.md", "Paper-level KABR/Baboonland/transfer/joint story"],
        ["small_vlm_summary.md", "New small-VLM zero-shot screening"],
    ]
    add_table(slide, 0.8, 3.05, 7.9, 2.2, rows, [3.2, 4.1], 10.5)
    card(slide, 9.35, 1.35, 3.0, 1.15, "Date", "2026-05-20", TEAL)
    card(slide, 9.35, 2.85, 3.0, 1.15, "Format", "Table-forward\nresults deck", GOLD)
    card(slide, 9.35, 4.35, 3.0, 1.15, "Use", "Advisor update\nor paper meeting", NAVY)
    add_footer(slide, 1)


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    title_slide(prs)

    # 2
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_title(slide, "Dataset And Benchmark Setup", "Clip-level behavior recognition with shared manifests across model families.")
    rows = [
        ["Dataset", "Domain", "Classes", "Train clips", "Val clips", "Difficulty"],
        ["KABR-Clip", "Drone footage of Kenyan wildlife", "8", "11,003", "3,761", "Moderate imbalance"],
        ["Baboonland-Clip", "Drone footage of baboons", "13", "20,130", "6,198", "Severe imbalance"],
        ["Joint train", "KABR + Baboonland union", "Native label spaces", "31,133", "-", "Mixed"],
    ]
    add_table(slide, 0.7, 1.35, 11.9, 1.65, rows, [1.9, 3.35, 1.55, 1.55, 1.35, 2.0], 10)
    rows = [
        ["Construction step", "Description"],
        ["Raw annotations", "Start from frame/span behavior annotations with original video id and label id."],
        ["Clip derivation", "Group consecutive frames with same original video id and same behavior label."],
        ["Frame sampling", "Sample ordered RGB frames per derived clip for VLM inputs."],
        ["JSONL export", "Write canonical train/val manifests used by both VLM and video baselines."],
    ]
    add_table(slide, 0.7, 3.55, 11.9, 2.25, rows, [2.65, 8.65], 10)
    add_footer(slide, 2)

    # 3
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_title(slide, "KABR Validation Label Support", "KABR has moderate imbalance, with several rare behavior classes.")
    rows = [
        ["Class", "Val clips"],
        ["Walk", "1,264"],
        ["Graze", "1,204"],
        ["Browse", "108"],
        ["Head Up", "968"],
        ["Auto-Groom", "35"],
        ["Trot", "50"],
        ["Run", "51"],
        ["Occluded", "81"],
    ]
    add_table(slide, 0.9, 1.35, 4.4, 4.25, rows, [2.5, 1.4], 13)
    bullet_box(
        slide,
        6.0,
        1.55,
        6.2,
        3.2,
        [
            "Most clips are Walk, Graze, or Head Up.",
            "Rare classes make macro-F1 necessary.",
            "A high-accuracy model can still fail the ecological use case if rare behaviors collapse.",
        ],
        font_size=18,
    )
    add_footer(slide, 3)

    # 4
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_title(slide, "KABR Method Ladder", "Reliable rows on the matched 3,761-clip KABR behavior validation set.")
    rows = [
        ["Family", "Model / method", "Acc", "F1"],
        ["Zero-shot", "SmolVLM-256M", "0.249", "0.069"],
        ["Zero-shot", "Qwen2.5-VL-3B", "0.506", "0.260"],
        ["Zero-shot", "Gemma 3 4B", "0.550", "0.227"],
        ["Zero-shot", "Idefics3-8B", "0.529", "0.231"],
        ["Zero-shot", "LLaVA-OV-Qwen2-7B", "0.573", "0.269"],
        ["Zero-shot", "Qwen2.5-VL-7B", "0.611", "0.317"],
        ["Video", "X3D-XS / S / M best", "0.670", "0.426"],
        ["Frozen", "Qwen3B frozen linear", "0.682", "0.354"],
        ["Distill", "Qwen3B distill MLP", "0.652", "0.274"],
        ["LoRA", "Qwen3B LoRA 8 frames", "0.854", "0.485"],
        ["LoRA", "Qwen3B LoRA 4 frames", "0.849", "0.499"],
    ]
    add_table(slide, 0.55, 1.25, 12.25, 5.35, rows, [1.45, 7.25, 1.35, 1.35], 8.7)
    add_footer(slide, 4)

    # 5
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_title(slide, "KABR Paper-Level Model Comparison", "Broader KABR table used for the adapted-VLM story.")
    rows = [
        ["Model", "Zero-shot Acc/F1", "LoRA Acc/F1", "Joint Acc/F1"],
        ["Gemma3-4B", "0.549 / 0.226", "0.864 / 0.515", "-"],
        ["LLaVA-OV-7B", "0.581 / 0.254", "0.863 / 0.523", "-"],
        ["Qwen2.5-VL-3B", "0.549 / 0.271", "0.853 / 0.485", "0.863 / 0.514"],
        ["Qwen2.5-VL-7B", "0.626 / 0.327", "0.848 / 0.503", "-"],
        ["LLaVA-OV-0.5B", "0.307 / 0.119", "0.819 / 0.447", "-"],
        ["MobileVLM-1.7B", "0.327 / 0.062", "0.782 / 0.431", "-"],
        ["X3D-L specialist", "-", "0.797 / 0.520", "-"],
        ["I3D-L specialist", "-", "0.814 / -", "-"],
    ]
    add_table(slide, 0.7, 1.25, 11.9, 4.65, rows, [3.2, 2.5, 2.5, 2.5], 9.5)
    add_footer(slide, 5)

    # 6
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_title(slide, "Baboonland Paper-Level Model Comparison", "Baboonland is harder; adaptation gives the cleanest gains.")
    rows = [
        ["Model", "Zero-shot Acc/F1", "LoRA Acc/F1", "Joint Acc/F1"],
        ["LLaVA-OV-7B", "0.338 / 0.077", "0.775 / 0.443", "-"],
        ["LLaVA-OV-0.5B", "0.418 / 0.078", "0.768 / 0.430", "-"],
        ["Gemma3-4B", "0.540 / 0.145", "0.756 / 0.405", "-"],
        ["Qwen2.5-VL-7B", "0.329 / 0.143", "0.755 / 0.394", "-"],
        ["Qwen2.5-VL-3B + inv-sqrt", "0.242 / 0.080", "0.752 / 0.374", "-"],
        ["Qwen2.5-VL-3B baseline", "0.242 / 0.080", "0.734 / 0.327", "0.759 / 0.377"],
        ["MobileVLM-1.7B", "0.174 / 0.024", "0.712 / 0.304", "-"],
        ["X3D-L / I3D-L specialist", "-", "0.671 / - ; 0.630 / -", "-"],
    ]
    add_table(slide, 0.55, 1.25, 12.25, 4.75, rows, [3.35, 2.5, 3.05, 2.3], 9.2)
    add_footer(slide, 6)

    # 7
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_title(slide, "Cross-Dataset Transfer", "Direct transfer is useful but much weaker than joint training.")
    rows = [
        ["Direction", "Source LoRA", "Acc", "Macro-F1"],
        ["Baboonland -> KABR", "LLaVA-OV-7B", "0.652", "0.256"],
        ["Baboonland -> KABR", "Qwen2.5-VL-7B", "0.562", "0.261"],
        ["Baboonland -> KABR", "LLaVA-OV-0.5B", "0.517", "0.181"],
        ["Baboonland -> KABR", "MobileVLM-1.7B", "0.447", "0.179"],
        ["Baboonland -> KABR", "Qwen2.5-VL-3B", "0.385", "0.114"],
        ["KABR -> Baboonland", "Qwen2.5-VL-3B, 8-frame", "0.397", "0.155"],
        ["KABR -> Baboonland", "Qwen2.5-VL-3B, 4-frame", "0.298", "0.129"],
        ["KABR -> Baboonland", "Qwen2.5-VL-3B, structured", "0.191", "0.072"],
    ]
    add_table(slide, 0.75, 1.25, 11.8, 4.6, rows, [2.7, 5.0, 1.5, 1.5], 9.5)
    add_footer(slide, 7)

    # 8
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_title(slide, "Class-Collapse-Aware Training", "Inverse-square-root class weighting is the best current Baboonland mitigation.")
    rows = [
        ["Variant", "Training change", "Acc", "F1", "Delta"],
        ["B1 baseline", "Standard CE", "0.734", "0.327", "-"],
        ["B2 inv-sqrt CE", "Weight = sqrt(max_count / class_count)", "0.752", "0.374", "+0.018 acc, +0.047 F1"],
        ["B3 focal loss", "Focal loss, gamma=2", "0.715", "0.302", "-0.018 acc, -0.025 F1"],
        ["B4 balanced sampler", "WeightedRandomSampler", "0.616", "0.349", "-0.117 acc, +0.022 F1"],
    ]
    add_table(slide, 0.65, 1.25, 12.0, 2.85, rows, [2.2, 4.55, 1.0, 1.0, 2.65], 9.8)
    rows = [
        ["Why this matters", "Macro-F1 exposes whether rare behavior classes are learned."],
        ["Best current fix", "B2 improves accuracy and macro-F1 with no inference-time cost."],
    ]
    add_table(slide, 0.65, 4.75, 12.0, 0.95, rows, [2.8, 8.6], 10)
    add_footer(slide, 8)

    # 9
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_title(slide, "Joint Cross-Dataset LoRA", "One adapter can match or exceed per-dataset adapters.")
    rows = [
        ["Training setup", "KABR Acc/F1", "Baboonland Acc/F1"],
        ["Joint LoRA, no class weighting", "0.863 / 0.514", "0.759 / 0.377"],
        ["Joint LoRA, inv-sqrt class weighting", "Evaluation pending", "Evaluation pending"],
        ["KABR-only LoRA reference", "0.853 / 0.485", "-"],
        ["Baboonland-only LoRA reference", "-", "0.734 / 0.327"],
        ["Baboonland-only LoRA evaluated on KABR", "0.385 / 0.114", "-"],
        ["KABR-only LoRA evaluated on Baboonland", "-", "0.397 / 0.155"],
    ]
    add_table(slide, 0.85, 1.3, 11.55, 3.65, rows, [4.5, 3.0, 3.0], 10)
    bullet_box(
        slide,
        1.0,
        5.35,
        11.2,
        0.8,
        ["Interpretation: joint training is much stronger than transfer-only and is a central paper contribution."],
        font_size=15,
    )
    add_footer(slide, 9)

    # 10
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_title(slide, "Small-VLM Zero-Shot Screening", "Useful resource-aware comparison points, but still below adapted VLMs.")
    rows = [
        ["Model", "KABR Acc/F1", "KABR parsed", "Bbl Acc/F1", "Bbl parsed", "Combined Acc", "Combined F1", "Status"],
        ["MiniCPM-V 4.6", "0.6523 / 0.2841", "0.9739", "0.3767 / 0.1022", "1.0000", "0.4845", "0.1932", "complete"],
        ["LFM2.5-VL-1.6B", "0.4361 / 0.1613", "0.9764", "0.5092 / 0.1339", "1.0000", "0.4806", "0.1476", "complete"],
        ["Qwen3.5-0.8B", "-", "-", "-", "-", "-", "-", "missing"],
        ["Gemma4-E2B-it", "-", "-", "-", "-", "-", "-", "missing"],
    ]
    add_table(slide, 0.45, 1.35, 12.45, 3.35, rows, [2.0, 1.65, 1.25, 1.65, 1.25, 1.25, 1.25, 1.25], 8.4)
    add_footer(slide, 10)

    # 11
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, WHITE)
    add_title(slide, "Comprehensive Takeaways", "The stronger story is a progression from zero-shot to efficient adaptation.")
    rows = [
        ["Claim", "Evidence"],
        ["Zero-shot alone is insufficient", "Best KABR zero-shot in older ladder is below X3D-M; Baboonland zero-shot macro-F1 is very low."],
        ["LoRA is the main performance jump", "Qwen3B LoRA reaches 0.853-0.854 KABR accuracy; top adapted Baboonland rows reach 0.775 accuracy."],
        ["Macro-F1 is essential", "Rare classes and class collapse make accuracy-only reporting misleading."],
        ["Class-aware loss helps", "Inv-sqrt CE improves Baboonland Qwen3B LoRA by +0.047 macro-F1."],
        ["Joint training is compelling", "A single Qwen3B adapter reaches 0.863/0.514 on KABR and 0.759/0.377 on Baboonland."],
        ["Next paper angle", "Use small VLMs as screening; use NAS-style adapter/input search for the stronger resource-aware contribution."],
    ]
    add_table(slide, 0.7, 1.25, 11.9, 4.85, rows, [3.0, 8.3], 9.8)
    add_footer(slide, 11)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
