from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


OUT = Path("artifacts/presentations/current_status_plan_tables_2026-05-20.pptx")

NAVY = RGBColor(28, 43, 67)
TEAL = RGBColor(0, 132, 132)
GOLD = RGBColor(202, 143, 45)
GRAY = RGBColor(88, 96, 105)
LIGHT = RGBColor(245, 247, 250)
WHITE = RGBColor(255, 255, 255)


def set_bg(slide, color=WHITE):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_title(slide, title: str, subtitle: str | None = None):
    box = slide.shapes.add_textbox(Inches(0.55), Inches(0.32), Inches(12.25), Inches(0.58))
    p = box.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(25)
    p.font.bold = True
    p.font.color.rgb = NAVY
    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.58), Inches(0.88), Inches(11.9), Inches(0.35))
        sp = sub.text_frame.paragraphs[0]
        sp.text = subtitle
        sp.font.size = Pt(11)
        sp.font.color.rgb = GRAY


def add_footer(slide, idx: int):
    line = slide.shapes.add_shape(1, Inches(0.55), Inches(7.02), Inches(12.25), Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(220, 225, 232)
    line.line.color.rgb = RGBColor(220, 225, 232)
    box = slide.shapes.add_textbox(Inches(11.85), Inches(7.08), Inches(0.9), Inches(0.25))
    p = box.text_frame.paragraphs[0]
    p.text = str(idx)
    p.font.size = Pt(9)
    p.font.color.rgb = GRAY
    p.alignment = PP_ALIGN.RIGHT


def bullet_box(slide, x, y, w, h, bullets, font_size=16):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    for i, text in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.font.size = Pt(font_size)
        p.font.color.rgb = NAVY
        p.space_after = Pt(7)


def card(slide, x, y, w, h, title, body, accent=TEAL):
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT
    shape.line.color.rgb = RGBColor(218, 224, 232)
    bar = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(0.08), Inches(h))
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent
    bar.line.color.rgb = accent
    title_box = slide.shapes.add_textbox(Inches(x + 0.22), Inches(y + 0.16), Inches(w - 0.35), Inches(0.34))
    tp = title_box.text_frame.paragraphs[0]
    tp.text = title
    tp.font.size = Pt(14)
    tp.font.bold = True
    tp.font.color.rgb = NAVY
    body_box = slide.shapes.add_textbox(Inches(x + 0.22), Inches(y + 0.56), Inches(w - 0.35), Inches(h - 0.65))
    bp = body_box.text_frame.paragraphs[0]
    bp.text = body
    bp.font.size = Pt(12)
    bp.font.color.rgb = GRAY


def add_table(slide, x, y, w, h, rows, col_widths=None, font_size=9.5):
    table_shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h))
    table = table_shape.table
    if col_widths:
        for i, cw in enumerate(col_widths):
            table.columns[i].width = Inches(cw)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(val)
            cell.margin_left = Inches(0.04)
            cell.margin_right = Inches(0.04)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(font_size)
                p.font.color.rgb = NAVY if r == 0 else RGBColor(45, 52, 62)
                if r == 0:
                    p.font.bold = True
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = RGBColor(224, 235, 240)
            else:
                cell.fill.fore_color.rgb = WHITE if r % 2 else RGBColor(248, 250, 252)
    return table_shape


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # 1
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, RGBColor(248, 250, 252))
    title = slide.shapes.add_textbox(Inches(0.7), Inches(0.95), Inches(8.6), Inches(1.1))
    p = title.text_frame.paragraphs[0]
    p.text = "Wildlife Behavior VLMs"
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = NAVY
    subtitle = slide.shapes.add_textbox(Inches(0.75), Inches(1.92), Inches(8.8), Inches(0.6))
    sp = subtitle.text_frame.paragraphs[0]
    sp.text = "Current status, benchmark construction, evaluation, and plan"
    sp.font.size = Pt(20)
    sp.font.color.rgb = GRAY
    rows = [
        ["Topic", "What this deck shows"],
        ["Background", "Why wildlife behavior clips are a hard, imbalanced recognition task"],
        ["Dataset", "How frame annotations become clip-level JSONL benchmarks"],
        ["Evaluation", "How VLM text outputs become fixed-label metrics"],
        ["Results", "Adapted VLMs, small VLM zero-shot results, ablations, and plan"],
    ]
    add_table(slide, 0.8, 3.05, 7.6, 2.25, rows, [2.2, 5.1], 11)
    card(slide, 9.25, 1.25, 3.15, 1.2, "Date", "2026-05-20", TEAL)
    card(slide, 9.25, 2.85, 3.15, 1.2, "Datasets", "KABR-Clip\nBaboonland-Clip", GOLD)
    card(slide, 9.25, 4.45, 3.15, 1.2, "Focus", "Status + plan\nwith tables", NAVY)
    add_footer(slide, 1)

    # 2
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Background", "The project compares adapted VLMs against specialist video models for wildlife behavior recognition.")
    rows = [
        ["Item", "Description", "Why it matters"],
        ["Task", "Classify a short drone-video clip into a behavior label.", "Useful for scalable wildlife monitoring."],
        ["Traditional baseline", "Train X3D/I3D/SlowFast-style video models from scratch.", "Strong but data-hungry and task-specific."],
        ["VLM approach", "Prompt a vision-language model with sampled frames and labels.", "Uses broad visual priors and language label grounding."],
        ["Adaptation", "Train small LoRA adapters instead of full model fine-tuning.", "Cheap route to dataset-specific behavior recognition."],
        ["Core challenge", "Severe class imbalance, especially in Baboonland.", "Accuracy can hide rare-class failure; macro-F1 is essential."],
    ]
    add_table(slide, 0.7, 1.35, 11.9, 4.55, rows, [2.0, 5.25, 4.05], 10)
    add_footer(slide, 2)

    # 3
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Dataset Statistics", "Both datasets are represented as clip-level behavior classification benchmarks.")
    rows = [
        ["Dataset", "Source/domain", "Classes", "Train clips", "Val clips", "Imbalance"],
        ["KABR-Clip", "Drone footage of Kenyan wildlife", "8", "11,003", "3,761", "Moderate"],
        ["Baboonland-Clip", "Drone footage of baboons", "13", "20,130", "6,198", "Severe"],
        ["Joint train", "KABR + Baboonland union", "Native label spaces", "31,133", "-", "Mixed"],
    ]
    add_table(slide, 0.7, 1.35, 11.9, 1.85, rows, [2.0, 3.45, 1.55, 1.65, 1.45, 1.5], 10.5)
    rows = [
        ["Processed file", "Use"],
        ["data/processed/kabr_behavior_train.jsonl", "KABR LoRA/video train split"],
        ["data/processed/kabr_behavior_val_cleaned.jsonl", "KABR zero-shot / LoRA / video validation"],
        ["data/processed/baboonland_behavior_train.jsonl", "Baboonland LoRA/video train split"],
        ["data/processed/baboonland_behavior_val.jsonl", "Baboonland zero-shot / LoRA / video validation"],
        ["data/processed/joint_kabr_bbl_behavior_train.jsonl", "Joint cross-dataset LoRA train split"],
    ]
    add_table(slide, 0.7, 3.75, 11.9, 2.45, rows, [5.1, 6.2], 9.5)
    add_footer(slide, 3)

    # 4
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Dataset Construction Pipeline", "Raw frame annotations are converted into reusable clip manifests.")
    rows = [
        ["Step", "Input", "Operation", "Output"],
        ["1", "Frame-level / span labels", "Read video id, frame interval, and behavior id/name.", "Annotation records"],
        ["2", "Annotation records", "Group contiguous same-label runs per original video.", "Single-label temporal clips"],
        ["3", "Temporal clips", "Sample frames uniformly; store paths and clip metadata.", "Clip examples"],
        ["4", "Clip examples", "Split/export into JSONL manifests.", "Train/val benchmark files"],
        ["5", "JSONL manifests", "Feed same examples to VLM and video baseline code.", "Comparable metrics"],
    ]
    add_table(slide, 0.7, 1.35, 11.9, 3.25, rows, [0.7, 3.05, 4.25, 3.25], 10)
    bullet_box(
        slide,
        0.9,
        5.15,
        11.3,
        0.8,
        ["Key control: all model families evaluate on the same derived clips and labels."],
        font_size=16,
    )
    add_footer(slide, 4)

    # 5
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Evaluation Regimes", "The same validation manifests support zero-shot, LoRA, transfer, joint training, and specialist baselines.")
    rows = [
        ["Regime", "Training data", "Validation data", "Model output", "Main metrics"],
        ["Zero-shot VLM", "None", "Target val split", "Free text label", "Acc, macro-F1, parsed rate"],
        ["In-domain LoRA", "Target train split", "Target val split", "Free text label", "Acc, macro-F1"],
        ["Cross-dataset transfer", "Other dataset train split", "Target val split", "Free text label", "Acc, macro-F1"],
        ["Joint LoRA", "KABR + Baboonland train", "Both val splits", "Native label text", "Per-dataset Acc/F1"],
        ["Video baseline", "Target train split", "Target val split", "Class logits", "Acc, macro-F1 when available"],
    ]
    add_table(slide, 0.7, 1.35, 11.9, 3.55, rows, [2.15, 2.55, 2.3, 2.0, 2.25], 9.5)
    rows = [
        ["Evaluator artifact", "Purpose"],
        ["predictions.jsonl", "Per-clip gold label, raw text prediction, parsed label"],
        ["predictions.jsonl.metrics.json", "Accuracy, macro-F1, parsed rate, class metrics, distributions"],
        ["small_vlm_summary.md/csv", "Aggregated small-VLM comparison table"],
    ]
    add_table(slide, 0.7, 5.25, 11.9, 1.0, rows, [4.4, 6.9], 9.5)
    add_footer(slide, 5)

    # 6
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Current Project Status", "The main result is mature; current additions strengthen the resource-aware story.")
    rows = [
        ["Phase", "Status", "Evidence / next action"],
        ["Baseline comparison", "Complete", "Zero-shot VLMs, LoRA VLMs, transfer, X3D/I3D baselines"],
        ["Class-collapse-aware training", "Complete", "Inv-sqrt class-weighted CE is best among tested fixes"],
        ["Joint cross-dataset LoRA", "Partly complete", "A1 strong; A3 eval pending in earlier handoff"],
        ["Small VLM screening", "In progress", "MiniCPM and LFM full eval complete; Qwen/Gemma pending"],
        ["NAS/resource-aware search", "Planned", "Implementation plan exists; target-module config still needed"],
        ["Figures/tables", "Partly complete", "Need final paper tables, Pareto plot, class-collapse diagnostics"],
    ]
    add_table(slide, 0.7, 1.35, 11.9, 4.55, rows, [3.0, 2.05, 6.25], 10)
    add_footer(slide, 6)

    # 7
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Headline Adapted-VLM Results", "LoRA-adapted VLMs are the strongest evidence for the current paper.")
    rows = [
        ["Dataset", "Model / setup", "Acc", "Macro-F1", "Reference baseline"],
        ["KABR", "Gemma3-4B LoRA", "0.864", "0.515", "X3D-L: 0.797 / 0.520"],
        ["KABR", "LLaVA-OV-7B LoRA", "0.863", "0.523", "I3D-L: 0.814 / -"],
        ["KABR", "Qwen2.5-VL-3B joint A1", "0.863", "0.514", "In-domain Qwen3B: 0.853 / 0.485"],
        ["Baboonland", "LLaVA-OV-7B LoRA", "0.775", "0.443", "X3D-L: 0.671 / -"],
        ["Baboonland", "LLaVA-OV-0.5B LoRA", "0.768", "0.430", "I3D-L: 0.630 / -"],
        ["Baboonland", "Qwen2.5-VL-3B joint A1", "0.759", "0.377", "In-domain B1: 0.734 / 0.327"],
    ]
    add_table(slide, 0.7, 1.35, 11.9, 4.6, rows, [1.65, 3.55, 1.0, 1.25, 3.85], 10)
    add_footer(slide, 7)

    # 8
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Class Collapse And Mitigation", "Baboonland exposes rare-class failure; inv-sqrt class weighting is the best current fix.")
    rows = [
        ["Variant", "Training change", "Acc", "Macro-F1", "Delta vs B1"],
        ["B1 baseline LoRA", "Standard CE", "0.734", "0.327", "-"],
        ["B2 inv-sqrt CE", "Weight = sqrt(max_count / class_count)", "0.752", "0.374", "+0.018 acc, +0.047 F1"],
        ["B3 focal loss", "Focal loss gamma=2", "0.715", "0.302", "-0.018 acc, -0.025 F1"],
        ["B4 class-balanced sampler", "WeightedRandomSampler", "0.616", "0.349", "-0.117 acc, +0.022 F1"],
    ]
    add_table(slide, 0.7, 1.35, 11.9, 2.8, rows, [2.45, 4.45, 1.0, 1.25, 2.15], 10)
    rows = [
        ["Claim", "Evidence"],
        ["Accuracy alone is insufficient", "Zero-shot models can overpredict common labels while missing rare behaviors."],
        ["Macro-F1 is central", "It reveals whether minority classes are actually learned."],
        ["Best current fix", "B2 improves both accuracy and macro-F1 with no inference-time cost."],
    ]
    add_table(slide, 0.7, 4.7, 11.9, 1.3, rows, [3.0, 8.3], 10)
    add_footer(slide, 8)

    # 9
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Joint Training And Transfer", "One joint adapter closes much of the gap between transfer and in-domain training.")
    rows = [
        ["Setup", "KABR Acc/F1", "Baboonland Acc/F1", "Interpretation"],
        ["Joint A1, no weighting", "0.863 / 0.514", "0.759 / 0.377", "One adapter works well on both datasets"],
        ["Joint A3, inv-sqrt", "Eval pending", "Eval pending", "Expected to combine A + B ideas"],
        ["In-domain KABR LoRA", "0.853 / 0.485", "-", "Per-dataset reference"],
        ["In-domain Baboonland B1", "-", "0.734 / 0.327", "Standard CE reference"],
        ["In-domain Baboonland B2", "-", "0.752 / 0.374", "Best class-weighted reference"],
        ["KABR-only to Baboonland", "-", "0.397 / 0.155", "Transfer-only remains weak"],
    ]
    add_table(slide, 0.7, 1.35, 11.9, 4.5, rows, [3.0, 2.05, 2.3, 4.0], 9.5)
    add_footer(slide, 9)

    # 10
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Small-VLM Zero-Shot Results", "MiniCPM and LFM are complete; Qwen/Gemma are still pending.")
    rows = [
        ["Model", "KABR Acc/F1", "KABR parsed", "Bbl Acc/F1", "Bbl parsed", "Combined Acc", "Combined F1", "Status"],
        ["MiniCPM-V 4.6", "0.6523 / 0.2841", "0.9739", "0.3767 / 0.1022", "1.0000", "0.4845", "0.1932", "complete"],
        ["LFM2.5-VL-1.6B", "0.4361 / 0.1613", "0.9764", "0.5092 / 0.1339", "1.0000", "0.4806", "0.1476", "complete"],
        ["Qwen3.5-0.8B", "-", "-", "-", "-", "-", "-", "missing"],
        ["Gemma4-E2B-it", "-", "-", "-", "-", "-", "-", "missing"],
    ]
    add_table(slide, 0.45, 1.35, 12.45, 3.35, rows, [2.0, 1.65, 1.25, 1.65, 1.25, 1.25, 1.25, 1.25], 8.4)
    rows = [
        ["Takeaway", "MiniCPM wins KABR and combined macro-F1; LFM wins Baboonland accuracy."],
        ["Use in paper", "Good resource-aware zero-shot comparison, but LoRA/search remains the stronger main path."],
    ]
    add_table(slide, 0.7, 5.15, 11.9, 0.85, rows, [2.2, 9.1], 9.5)
    add_footer(slide, 10)

    # 11
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Planned NAS-Style Search", "Search the adapter/input recipe around an existing VLM, not the full VLM architecture.")
    rows = [
        ["Axis", "Values", "Runs", "Why search it?"],
        ["Dataset", "Baboonland first", "1", "Severe imbalance makes macro-F1 and class collapse visible."],
        ["Backbone", "Qwen2.5-VL-3B", "1", "Existing infrastructure and B1/B2/A1 references."],
        ["Frames per clip", "4, 8", "2", "Temporal evidence vs input/inference cost."],
        ["LoRA rank", "8, 16", "2", "Adapter capacity vs trainable parameters."],
        ["LoRA targets", "attention only; attention+MLP", "2", "Where behavior adaptation should happen."],
        ["Loss", "CE; inv-sqrt CE", "2", "Whether class-aware loss remains best in the grid."],
        ["Total", "2 x 2 x 2 x 2", "16", "Compact enough to run, rich enough for Pareto analysis."],
    ]
    add_table(slide, 0.7, 1.35, 11.9, 4.8, rows, [2.0, 3.0, 1.0, 5.3], 9.5)
    add_footer(slide, 11)

    # 12
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Immediate Execution Plan", "The next phase converts status into a paper-ready resource-aware story.")
    rows = [
        ["Priority", "Task", "Output"],
        ["1", "Resolve Qwen3.5-0.8B loader issue; decide whether Gemma4 is accessible.", "Complete small-VLM status table"],
        ["2", "Add `training.lora_target_modules` to config/model creation.", "NAS target-module search enabled"],
        ["3", "Generate 16 Baboonland NAS configs.", "Config grid under configs/nas/"],
        ["4", "Smoke-test two representative configs.", "Fast validation before full queue"],
        ["5", "Launch full NAS grid and evals.", "Accuracy/F1/cost table"],
        ["6", "Build figures.", "Pareto frontier + class-collapse diagnostics"],
    ]
    add_table(slide, 0.7, 1.35, 11.9, 4.15, rows, [1.0, 7.0, 3.3], 10)
    card(slide, 0.9, 5.85, 11.3, 0.65, "Recommendation", "Use MiniCPM/LFM as zero-shot resource-aware evidence; use NAS LoRA search as the main new contribution.", TEAL)
    add_footer(slide, 12)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
