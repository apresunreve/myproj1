from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


OUT = Path("artifacts/presentations/current_status_and_plan_2026-05-20.pptx")

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
    box = slide.shapes.add_textbox(Inches(0.55), Inches(0.35), Inches(12.25), Inches(0.6))
    p = box.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = NAVY
    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.58), Inches(0.93), Inches(11.8), Inches(0.35))
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


def bullet_box(slide, x, y, w, h, bullets, font_size=17, color=NAVY):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    for i, text in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.level = 0
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.space_after = Pt(8)
    return box


def card(slide, x, y, w, h, title, body, accent=TEAL):
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT
    shape.line.color.rgb = RGBColor(218, 224, 232)
    bar = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(0.08), Inches(h))
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent
    bar.line.color.rgb = accent
    title_box = slide.shapes.add_textbox(Inches(x + 0.22), Inches(y + 0.16), Inches(w - 0.35), Inches(0.35))
    tp = title_box.text_frame.paragraphs[0]
    tp.text = title
    tp.font.size = Pt(14)
    tp.font.bold = True
    tp.font.color.rgb = NAVY
    body_box = slide.shapes.add_textbox(Inches(x + 0.22), Inches(y + 0.58), Inches(w - 0.35), Inches(h - 0.7))
    bp = body_box.text_frame.paragraphs[0]
    bp.text = body
    bp.font.size = Pt(12)
    bp.font.color.rgb = GRAY
    return shape


def add_table(slide, x, y, w, h, rows, col_widths=None, font_size=10):
    table_shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h))
    table = table_shape.table
    if col_widths:
        for i, cw in enumerate(col_widths):
            table.columns[i].width = Inches(cw)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = val
            cell.margin_left = Inches(0.05)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(font_size)
            para.font.color.rgb = NAVY if r == 0 else RGBColor(46, 52, 60)
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(224, 235, 240)
                para.font.bold = True
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if r % 2 else RGBColor(248, 250, 252)
    return table_shape


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # 1
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, RGBColor(248, 250, 252))
    title = slide.shapes.add_textbox(Inches(0.7), Inches(1.05), Inches(8.4), Inches(1.1))
    p = title.text_frame.paragraphs[0]
    p.text = "Wildlife Behavior VLMs"
    p.font.size = Pt(42)
    p.font.bold = True
    p.font.color.rgb = NAVY
    subtitle = slide.shapes.add_textbox(Inches(0.75), Inches(2.0), Inches(8.8), Inches(0.8))
    sp = subtitle.text_frame.paragraphs[0]
    sp.text = "Current status and near-term plan"
    sp.font.size = Pt(22)
    sp.font.color.rgb = GRAY
    bullet_box(
        slide,
        0.8,
        3.05,
        7.8,
        2.1,
        [
            "Goal: robust clip-level behavior recognition from drone video.",
            "Main thread: VLM adaptation beats specialist video baselines.",
            "Next thread: resource-aware model and adapter choices.",
        ],
        font_size=18,
    )
    card(slide, 9.25, 1.25, 3.2, 1.25, "Datasets", "KABR-Clip: 8 classes\nBaboonland-Clip: 13 classes", TEAL)
    card(slide, 9.25, 2.85, 3.2, 1.25, "Current focus", "Small VLM zero-shot\nNAS-style adapter search", GOLD)
    card(slide, 9.25, 4.45, 3.2, 1.25, "Date", "2026-05-20", NAVY)
    add_footer(slide, 1)

    # 2
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Background And Motivation", "The task is behavior recognition from short wildlife video clips, where labels are scarce and class imbalance is severe.")
    card(slide, 0.7, 1.35, 3.75, 1.35, "Problem", "Given a short drone-video clip, predict the animal behavior label.", TEAL)
    card(slide, 4.85, 1.35, 3.75, 1.35, "Prior default", "Train specialist video models such as X3D, I3D, or SlowFast from scratch.", GOLD)
    card(slide, 9.0, 1.35, 3.4, 1.35, "This project", "Use general VLMs and adapt them cheaply with LoRA.", NAVY)
    bullet_box(
        slide,
        0.9,
        3.3,
        11.6,
        2.4,
        [
            "Why VLMs: they bring broad visual priors and language-conditioned label selection.",
            "Why adaptation: zero-shot VLMs are brittle on rare behavior classes, but LoRA is cheap.",
            "Why this matters: wildlife datasets are small, imbalanced, and expensive to annotate.",
        ],
        font_size=18,
    )
    add_footer(slide, 2)

    # 3
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Dataset Construction", "Both datasets are converted into the same clip-level behavior-recognition format.")
    rows = [
        ["Step", "What happens", "Output"],
        ["1. Start", "Charades-style frame annotations with video id, frame span, and behavior label.", "Raw label table"],
        ["2. Segment", "Group contiguous same-label runs within each original video.", "Single-label clips"],
        ["3. Sample", "Sample frames uniformly from each clip, keeping clip id and label metadata.", "Frame paths + label"],
        ["4. Export", "Write train/val JSONL manifests consumed by VLM and video baselines.", "Canonical benchmark files"],
    ]
    add_table(slide, 0.7, 1.28, 11.9, 2.65, rows, [1.35, 7.05, 2.85], 10)
    rows = [
        ["Dataset", "Domain", "Classes", "Train", "Val", "Imbalance"],
        ["KABR-Clip", "Kenyan wildlife drone video", "8", "11,003", "3,761", "moderate"],
        ["Baboonland-Clip", "Baboon drone video", "13", "20,130", "6,198", "severe"],
    ]
    add_table(slide, 0.7, 4.4, 11.9, 1.15, rows, [2.1, 4.0, 1.25, 1.35, 1.35, 1.8], 10)
    bullet_box(
        slide,
        0.9,
        5.85,
        11.3,
        0.65,
        ["Important: VLMs and specialist video baselines evaluate on the same derived clip manifests."],
        font_size=15,
    )
    add_footer(slide, 3)

    # 4
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Evaluation Protocol", "All models produce text; the evaluator maps the text back to the fixed behavior label set.")
    card(slide, 0.7, 1.25, 3.65, 1.35, "Inputs", "8 sampled frames per clip\nDataset-specific label list", TEAL)
    card(slide, 4.85, 1.25, 3.65, 1.35, "Prompt", "Ask the VLM to choose the best behavior label from the allowed set.", GOLD)
    card(slide, 9.0, 1.25, 3.4, 1.35, "Parsing", "Fuzzy label extraction maps free text to a class or marks it unparsed.", NAVY)
    bullet_box(
        slide,
        0.9,
        3.25,
        5.7,
        2.5,
        [
            "Zero-shot: no task training; run the model directly on validation clips.",
            "LoRA: train a lightweight adapter, then evaluate with the same prompt/parser.",
            "Video baselines: train X3D/I3D from scratch on the same clip labels.",
        ],
        font_size=16,
    )
    bullet_box(
        slide,
        7.0,
        3.25,
        5.3,
        2.5,
        [
            "Primary metrics: accuracy and macro-F1.",
            "Sanity metric: parsed rate.",
            "Diagnostics: per-class F1 and prediction distribution for class collapse.",
        ],
        font_size=16,
    )
    add_footer(slide, 4)

    # 5
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Where The Project Stands", "The core paper story is already supported; the current work sharpens the resource-aware angle.")
    card(slide, 0.7, 1.35, 3.8, 1.35, "1. Adapted VLMs", "LoRA-adapted VLMs outperform X3D/I3D specialist video baselines on accuracy.", TEAL)
    card(slide, 4.85, 1.35, 3.8, 1.35, "2. Failure mode", "Zero-shot models show class collapse: decent accuracy can hide poor macro-F1.", GOLD)
    card(slide, 9.0, 1.35, 3.4, 1.35, "3. Fixes", "Inv-sqrt class weighting improves minority behavior recognition.", NAVY)
    card(slide, 0.7, 3.35, 3.8, 1.35, "4. Joint training", "One joint LoRA adapter matches or exceeds in-domain adapters on both datasets.", TEAL)
    card(slide, 4.85, 3.35, 3.8, 1.35, "5. Small VLMs", "MiniCPM-V 4.6 and LFM2.5-VL-1.6B full zero-shot evals are complete.", GOLD)
    card(slide, 9.0, 3.35, 3.4, 1.35, "6. Next", "Run a compact adapter/input search for Pareto-efficient recipes.", NAVY)
    add_footer(slide, 5)

    # 6
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Headline Adapted-VLM Results", "LoRA adaptation gives the strongest current evidence for the main paper.")
    rows = [
        ["Dataset", "Strong adapted VLM", "Acc / macro-F1", "Video baseline"],
        ["KABR", "Gemma3-4B LoRA", "0.864 / 0.515", "X3D-L: 0.797 / 0.520"],
        ["KABR", "LLaVA-OV-7B LoRA", "0.863 / 0.523", "I3D-L: 0.814 / -"],
        ["Baboonland", "LLaVA-OV-7B LoRA", "0.775 / 0.443", "X3D-L: 0.671 / -"],
        ["Baboonland", "LLaVA-OV-0.5B LoRA", "0.768 / 0.430", "I3D-L: 0.630 / -"],
        ["Joint", "Qwen2.5-VL-3B A1", "KABR 0.863 / 0.514\nBbl 0.759 / 0.377", "Single adapter"],
    ]
    add_table(slide, 0.7, 1.35, 11.9, 3.4, rows, [1.5, 3.2, 3.1, 3.1], 11)
    bullet_box(
        slide,
        0.85,
        5.15,
        11.4,
        1.1,
        [
            "Takeaway: the adapted VLM story is strong enough to anchor the paper.",
            "Macro-F1 remains the key metric for rare/imbalanced behaviors.",
        ],
        font_size=16,
    )
    add_footer(slide, 6)

    # 7
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "New Small-VLM Zero-Shot Results", "Completed full evals for MiniCPM-V 4.6 and LFM2.5-VL-1.6B.")
    rows = [
        ["Model", "KABR Acc/F1", "Bbl Acc/F1", "Combined Acc", "Status"],
        ["MiniCPM-V 4.6", "0.652 / 0.284", "0.377 / 0.102", "0.485", "complete"],
        ["LFM2.5-VL-1.6B", "0.436 / 0.161", "0.509 / 0.134", "0.481", "complete"],
        ["Qwen3.5-0.8B", "-", "-", "-", "pending/blocker"],
        ["Gemma4-E2B-it", "-", "-", "-", "pending"],
    ]
    add_table(slide, 0.7, 1.35, 11.9, 2.7, rows, [2.7, 2.2, 2.2, 1.8, 2.0], 11)
    card(slide, 0.9, 4.55, 3.7, 1.2, "MiniCPM", "Best KABR zero-shot among the new small candidates so far.", TEAL)
    card(slide, 4.85, 4.55, 3.7, 1.2, "LFM", "Better Baboonland accuracy, but lower macro-F1 than MiniCPM overall.", GOLD)
    card(slide, 8.8, 4.55, 3.45, 1.2, "Interpretation", "Zero-shot remains useful for screening, not final performance.", NAVY)
    add_footer(slide, 7)

    # 8
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "What We Learned", "The data points to a paper centered on adaptation, failure modes, and efficient choices.")
    bullet_box(
        slide,
        0.9,
        1.4,
        11.6,
        4.6,
        [
            "Adaptation matters more than backbone size alone: even 0.5B LoRA beats large zero-shot rows.",
            "Accuracy alone is not enough; macro-F1 exposes class collapse on imbalanced behavior labels.",
            "Inv-sqrt class weighting is the best current low-cost fix for Baboonland imbalance.",
            "Joint LoRA is unexpectedly strong: one adapter can serve both KABR and Baboonland.",
            "Small VLM zero-shot results add a useful resource-aware comparison, but LoRA/search is still the stronger route.",
        ],
        font_size=18,
    )
    add_footer(slide, 8)

    # 9
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Planned Contribution C: Resource-Aware Search", "A compact NAS-style search around the adapter/input recipe, not full VLM architecture search.")
    rows = [
        ["Axis", "Values", "Question"],
        ["Frames", "4, 8", "How much temporal evidence is needed?"],
        ["LoRA rank", "8, 16", "How much adapter capacity is enough?"],
        ["LoRA targets", "attention only, attention+MLP", "Where should behavior adaptation live?"],
        ["Loss", "CE, inv-sqrt CE", "Does class-aware loss remain best under search?"],
    ]
    add_table(slide, 0.7, 1.35, 11.9, 2.7, rows, [2.0, 3.4, 5.9], 11)
    bullet_box(
        slide,
        0.9,
        4.45,
        11.3,
        1.3,
        [
            "First grid: Baboonland + Qwen2.5-VL-3B = 16 primary runs.",
            "Paper claim: find Pareto-efficient recipes that improve minority-class behavior recognition without increasing inference cost.",
        ],
        font_size=17,
    )
    add_footer(slide, 9)

    # 10
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "Immediate Plan", "Short path from current status to a stronger paper package.")
    bullet_box(
        slide,
        0.9,
        1.35,
        11.6,
        4.4,
        [
            "1. Finish small-VLM screening: resolve Qwen3.5-0.8B loader issue; decide whether Gemma4 is accessible/worth running.",
            "2. Implement `training.lora_target_modules` and generate 16 NAS configs.",
            "3. Smoke-test two NAS configs, then launch the full Baboonland grid.",
            "4. Aggregate accuracy, macro-F1, parsed rate, trainable parameters, and inference cost.",
            "5. Build paper figures: Pareto frontier, class-collapse diagnostics, and final comparison table.",
        ],
        font_size=17,
    )
    card(slide, 0.9, 5.85, 11.3, 0.65, "Decision point", "Use MiniCPM/LFM as zero-shot resource-aware evidence; use NAS LoRA search as the main new contribution.", TEAL)
    add_footer(slide, 10)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
