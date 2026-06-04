"""Generate project overview PPT."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── Palette ─────────────────────────────────────────────────────────────────
BG       = RGBColor(0x0D, 0x1B, 0x2A)   # dark navy
ACCENT   = RGBColor(0x00, 0xA8, 0xE8)   # sky blue
ACCENT2  = RGBColor(0xF9, 0xA8, 0x25)   # amber
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
GRAY     = RGBColor(0xB0, 0xC4, 0xD8)
GREEN    = RGBColor(0x4C, 0xAF, 0x50)
RED      = RGBColor(0xEF, 0x53, 0x50)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

# ── Helpers ──────────────────────────────────────────────────────────────────

def new_prs():
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_slide(prs):
    layout = prs.slide_layouts[6]   # totally blank
    return prs.slides.add_slide(layout)


def bg(slide, color=BG):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def box(slide, left, top, width, height,
        fill_color=None, line_color=None, line_width=Pt(0)):
    shape = slide.shapes.add_shape(
        1,   # MSO_SHAPE_TYPE.RECTANGLE
        left, top, width, height)
    shape.line.width = line_width
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = line_width if line_width else Pt(1.5)
    else:
        shape.line.fill.background()
    return shape


def txt(slide, text, left, top, width, height,
        size=Pt(18), bold=False, color=WHITE, align=PP_ALIGN.LEFT,
        wrap=True, italic=False):
    txb = slide.shapes.add_textbox(left, top, width, height)
    txb.word_wrap = wrap
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = size
    run.font.bold  = bold
    run.font.color.rgb = color
    run.font.italic = italic
    return txb


def accent_bar(slide, height=Inches(0.07)):
    """Horizontal accent bar at top."""
    box(slide, 0, 0, SLIDE_W, height, fill_color=ACCENT)


def section_header(slide, section_label, title):
    """Top-left section tag + big title."""
    accent_bar(slide)
    txt(slide, section_label.upper(),
        Inches(0.45), Inches(0.22), Inches(4), Inches(0.4),
        size=Pt(11), bold=True, color=ACCENT)
    txt(slide, title,
        Inches(0.45), Inches(0.62), Inches(12.4), Inches(1.1),
        size=Pt(36), bold=True, color=WHITE)


def bullet_column(slide, items, left, top, width, height,
                  size=Pt(17), bullet="▸ ", color=WHITE, spacing=Inches(0.05)):
    """Render a list of strings as bullets in a text box."""
    txb = slide.shapes.add_textbox(left, top, width, height)
    txb.word_wrap = True
    tf  = txb.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.space_before = spacing
        run = p.add_run()
        run.text = bullet + item
        run.font.size  = size
        run.font.color.rgb = color
    return txb


def kv_card(slide, label, value, left, top, width, height,
            label_color=ACCENT, value_color=WHITE,
            card_color=RGBColor(0x16, 0x2A, 0x40)):
    box(slide, left, top, width, height, fill_color=card_color)
    txt(slide, label, left+Inches(0.15), top+Inches(0.08),
        width-Inches(0.3), Inches(0.38),
        size=Pt(13), bold=True, color=label_color)
    txt(slide, value, left+Inches(0.15), top+Inches(0.45),
        width-Inches(0.3), height-Inches(0.55),
        size=Pt(22), bold=True, color=value_color, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════════
#  SLIDE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def slide_title(prs):
    s = blank_slide(prs)
    bg(s)
    # left accent stripe
    box(s, 0, 0, Inches(0.25), SLIDE_H, fill_color=ACCENT)
    # title
    txt(s, "Adapting Vision-Language Models\nfor Wildlife Behavior Recognition",
        Inches(0.6), Inches(1.5), Inches(8.5), Inches(2.8),
        size=Pt(40), bold=True, color=WHITE)
    txt(s, "KABR-Clip  ·  Baboonland-Clip",
        Inches(0.6), Inches(4.4), Inches(8), Inches(0.6),
        size=Pt(22), color=ACCENT)
    txt(s, "Jinhao Yan  ·  University of Utah",
        Inches(0.6), Inches(5.2), Inches(8), Inches(0.5),
        size=Pt(18), color=GRAY)
    # decorative right panel
    box(s, Inches(9.5), Inches(1.0), Inches(3.4), Inches(5.2),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    for i, line in enumerate([
        "Zero-shot VLMs → class collapse",
        "LoRA adapts VLMs effectively",
        "VLMs outperform X3D / I3D",
        "Joint LoRA closes transfer gap",
        "Class-aware loss fixes imbalance",
    ]):
        txt(s, "✓  " + line,
            Inches(9.65), Inches(1.3)+Inches(0.82)*i,
            Inches(3.1), Inches(0.7),
            size=Pt(13), color=ACCENT2 if i < 2 else WHITE)


def slide_motivation(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "1  Background & Methods", "Why Adapt VLMs for Wildlife Behavior?")

    # Two columns
    # Left: Problem
    box(s, Inches(0.4), Inches(1.85), Inches(5.8), Inches(5.2),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    txt(s, "The Problem", Inches(0.55), Inches(1.95), Inches(5.5), Inches(0.5),
        size=Pt(18), bold=True, color=ACCENT)
    bullet_column(s, [
        "Wildlife behavior recognition = clip-level action classification",
        "Historically: specialist 3D-CNNs (X3D, I3D) trained from scratch",
        "Small, expensive labeled datasets per species / location",
        "Models don't transfer across datasets or species",
    ], Inches(0.55), Inches(2.5), Inches(5.5), Inches(4.3),
       size=Pt(15.5), bullet="• ")

    # Right: Opportunity
    box(s, Inches(6.5), Inches(1.85), Inches(6.4), Inches(5.2),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    txt(s, "The Opportunity", Inches(6.65), Inches(1.95), Inches(6.1), Inches(0.5),
        size=Pt(18), bold=True, color=ACCENT2)
    bullet_column(s, [
        "General VLMs (Qwen2.5-VL, LLaVA, Gemma3) can answer visual questions",
        "LoRA fine-tuning: cheap adaptation with a small labeled set",
        "One model family → multiple species with minimal re-training",
        "Open question: does adaptation actually beat specialist video models?",
    ], Inches(6.65), Inches(2.5), Inches(6.1), Inches(4.3),
       size=Pt(15.5), bullet="• ")


def slide_prior_work(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "1  Background", "Prior Work & Our Position")

    # Mamooler box
    box(s, Inches(0.4), Inches(1.85), Inches(12.5), Inches(2.2),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    txt(s, "Closest prior work  —  Mamooler et al., COLM 2025",
        Inches(0.6), Inches(1.95), Inches(12), Inches(0.45),
        size=Pt(16), bold=True, color=ACCENT)
    bullet_column(s, [
        "Joint multi-task VLM fine-tuning (InternVL3-8B) on multiple animal behavior datasets",
        "Outperforms vision-only baselines  ·  Shows adaptation matters",
        "Does NOT: measure cross-dataset transfer, name class collapse, or compare across backbone sizes",
    ], Inches(0.6), Inches(2.45), Inches(12.1), Inches(1.5),
       size=Pt(15), bullet="• ")

    # Our additions
    txt(s, "What this work adds",
        Inches(0.4), Inches(4.25), Inches(12.5), Inches(0.45),
        size=Pt(18), bold=True, color=WHITE)

    cards = [
        ("6 VLM backbones\n0.5B – 7B", ACCENT),
        ("Named failure mode\n\"Class Collapse\"", ACCENT2),
        ("Cross-dataset\ntransfer measured", ACCENT),
        ("Specialist video\ncomparison (X3D, I3D)", ACCENT2),
        ("2 training\ncontributions (A+B)", GREEN),
    ]
    card_w = Inches(2.4)
    card_h = Inches(2.55)
    for i, (label, col) in enumerate(cards):
        lft = Inches(0.4) + i * (card_w + Inches(0.08))
        box(s, lft, Inches(4.72), card_w, card_h,
            fill_color=RGBColor(0x16, 0x2A, 0x40))
        txt(s, label, lft+Inches(0.1), Inches(4.82), card_w-Inches(0.2), card_h-Inches(0.2),
            size=Pt(15.5), bold=True, color=col, align=PP_ALIGN.CENTER)


def slide_datasets(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "1  Background & Methods", "Two Drone-Video Wildlife Behavior Datasets")

    # KABR card
    box(s, Inches(0.4), Inches(1.85), Inches(6.1), Inches(5.25),
        fill_color=RGBColor(0x0A, 0x1E, 0x3A))
    box(s, Inches(0.4), Inches(1.85), Inches(6.1), Inches(0.45),
        fill_color=ACCENT)
    txt(s, "KABR-Clip", Inches(0.55), Inches(1.88), Inches(5.8), Inches(0.38),
        size=Pt(17), bold=True, color=BG)
    bullet_column(s, [
        "Kenyan wildlife: giraffes, plains zebras, Grevy's zebras",
        "Drone footage (aerial view)",
        "8 behavior classes",
        "11,003 train clips  /  3,761 val clips",
        "Moderate class imbalance",
        "Classes: Walk · Graze · Browse · Head Up\n    Auto-Groom · Trot · Run · Occluded",
    ], Inches(0.55), Inches(2.38), Inches(5.7), Inches(4.5),
       size=Pt(14.5), bullet="• ")

    # Baboonland card
    box(s, Inches(6.85), Inches(1.85), Inches(6.1), Inches(5.25),
        fill_color=RGBColor(0x1A, 0x14, 0x00))
    box(s, Inches(6.85), Inches(1.85), Inches(6.1), Inches(0.45),
        fill_color=ACCENT2)
    txt(s, "Baboonland-Clip", Inches(7.0), Inches(1.88), Inches(5.8), Inches(0.38),
        size=Pt(17), bold=True, color=BG)
    bullet_column(s, [
        "Baboons (drone footage from South Africa)",
        "13 behavior classes (social + locomotion)",
        "20,130 train clips  /  6,198 val clips",
        "Severe class imbalance  (max:min ≈ 5,000:1)\n   e.g. Sitting/Standing 9,843  vs  Mounting 63",
        "Classes include: Walking/Running · Sitting/Standing · Fighting/Playing · Self-Grooming · Being Groomed · Grooming Somebody · Mutual Grooming · Infant-Carrying · Foraging · Drinking · Mounting · Sleeping · Occluded",
    ], Inches(7.0), Inches(2.38), Inches(5.7), Inches(4.5),
       size=Pt(14.5), bullet="• ")

    # built in-house badge
    box(s, Inches(8.5), Inches(6.62), Inches(4.1), Inches(0.38),
        fill_color=RGBColor(0x2A, 0x1A, 0x00))
    txt(s, "★  Clip manifests built in-house this project",
        Inches(8.6), Inches(6.63), Inches(4.0), Inches(0.35),
        size=Pt(12), color=ACCENT2, bold=True)


def slide_dataset_construction(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "1  Background & Methods", "How We Build Clips from Frame Annotations")

    # Pipeline boxes
    stages = [
        ("Raw frames\n(Charades-style CSV)",  "original_video_id · frame_id · label_id · rel_path", ACCENT),
        ("Stage 1 — Clip derivation",          "Group contiguous same-label frames per video\n→ one clip = one behavior label\nKABR: 11,003 train / 3,761 val\nBbl:  20,130 train / 6,198 val", ACCENT),
        ("Stage 2 — VLM-ready JSONL",          "Sample 8 frames uniformly · build text prompt\n→ {image_paths, prompt, answer, metadata}\nSame 8-frame convention at train + eval time", ACCENT),
        ("Stage 3 — Training / Evaluation",    "LoRA training (finetune.py)\nZero-shot / adapter eval (eval_cross_dataset_vlm.py)\nSlowFast X3D/I3D (share clip boundaries)", ACCENT2),
    ]

    arrows = ["→", "→", "→"]
    box_w  = Inches(2.8)
    box_h  = Inches(4.4)
    gap    = Inches(0.42)
    start  = Inches(0.35)
    top    = Inches(1.88)

    for i, (title, body, col) in enumerate(stages):
        lft = start + i*(box_w + gap)
        box(s, lft, top, box_w, box_h,
            fill_color=RGBColor(0x12, 0x24, 0x36))
        box(s, lft, top, box_w, Inches(0.48), fill_color=col)
        txt(s, title, lft+Inches(0.1), top+Inches(0.04),
            box_w-Inches(0.2), Inches(0.42),
            size=Pt(13), bold=True, color=BG)
        txt(s, body, lft+Inches(0.1), top+Inches(0.56),
            box_w-Inches(0.2), box_h-Inches(0.65),
            size=Pt(12.5), color=WHITE)
        if i < 3:
            txt(s, "→",
                lft+box_w+Inches(0.06), top+Inches(1.7),
                Inches(0.35), Inches(0.55),
                size=Pt(28), bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

    # Key design choices
    txt(s, "Key design choices",
        Inches(0.35), Inches(6.42), Inches(12.6), Inches(0.38),
        size=Pt(15), bold=True, color=ACCENT)
    bullet_column(s, [
        "8 frames per clip (uniform, temporal order)  ·  Same at train and eval",
        "Prompt explicitly lists all allowed labels → substring-match scoring",
        "Plain-text label as answer (not structured output) → better cross-dataset transfer",
    ], Inches(0.35), Inches(6.82), Inches(12.6), Inches(0.62),
       size=Pt(13.5), bullet="• ", spacing=Inches(0))


def slide_models(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "1  Background & Methods", "Models Compared")

    # VLM side
    box(s, Inches(0.4), Inches(1.85), Inches(7.5), Inches(4.9),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    txt(s, "Vision-Language Models (VLMs)",
        Inches(0.55), Inches(1.95), Inches(7.2), Inches(0.45),
        size=Pt(17), bold=True, color=ACCENT)

    vlm_rows = [
        ("Qwen2.5-VL-3B",        "3B",  "Zero-shot + LoRA + Joint"),
        ("Qwen2.5-VL-7B",        "7B",  "Zero-shot + LoRA"),
        ("LLaVA-OneVision-0.5B", "0.5B","Zero-shot + LoRA"),
        ("LLaVA-OneVision-7B",   "7B",  "Zero-shot + LoRA"),
        ("Gemma3-4B",            "4B",  "Zero-shot + LoRA"),
        ("MobileVLM-V2-1.7B",    "1.7B","Zero-shot + LoRA"),
    ]
    cols_x = [Inches(0.55), Inches(3.6), Inches(5.2)]
    for i, (name, size, regime) in enumerate(vlm_rows):
        y = Inches(2.48) + i*Inches(0.68)
        txt(s, name,   cols_x[0], y, Inches(3.0), Inches(0.6), size=Pt(14), color=WHITE)
        txt(s, size,   cols_x[1], y, Inches(1.5), Inches(0.6), size=Pt(14), color=ACCENT2, align=PP_ALIGN.CENTER)
        txt(s, regime, cols_x[2], y, Inches(2.5), Inches(0.6), size=Pt(12), color=GRAY)

    # Specialist video side
    box(s, Inches(8.2), Inches(1.85), Inches(4.7), Inches(4.9),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    txt(s, "Specialist Video Models",
        Inches(8.35), Inches(1.95), Inches(4.4), Inches(0.45),
        size=Pt(17), bold=True, color=ACCENT2)
    bullet_column(s, [
        "X3D-L  — trained from scratch on\n  derived clip manifests",
        "I3D-L  — trained from scratch",
        "Both share the same clip boundaries\nand train/val split as VLMs",
        "Evaluated on the same val set\n→ direct apples-to-apples comparison",
    ], Inches(8.35), Inches(2.45), Inches(4.4), Inches(4.2),
       size=Pt(14.5), bullet="• ")

    # Eval regimes bar
    box(s, Inches(0.4), Inches(6.95), Inches(12.5), Inches(0.45),
        fill_color=RGBColor(0x16, 0x2A, 0x40))
    txt(s, "Three evaluation regimes:  "
           "  ①  Zero-shot (no training)     "
           "  ②  In-domain LoRA     "
           "  ③  Cross-dataset transfer",
        Inches(0.55), Inches(6.97), Inches(12.2), Inches(0.40),
        size=Pt(14), color=WHITE)


def table_on_slide(slide, headers, rows,
                   left, top, col_widths, row_height=Inches(0.46),
                   header_color=ACCENT, header_text_color=BG,
                   odd_color=RGBColor(0x12, 0x24, 0x36),
                   even_color=RGBColor(0x0D, 0x1B, 0x2A),
                   text_color=WHITE, font_size=Pt(13.5),
                   highlight_col=None, highlight_color=None):
    total_w = sum(col_widths)
    # Header row
    x = left
    for h, w in zip(headers, col_widths):
        box(slide, x, top, w, row_height, fill_color=header_color)
        txt(slide, h, x+Inches(0.07), top+Inches(0.05),
            w-Inches(0.14), row_height-Inches(0.08),
            size=Pt(13), bold=True, color=header_text_color, align=PP_ALIGN.CENTER)
        x += w
    # Data rows
    for r_idx, row in enumerate(rows):
        y = top + row_height*(r_idx+1)
        bg_col = odd_color if r_idx % 2 == 0 else even_color
        x = left
        for c_idx, (cell, w) in enumerate(zip(row, col_widths)):
            cell_color = (highlight_color if highlight_col is not None
                          and c_idx == highlight_col else bg_col)
            box(slide, x, y, w, row_height, fill_color=cell_color)
            # color-code numeric deltas
            tcol = text_color
            if isinstance(cell, str) and cell.startswith("+"):
                tcol = GREEN
            elif isinstance(cell, str) and cell.startswith("−") or (
                 isinstance(cell, str) and cell.startswith("-") and len(cell) > 1):
                tcol = RED
            txt(slide, str(cell), x+Inches(0.07), y+Inches(0.05),
                w-Inches(0.14), row_height-Inches(0.08),
                size=font_size, color=tcol, align=PP_ALIGN.CENTER)
            x += w


def slide_kabr_results(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "2  Results", "KABR-Clip — Adapted VLMs Beat X3D-L")

    headers = ["Model", "Zero-shot Acc", "LoRA Acc", "Δ vs X3D-L"]
    rows = [
        ["Gemma3-4B",            "0.549", "0.864", "+6.8 pts ✓"],
        ["LLaVA-OV-7B",          "0.581", "0.863", "+6.7 pts ✓"],
        ["Qwen2.5-VL-3B Joint",  "0.549", "0.863", "+6.7 pts ✓"],
        ["Qwen2.5-VL-3B",        "0.549", "0.853", "+5.6 pts ✓"],
        ["Qwen2.5-VL-7B",        "0.626", "0.848", "+5.1 pts ✓"],
        ["LLaVA-OV-0.5B",        "0.307", "0.819", "+2.2 pts ✓"],
        ["I3D-L (scratch)",      "—",     "0.814", "+1.7 pts"],
        ["X3D-L (scratch)",      "—",     "0.797", "baseline"],
        ["MobileVLM-1.7B",       "0.327", "0.782", "−1.5 pts"],
        ["MiniCPM-V 4.6 †",      "0.652", "LoRA pending", "—"],
        ["LFM2.5-VL-1.6B †",     "0.436", "LoRA pending", "—"],
    ]
    col_widths = [Inches(3.5), Inches(2.5), Inches(2.6), Inches(2.6)]
    table_on_slide(s, headers, rows,
                   left=Inches(1.1), top=Inches(1.85),
                   col_widths=col_widths, row_height=Inches(0.42),
                   highlight_col=3, highlight_color=RGBColor(0x0A, 0x2A, 0x18))

    txt(s, "★  5 of 6 LoRA VLMs beat X3D-L  ·  Even 0.5B wins  ·  Joint LoRA ties best in-domain  "
           "·  † small VLMs: zero-shot screening only, LoRA not yet started",
        Inches(1.1), Inches(6.68), Inches(11.1), Inches(0.55),
        size=Pt(12.5), bold=False, color=ACCENT2)


def slide_baboonland_results(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "2  Results", "Baboonland-Clip — VLMs Dominate Specialist Baselines")

    headers = ["Model", "Zero-shot Acc", "LoRA Acc", "vs X3D-L (0.671)"]
    rows = [
        ["LLaVA-OV-7B",         "0.338", "0.775", "+10.4 pts ✓"],
        ["LLaVA-OV-0.5B",       "0.418", "0.768", "+9.7 pts ✓"],
        ["Qwen2.5-VL-3B Joint", "0.242", "0.759", "+8.8 pts ✓"],
        ["Gemma3-4B",           "0.540", "0.756", "+8.5 pts ✓"],
        ["Qwen2.5-VL-7B",       "0.329", "0.755", "+8.4 pts ✓"],
        ["Qwen2.5-VL-3B",       "0.242", "0.734", "+6.3 pts ✓"],
        ["MobileVLM-1.7B",      "0.174", "0.712", "+4.1 pts ✓"],
        ["X3D-L (scratch)",     "—",     "0.671", "baseline"],
        ["I3D-L (scratch)",     "—",     "0.630", "−4.1 pts"],
        ["LFM2.5-VL-1.6B †",   "0.509", "LoRA pending", "—"],
        ["MiniCPM-V 4.6 †",    "0.377", "LoRA pending", "—"],
    ]
    col_widths = [Inches(3.5), Inches(2.5), Inches(2.6), Inches(2.6)]
    table_on_slide(s, headers, rows,
                   left=Inches(1.1), top=Inches(1.85),
                   col_widths=col_widths, row_height=Inches(0.42),
                   font_size=Pt(13.5),
                   highlight_col=3, highlight_color=RGBColor(0x0A, 0x2A, 0x18))

    txt(s, "★  Every adapted VLM beats X3D-L  ·  0.5B LoRA > X3D-L by +9.7 pts  "
           "·  † small VLMs: zero-shot screening only, LoRA not yet started",
        Inches(1.1), Inches(6.68), Inches(11.1), Inches(0.55),
        size=Pt(12.5), bold=False, color=ACCENT2)


def slide_class_collapse(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "2  Results", "Named Failure Mode: Class Collapse")

    # Explanation box
    box(s, Inches(0.4), Inches(1.85), Inches(12.5), Inches(1.55),
        fill_color=RGBColor(0x1A, 0x0A, 0x00))
    txt(s, "Class Collapse",
        Inches(0.6), Inches(1.92), Inches(5), Inches(0.45),
        size=Pt(18), bold=True, color=ACCENT2)
    txt(s, "A zero-shot VLM predicts only 1-2 majority classes on a severely imbalanced dataset.\n"
           "Accuracy looks acceptable (it predicts the common class). Macro-F1 exposes the failure.",
        Inches(0.6), Inches(2.38), Inches(12.1), Inches(0.9),
        size=Pt(14.5), color=WHITE)

    # Evidence cards
    examples = [
        ("MobileVLM zero-shot\nBaboonland",   "0.174 acc\n0.024 F1",  "Predicts Walking/Running\non 99% of 6,198 clips",  RED),
        ("LLaVA-OV-7B zero-shot\nBaboonland", "0.338 acc\n0.077 F1",  "High acc, collapsed F1\n3 of 13 classes used",    RED),
        ("Qwen3B zero-shot\nBaboonland",       "0.242 acc\n0.080 F1",  "Similar pattern:\nmajority-class collapse",       RED),
        ("After LoRA fine-tuning\nQwen3B",     "0.734 acc\n0.327 F1",  "All 13 classes predicted\ncollapse resolved",     GREEN),
    ]
    card_w = Inches(2.9)
    card_h = Inches(2.85)
    for i, (title, metrics, note, col) in enumerate(examples):
        lft = Inches(0.4) + i*(card_w + Inches(0.25))
        box(s, lft, Inches(3.55), card_w, card_h,
            fill_color=RGBColor(0x12, 0x24, 0x36))
        box(s, lft, Inches(3.55), card_w, Inches(0.08), fill_color=col)
        txt(s, title, lft+Inches(0.1), Inches(3.65), card_w-Inches(0.2), Inches(0.72),
            size=Pt(13.5), bold=True, color=WHITE)
        txt(s, metrics, lft+Inches(0.1), Inches(4.4), card_w-Inches(0.2), Inches(0.72),
            size=Pt(22), bold=True, color=col, align=PP_ALIGN.CENTER)
        txt(s, note, lft+Inches(0.1), Inches(5.15), card_w-Inches(0.2), Inches(1.15),
            size=Pt(13), color=GRAY, italic=True)

    txt(s, "Key insight: accuracy hides class collapse — macro-F1 always needs to be reported alongside accuracy",
        Inches(0.4), Inches(6.55), Inches(12.5), Inches(0.4),
        size=Pt(14), bold=True, color=ACCENT)


def slide_contributions(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "2  Results", "Joint Cross-Dataset LoRA (Contribution A)")

    # Motivation box
    box(s, Inches(0.4), Inches(1.85), Inches(12.5), Inches(1.0),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    txt(s, "Goal: train ONE LoRA adapter on KABR + Baboonland together — can it match separate in-domain adapters?",
        Inches(0.6), Inches(1.98), Inches(12.1), Inches(0.75),
        size=Pt(15.5), color=WHITE)

    # Comparison table
    a_headers = ["Training setup", "KABR Accuracy", "Baboonland Accuracy"]
    a_rows = [
        ["KABR-only LoRA  (in-domain reference)",      "0.853", "—"],
        ["Baboonland-only LoRA  (in-domain reference)", "—",     "0.734"],
        ["KABR-only LoRA evaluated on Baboonland",      "—",     "0.397  (transfer)"],
        ["Joint LoRA A1  ✓  BEST",                      "0.863", "0.759"],
    ]
    col_widths_a = [Inches(5.8), Inches(2.9), Inches(3.3)]
    table_on_slide(s, a_headers, a_rows,
                   left=Inches(0.55), top=Inches(2.95),
                   col_widths=col_widths_a, row_height=Inches(0.62),
                   font_size=Pt(14))

    bullet_column(s, [
        "One adapter trained on KABR + Baboonland simultaneously",
        "Each example keeps its own dataset-specific prompt and label set",
        "Joint A1 matches or exceeds in-domain LoRA on BOTH val sets simultaneously",
        "Direct transfer (KABR→Baboonland) gives 0.397 — joint training gives 0.759",
    ], Inches(0.55), Inches(5.65), Inches(12.3), Inches(1.7),
       size=Pt(15), bullet="▸ ", color=ACCENT2)

    txt(s, "★  One adapter — two datasets — no per-dataset fine-tuning needed",
        Inches(0.55), Inches(7.08), Inches(12.3), Inches(0.38),
        size=Pt(14), bold=True, color=WHITE)


def slide_transfer(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "2  Results", "Cross-Dataset Transfer")

    headers = ["Direction", "Source LoRA", "Accuracy", "Macro-F1"]
    rows = [
        ["Baboonland → KABR", "LLaVA-OV-7B",           "0.652", "0.256"],
        ["Baboonland → KABR", "Qwen2.5-VL-7B",         "0.562", "0.261"],
        ["Baboonland → KABR", "LLaVA-OV-0.5B",         "0.517", "0.181"],
        ["Baboonland → KABR", "MobileVLM-1.7B",         "0.447", "0.179"],
        ["Baboonland → KABR", "Qwen2.5-VL-3B",          "0.385", "0.114"],
        ["KABR → Baboonland", "Qwen3B 8-frame",          "0.397", "0.155"],
        ["KABR → Baboonland", "Qwen3B 4-frame",          "0.298", "0.129"],
        ["KABR → Baboonland", "Qwen3B structured output","0.191", "0.072"],
    ]
    col_widths = [Inches(3.2), Inches(3.5), Inches(2.2), Inches(2.2)]
    table_on_slide(s, headers, rows,
                   left=Inches(1.1), top=Inches(1.85),
                   col_widths=col_widths)

    findings = [
        "Model scale helps transfer  (7B > 3B > 0.5B across families)",
        "8 frames > 4 frames for KABR→Bbl transfer",
        "Structured-output prompts HURT transfer worst (0.191 acc) — rigid format over-specializes to source label set",
        "Joint training (A1) fully closes the gap  —  much stronger than direct transfer",
    ]
    bullet_column(s, findings,
                  Inches(1.1), Inches(5.95), Inches(11.1), Inches(1.45),
                  size=Pt(14.5), bullet="▸ ", color=ACCENT2)


def slide_small_vlm(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "2  Results", "Small-VLM Zero-Shot Screening")

    headers = ["Model", "Params", "KABR Accuracy", "Baboonland Accuracy", "Status"]
    rows = [
        ["MiniCPM-V 4.6",      "~4B", "0.652", "0.377", "✓ complete"],
        ["LFM2.5-VL-1.6B",     "1.6B","0.436", "0.509", "✓ complete"],
        ["Qwen3.5-0.8B",       "0.8B","—",     "—",     "⚠ verbose output"],
        ["Gemma4-E2B-it",      "~2B", "—",     "—",     "✗ unsupported loader"],
        ["LLaVA-OV-0.5B (ref)","0.5B","0.307", "0.418", "↑ after LoRA: 0.768 acc"],
    ]
    col_widths = [Inches(2.9), Inches(0.9), Inches(2.4), Inches(2.9), Inches(3.0)]
    table_on_slide(s, headers, rows,
                   left=Inches(0.45), top=Inches(1.88),
                   col_widths=col_widths, font_size=Pt(13.5))

    bullet_column(s, [
        "MiniCPM-V 4.6 is the strongest small zero-shot model on KABR (0.652)",
        "LFM2.5-VL-1.6B stronger on Baboonland (0.509 acc) — both below LoRA-adapted models",
        "Zero-shot small VLMs confirm: adaptation (LoRA) remains essential for reliable wildlife behavior recognition",
        "Qwen3.5-0.8B and Gemma4 need loader fixes before full eval",
    ], Inches(0.45), Inches(5.55), Inches(12.5), Inches(1.8),
       size=Pt(14), bullet="▸ ", color=WHITE)


def slide_litreview_table(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "3  Efficiency", "How Our Evaluated VLMs Handle Visual Tokens")

    headers = ["Model", "Params", "Tokens/frame → LLM", "Key efficiency mechanism"]
    rows = [
        ["Qwen2.5-VL-3B/7B",   "3–7B",  "Dynamic (64–16K)",  "Window attn in 28/32 ViT layers + 2×2 MLP merger (4× compression)"],
        ["LLaVA-OV-0.5B/7B",   "0.5–7B","196  (bilinear pool)","Bilinear pool from 729 → 196 per frame; AnyRes tiling for stills"],
        ["Gemma3-4B",           "4B",    "Fixed 256",          "Fixed-budget MLP projector; 5:1 local/global attn in LLM backbone"],
        ["MobileVLM-V2-1.7B",  "1.7B",  "144  (4× avg pool)", "LDPv2 projector: 2×2 avg pool + lightweight PEG; 51 tok/s on Jetson"],
        ["MiniCPM-V 4.6",      "1.3B",  "64–96  (resampler)", "Perceiver resampler (learned cross-attn); 3D resampler for video"],
        ["LFM2.5-VL-1.6B",    "1.6B",  "64–256  (4× unshuffle)","Hybrid: 18 gated convolutions + 6 GQA layers; NAS-optimized"],
    ]
    col_widths = [Inches(2.35), Inches(0.9), Inches(2.25), Inches(7.1)]
    table_on_slide(s, headers, rows,
                   left=Inches(0.35), top=Inches(1.88),
                   col_widths=col_widths, font_size=Pt(13),
                   row_height=Inches(0.66))

    txt(s, "All 6 models compress visual tokens before the LLM — this is the single biggest latency lever",
        Inches(0.35), Inches(6.95), Inches(12.6), Inches(0.42),
        size=Pt(14), bold=True, color=ACCENT)


def slide_litreview_common(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "3  Efficiency", "Common Techniques & What Matters for 8-Frame Clips")

    # Left: common techniques
    box(s, Inches(0.35), Inches(1.85), Inches(6.0), Inches(5.35),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    txt(s, "Common across all 6 models",
        Inches(0.5), Inches(1.93), Inches(5.7), Inches(0.45),
        size=Pt(16), bold=True, color=ACCENT)
    bullet_column(s, [
        "Token compression at the projector (4× reduction typical)",
        "Frozen ViT encoder — no gradient through encoder",
        "SigLIP ~400M as preferred visual backbone",
        "Dynamic / adaptive tiling for variable aspect ratios",
        "4-bit quantization for edge/mobile deployment",
    ], Inches(0.5), Inches(2.45), Inches(5.7), Inches(4.6),
       size=Pt(14.5), bullet="• ")

    # Right: most relevant for 8 frames
    box(s, Inches(6.65), Inches(1.85), Inches(6.3), Inches(5.35),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    txt(s, "Most promising for 8-frame wildlife clips",
        Inches(6.8), Inches(1.93), Inches(6.0), Inches(0.45),
        size=Pt(16), bold=True, color=ACCENT2)

    items = [
        ("Gemma3 fixed-256 projection",
         "Predictable 2,048 total tokens for 8 frames — fits any 4B+ LLM cleanly"),
        ("MiniCPM-V 3D resampler",
         "Joint spatial-temporal compression across frames; 11.5× speedup on video with modest accuracy drop"),
        ("LFM2 gated-convolution backbone",
         "O(1) KV-cache growth regardless of sequence length — 2× faster than same-size transformer"),
        ("Qwen2.5-VL dynamic FPS + MRoPE",
         "Absolute temporal position encoding; handles variable-speed behaviors correctly"),
    ]
    for i, (title, body) in enumerate(items):
        y_title = Inches(2.45) + i * Inches(1.22)
        txt(s, f"▸  {title}", Inches(6.8), y_title,
            Inches(6.0), Inches(0.4),
            size=Pt(13.5), bold=True, color=WHITE)
        txt(s, body, Inches(7.0), y_title + Inches(0.38),
            Inches(5.85), Inches(0.78),
            size=Pt(12.5), color=GRAY)


def slide_litreview_gaps(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "3  Efficiency", "Planned Contribution — Efficiency for Wildlife VLMs")

    box(s, Inches(0.35), Inches(1.85), Inches(12.6), Inches(0.75),
        fill_color=RGBColor(0x0A, 0x1E, 0x38))
    txt(s, "The literature shows every model compresses tokens aggressively — "
           "but nobody has measured accuracy vs. latency on wildlife behavior datasets. "
           "My planned contribution fills this gap.",
        Inches(0.55), Inches(1.95), Inches(12.15), Inches(0.6),
        size=Pt(14), color=WHITE)

    contribs = [
        (ACCENT,  "Latency–accuracy Pareto curve  (empirical)",
         "Vary tokens/frame (64 → 729) across all 6 evaluated VLMs on KABR and Baboonland. "
         "First published Pareto frontier for wildlife behavior — shows exactly where the accuracy cliff is."),
        (ACCENT2, "Token budget sweep with existing LoRA adapters  (Option A, no retraining)",
         "Qwen2.5-VL supports min_pixels / max_pixels at inference. Run the sweep on the trained adapter. "
         "Target claim: X tokens/frame captures Y% accuracy at Z× speedup over the VLM default."),
        (GREEN,   "Multi-objective NAS: accuracy × latency × adapter size  (Option B, ~1 day)",
         "Add a latency timer to the planned 16-run NAS grid (frames × rank × modules × loss). "
         "Turns single-objective search into a 3-axis Pareto — novel framing for LoRA adapter design."),
        (RGBColor(0xB0, 0x6A, 0xFF), "Behavior-aware frame selection  (Option C, stretch goal)",
         "Use SigLIP CLS cosine similarity to drop redundant frames before the LLM. "
         "Low-motion behaviors (grooming, resting) need far fewer frames — this is behavior-class aware, not content-agnostic."),
    ]

    contrib_h = Inches(1.15)
    for i, (col, title, body) in enumerate(contribs):
        top = Inches(2.72) + i * contrib_h
        box(s, Inches(0.35), top, Inches(12.6), contrib_h - Inches(0.06),
            fill_color=RGBColor(0x12, 0x24, 0x36))
        box(s, Inches(0.35), top, Inches(0.08), contrib_h - Inches(0.06), fill_color=col)
        box(s, Inches(0.52), top + Inches(0.06), Inches(0.38), Inches(0.38), fill_color=col)
        txt(s, str(i + 1), Inches(0.52), top + Inches(0.06), Inches(0.38), Inches(0.38),
            size=Pt(14), bold=True, color=BG, align=PP_ALIGN.CENTER)
        txt(s, title, Inches(1.0), top + Inches(0.07),
            Inches(11.8), Inches(0.38),
            size=Pt(13.5), bold=True, color=col)
        txt(s, body, Inches(1.0), top + Inches(0.46),
            Inches(11.8), Inches(0.62),
            size=Pt(12.5), color=GRAY)
        if i < len(contribs) - 1:
            box(s, Inches(0.35), top + contrib_h - Inches(0.06),
                Inches(12.6), Inches(0.02),
                fill_color=RGBColor(0x1E, 0x35, 0x50))


def slide_summary(prs):
    s = blank_slide(prs)
    bg(s)
    accent_bar(s)
    txt(s, "Summary",
        Inches(0.45), Inches(0.35), Inches(12.4), Inches(0.85),
        size=Pt(36), bold=True, color=WHITE)

    # Claims
    box(s, Inches(0.4), Inches(1.42), Inches(7.4), Inches(5.6),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    txt(s, "Key findings",
        Inches(0.55), Inches(1.5), Inches(7.1), Inches(0.45),
        size=Pt(17), bold=True, color=ACCENT)
    claims = [
        "Zero-shot VLMs underperform specialist video models (X3D-L, I3D-L)",
        "LoRA adaptation delivers large, consistent accuracy gains (+~0.3) across all six backbones",
        "Adapted VLMs beat X3D-L on both datasets — even a 0.5B model wins",
        "Joint LoRA trains one adapter for both datasets and matches in-domain accuracy on each",
    ]
    bullet_column(s, claims,
                  Inches(0.55), Inches(2.05), Inches(7.15), Inches(4.7),
                  size=Pt(15.5), bullet="▸ ")

    # Numbers
    box(s, Inches(8.15), Inches(1.42), Inches(4.8), Inches(5.6),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    txt(s, "Best accuracy numbers",
        Inches(8.3), Inches(1.5), Inches(4.5), Inches(0.45),
        size=Pt(17), bold=True, color=ACCENT2)
    numbers = [
        ("KABR best LoRA",       "0.864  (Gemma3-4B)"),
        ("KABR X3D-L baseline",  "0.797"),
        ("Bbl best LoRA",        "0.775  (LLaVA-OV-7B)"),
        ("Bbl X3D-L baseline",   "0.671"),
        ("Joint LoRA on KABR",   "0.863"),
        ("Joint LoRA on Bbl",    "0.759"),
    ]
    for i, (label, val) in enumerate(numbers):
        y = Inches(2.08) + i*Inches(0.82)
        txt(s, label, Inches(8.3), y, Inches(2.5), Inches(0.7),
            size=Pt(13.5), color=GRAY)
        txt(s, val, Inches(10.8), y, Inches(2.1), Inches(0.7),
            size=Pt(14), bold=True, color=WHITE)

    txt(s, "One joint Qwen2.5-VL-3B LoRA adapter handles both wildlife behavior datasets at or above specialist-video accuracy.",
        Inches(0.4), Inches(7.12), Inches(12.5), Inches(0.35),
        size=Pt(13), color=GRAY, italic=True)


def slide_section(prs, number, title, subtitle="", items=None):
    """Full-bleed section divider slide."""
    s = blank_slide(prs)
    bg(s)
    # thick left accent stripe
    box(s, 0, 0, Inches(0.55), SLIDE_H, fill_color=ACCENT)
    # section number badge
    box(s, Inches(0.85), Inches(2.4), Inches(1.1), Inches(1.1),
        fill_color=ACCENT)
    txt(s, str(number),
        Inches(0.85), Inches(2.4), Inches(1.1), Inches(1.1),
        size=Pt(44), bold=True, color=BG, align=PP_ALIGN.CENTER)
    # title
    txt(s, title,
        Inches(2.2), Inches(2.35), Inches(10.6), Inches(1.15),
        size=Pt(44), bold=True, color=WHITE)
    # subtitle
    if subtitle:
        txt(s, subtitle,
            Inches(2.2), Inches(3.6), Inches(10.6), Inches(0.6),
            size=Pt(20), color=ACCENT, bold=False)
    # bullet items listing slides in this section
    if items:
        bullet_column(s, items,
                      Inches(2.2), Inches(4.35), Inches(10.6), Inches(2.8),
                      size=Pt(16), bullet="— ", color=GRAY)


def slide_option_a(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "4  Proposed Work", "Option A — Token Budget Sweep  (2–3 weeks)")

    # What it is
    box(s, Inches(0.35), Inches(1.85), Inches(12.6), Inches(1.08),
        fill_color=RGBColor(0x0A, 0x1E, 0x38))
    txt(s, "Vary the number of visual tokens per frame at inference time and measure accuracy + latency on KABR and Baboonland.",
        Inches(0.55), Inches(1.98), Inches(12.2), Inches(0.82),
        size=Pt(15.5), color=WHITE)

    # Left: what you already have / how to do it
    box(s, Inches(0.35), Inches(3.05), Inches(6.1), Inches(3.65),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    txt(s, "What you already have",
        Inches(0.5), Inches(3.13), Inches(5.8), Inches(0.42),
        size=Pt(15), bold=True, color=ACCENT)
    bullet_column(s, [
        "Trained Qwen2.5-VL-3B LoRA adapter (KABR + Baboonland)",
        "Qwen2.5-VL supports min_pixels / max_pixels at inference — no retraining needed",
        "4-frame vs 8-frame results already give two data points",
        "Eval script (eval_cross_dataset_vlm.py) just needs a token-budget loop",
    ], Inches(0.5), Inches(3.6), Inches(5.8), Inches(3.0),
       size=Pt(13.5), bullet="✓ ")

    # Right: proposed sweep + claim
    box(s, Inches(6.65), Inches(3.05), Inches(6.3), Inches(3.65),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    txt(s, "Proposed sweep",
        Inches(6.8), Inches(3.13), Inches(6.0), Inches(0.42),
        size=Pt(15), bold=True, color=ACCENT2)
    sweep_rows = ["64", "128", "256", "512", "1024 (default)"]
    for i, r in enumerate(sweep_rows):
        lft = Inches(6.8) + i * Inches(1.1)
        box(s, lft, Inches(3.6), Inches(1.0), Inches(0.62),
            fill_color=RGBColor(0x16, 0x2A, 0x40))
        txt(s, r + "\ntok/fr",
            lft, Inches(3.6), Inches(1.0), Inches(0.62),
            size=Pt(11), color=WHITE, align=PP_ALIGN.CENTER)
    txt(s, "→ accuracy + wall-clock latency at each point\n→ plot Pareto curve: accuracy vs. tokens/frame",
        Inches(6.8), Inches(4.32), Inches(6.0), Inches(0.75),
        size=Pt(13.5), color=GRAY)
    txt(s, "Estimated effort:  1–2 days of eval runs + plotting",
        Inches(6.8), Inches(5.12), Inches(6.0), Inches(0.42),
        size=Pt(13), color=GRAY, italic=True)

    # Paper claim
    box(s, Inches(0.35), Inches(6.82), Inches(12.6), Inches(0.55),
        fill_color=RGBColor(0x00, 0x50, 0x70))
    txt(s, "Paper claim:  "
           "\"For wildlife behavior clips, X tokens/frame captures Y% of full accuracy at Z× speedup — "
           "the Pareto frontier peaks far below the VLM default token budget.\"",
        Inches(0.55), Inches(6.87), Inches(12.2), Inches(0.45),
        size=Pt(13.5), color=WHITE)


def slide_option_b(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "4  Proposed Work", "Option B — Add Latency to the NAS Grid  (~1 extra day)")

    box(s, Inches(0.35), Inches(1.85), Inches(12.6), Inches(1.08),
        fill_color=RGBColor(0x0A, 0x1E, 0x38))
    txt(s, "The planned 16-run NAS grid already varies frames, LoRA rank, target modules, and loss. "
           "Add a latency timer to each eval run — the result becomes a 3-objective Pareto search.",
        Inches(0.55), Inches(1.98), Inches(12.2), Inches(0.82),
        size=Pt(15.5), color=WHITE)

    # NAS grid reminder
    box(s, Inches(0.35), Inches(3.05), Inches(5.6), Inches(3.65),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    txt(s, "Planned NAS grid (16 runs)",
        Inches(0.5), Inches(3.13), Inches(5.3), Inches(0.42),
        size=Pt(15), bold=True, color=ACCENT)
    nas_rows = [
        ("Frames/clip",      "4, 8"),
        ("LoRA rank",        "8, 16"),
        ("Target modules",   "attn-only,  attn+MLP"),
        ("Loss",             "CE,  inv-sqrt CE"),
        ("Total",            "16 training + eval runs"),
    ]
    for i, (k, v) in enumerate(nas_rows):
        y = Inches(3.62) + i * Inches(0.6)
        txt(s, k, Inches(0.5), y, Inches(2.4), Inches(0.55),
            size=Pt(13.5), color=GRAY)
        txt(s, v, Inches(2.9), y, Inches(2.9), Inches(0.55),
            size=Pt(13.5), bold=True, color=WHITE)

    # What to add
    box(s, Inches(6.2), Inches(3.05), Inches(6.75), Inches(3.65),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    txt(s, "The one-line addition",
        Inches(6.35), Inches(3.13), Inches(6.45), Inches(0.42),
        size=Pt(15), bold=True, color=ACCENT2)
    bullet_column(s, [
        "Add torch.cuda.Event timer around generation loop in eval script",
        "Record ms/clip alongside accuracy for every NAS eval run",
        "Plot 3-axis Pareto: accuracy × latency × trainable params",
        "Best-practice framing: the NAS objective is now multi-objective, not just accuracy",
        "Combines with Option A for a complete efficiency story",
    ], Inches(6.35), Inches(3.62), Inches(6.45), Inches(2.95),
       size=Pt(13.5), bullet="▸ ")

    box(s, Inches(0.35), Inches(6.82), Inches(12.6), Inches(0.55),
        fill_color=RGBColor(0x00, 0x50, 0x70))
    txt(s, "Paper claim:  "
           "\"Resource-aware adapter search identifies configurations that reduce inference latency by X× "
           "with <Y% accuracy drop — making adapted VLMs competitive with X3D-L on efficiency.\"",
        Inches(0.55), Inches(6.87), Inches(12.2), Inches(0.45),
        size=Pt(13.5), color=WHITE)


def slide_option_c(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "4  Proposed Work", "Option C — Adaptive Frame Selection  (4–6 weeks)")

    box(s, Inches(0.35), Inches(1.85), Inches(12.6), Inches(1.08),
        fill_color=RGBColor(0x0A, 0x1E, 0x38))
    txt(s, "Use cosine similarity between adjacent frame features (from the frozen SigLIP encoder) "
           "to drop redundant frames and keep only the K most temporally diverse ones from the clip.",
        Inches(0.55), Inches(1.98), Inches(12.2), Inches(0.82),
        size=Pt(15.5), color=WHITE)

    # How it works
    box(s, Inches(0.35), Inches(3.05), Inches(6.1), Inches(3.65),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    txt(s, "How it works",
        Inches(0.5), Inches(3.13), Inches(5.8), Inches(0.42),
        size=Pt(15), bold=True, color=ACCENT)
    bullet_column(s, [
        "Encode all N frames with the frozen SigLIP ViT (one forward pass per frame)",
        "Compute pairwise cosine similarity across frame CLS tokens",
        "Greedily select K frames that maximize temporal diversity (min-similarity set)",
        "Feed only those K frames to the LLM — same inference pipeline",
        "No retraining required for evaluation; optional LoRA training with selected frames",
    ], Inches(0.5), Inches(3.6), Inches(5.8), Inches(3.0),
       size=Pt(13.5), bullet="▸ ")

    # Motivation
    box(s, Inches(6.55), Inches(3.05), Inches(6.4), Inches(3.65),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    txt(s, "Why this is behavior-specific",
        Inches(6.7), Inches(3.13), Inches(6.1), Inches(0.42),
        size=Pt(15), bold=True, color=ACCENT2)
    bullet_column(s, [
        "Grooming / resting clips: frames 1–30 nearly identical → only 2–3 needed",
        "Running / fighting clips: high frame-to-frame change → keep more frames",
        "Existing compression (pooling, resampler) ignores this temporal structure entirely",
        "Novelty: behavior-class aware token savings without any class supervision signal",
        "Risk: harder to argue novelty; dynamic frame selection papers exist for general video",
    ], Inches(6.7), Inches(3.6), Inches(6.1), Inches(3.0),
       size=Pt(13.5), bullet="• ", color=WHITE)

    box(s, Inches(0.35), Inches(6.82), Inches(12.6), Inches(0.55),
        fill_color=RGBColor(0x00, 0x50, 0x70))
    txt(s, "Paper claim:  "
           "\"Behavior-aware frame selection reduces effective token count by X× on low-motion behaviors "
           "with <Y% accuracy loss — the reduction correlates with behavior class, not model size.\"",
        Inches(0.55), Inches(6.87), Inches(12.2), Inches(0.45),
        size=Pt(13.5), color=WHITE)


def slide_options_overview(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "4  Proposed Work", "Three Options — Effort vs. Impact")

    options = [
        ("A", "Token Budget Sweep",
         "2–3 weeks",
         "Vary tokens/frame on existing trained adapters. "
         "No new training. One eval loop + Pareto plot.",
         ACCENT,
         "Lowest effort · Fills a clear gap in the literature"),
        ("B", "NAS + Latency Timer",
         "~1 extra day",
         "Add timing to the already-planned 16-run NAS grid. "
         "Turns single-objective search into multi-objective Pareto.",
         GREEN,
         "Almost free · Amplifies existing planned work"),
        ("C", "Adaptive Frame Selection",
         "4–6 weeks",
         "Use SigLIP features to drop redundant frames before the LLM. "
         "Novel behavior-specific efficiency signal.",
         ACCENT2,
         "Most novel · Needs new module + risk of prior-art overlap"),
    ]

    for i, (letter, title, effort, body, col, tagline) in enumerate(options):
        lft = Inches(0.35) + i * Inches(4.38)
        w   = Inches(4.15)

        # card
        box(s, lft, Inches(1.85), w, Inches(5.35),
            fill_color=RGBColor(0x12, 0x24, 0x36))
        # top color band
        box(s, lft, Inches(1.85), w, Inches(0.55), fill_color=col)
        # letter badge
        box(s, lft + Inches(0.15), Inches(1.92), Inches(0.42), Inches(0.42),
            fill_color=BG)
        txt(s, letter, lft + Inches(0.15), Inches(1.92),
            Inches(0.42), Inches(0.42),
            size=Pt(18), bold=True, color=col, align=PP_ALIGN.CENTER)

        txt(s, title,
            lft + Inches(0.68), Inches(1.92), w - Inches(0.8), Inches(0.42),
            size=Pt(15), bold=True, color=BG)

        # effort pill
        box(s, lft + Inches(0.15), Inches(2.52), w - Inches(0.3), Inches(0.38),
            fill_color=RGBColor(0x0D, 0x1B, 0x2A))
        txt(s, "⏱  " + effort,
            lft + Inches(0.2), Inches(2.54), w - Inches(0.4), Inches(0.34),
            size=Pt(13), color=col, bold=True)

        txt(s, body,
            lft + Inches(0.15), Inches(3.02), w - Inches(0.3), Inches(2.3),
            size=Pt(13.5), color=WHITE)

        # tagline footer
        box(s, lft, Inches(6.72), w, Inches(0.48),
            fill_color=RGBColor(0x0A, 0x18, 0x28))
        txt(s, tagline,
            lft + Inches(0.15), Inches(6.76), w - Inches(0.3), Inches(0.42),
            size=Pt(12), color=col, italic=True)

    txt(s, "Recommendation: start with A + B in parallel (almost free), use C as a stretch goal",
        Inches(0.35), Inches(7.1), Inches(12.6), Inches(0.35),
        size=Pt(14), bold=True, color=WHITE)


def slide_next_steps_benchmarks(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "5  Next Steps", "Immediate — Latency Benchmark All 6 VLMs")

    box(s, Inches(0.35), Inches(1.85), Inches(12.6), Inches(0.9),
        fill_color=RGBColor(0x0A, 0x1E, 0x38))
    txt(s, "We have accuracy numbers but no latency numbers. "
           "Run a systematic latency benchmark (ms/clip) across all 6 evaluated VLMs in zero-shot and LoRA mode "
           "to build the Pareto curve needed for the efficiency contribution.",
        Inches(0.55), Inches(1.95), Inches(12.2), Inches(0.75),
        size=Pt(14.5), color=WHITE)

    # Left: what to measure
    box(s, Inches(0.35), Inches(2.9), Inches(6.1), Inches(3.85),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    txt(s, "What to measure per model",
        Inches(0.5), Inches(2.98), Inches(5.8), Inches(0.42),
        size=Pt(15), bold=True, color=ACCENT)
    bullet_column(s, [
        "Wall-clock ms/clip (median over 100 clips, warm GPU)",
        "Zero-shot mode vs. LoRA-adapted mode",
        "KABR and Baboonland val splits separately",
        "Peak GPU memory (MB) as secondary metric",
        "Throughput: clips/second at batch size 1",
    ], Inches(0.5), Inches(3.45), Inches(5.8), Inches(3.1),
       size=Pt(13.5), bullet="▸ ")

    # Right: models + effort
    box(s, Inches(6.6), Inches(2.9), Inches(6.35), Inches(3.85),
        fill_color=RGBColor(0x12, 0x24, 0x36))
    txt(s, "Models to benchmark",
        Inches(6.75), Inches(2.98), Inches(6.05), Inches(0.42),
        size=Pt(15), bold=True, color=ACCENT2)
    models = [
        ("Qwen2.5-VL-3B",   "3B params — primary LoRA model"),
        ("Qwen2.5-VL-7B",   "7B params — larger baseline"),
        ("LLaVA-OV-0.5B",   "0.5B — smallest adapted VLM"),
        ("LLaVA-OV-7B",     "7B — top Baboonland performer"),
        ("Gemma3-4B",        "4B — top KABR performer"),
        ("MiniCPM-V 4.6",   "best zero-shot small VLM"),
    ]
    for i, (name, note) in enumerate(models):
        y = Inches(3.45) + i * Inches(0.55)
        box(s, Inches(6.75), y, Inches(2.55), Inches(0.48),
            fill_color=RGBColor(0x16, 0x2A, 0x40))
        txt(s, name, Inches(6.8), y + Inches(0.04), Inches(2.45), Inches(0.42),
            size=Pt(12), bold=True, color=WHITE)
        txt(s, note, Inches(9.4), y + Inches(0.04), Inches(3.45), Inches(0.42),
            size=Pt(12), color=GRAY)

    box(s, Inches(0.35), Inches(6.88), Inches(12.6), Inches(0.48),
        fill_color=RGBColor(0x00, 0x50, 0x70))
    txt(s, "Estimated effort:  1–2 days of scripting + runs  →  "
           "directly enables Option A Pareto plot and Option B multi-objective framing",
        Inches(0.55), Inches(6.93), Inches(12.2), Inches(0.4),
        size=Pt(13.5), color=WHITE)


def slide_next_steps_plan(prs):
    s = blank_slide(prs)
    bg(s)
    section_header(s, "5  Next Steps", "Short-Term Plan")

    steps = [
        (ACCENT,  "This week",
         [
             "Add torch.cuda.Event timer to eval script — record ms/clip per model",
             "Run latency benchmark: all 6 VLMs × zero-shot + LoRA × KABR + Baboonland val",
             "Check A3 (joint + inv-sqrt) eval results in task DB — are runs complete?",
         ]),
        (ACCENT2, "Next 2–3 weeks",
         [
             "Token budget sweep on existing Qwen2.5-VL-3B adapter (Option A)",
             "Vary max_pixels: 64 / 128 / 256 / 512 / 1024 tok/frame at inference",
             "Plot accuracy vs. latency Pareto curve — no new training needed",
             "Add latency logging to NAS grid eval runs (Option B, ~1 line of code)",
         ]),
        (GREEN,   "Stretch goal (4–6 weeks)",
         [
             "Adaptive frame selection using SigLIP CLS cosine similarity (Option C)",
             "Prototype: encode N frames, drop K most-redundant before LLM",
             "Evaluate on low-motion behaviors (grooming, resting) vs. high-motion",
         ]),
    ]

    for i, (col, label, bullets) in enumerate(steps):
        top = Inches(1.72) + i * Inches(1.85)
        box(s, Inches(0.35), top, Inches(12.6), Inches(1.75),
            fill_color=RGBColor(0x12, 0x24, 0x36))
        box(s, Inches(0.35), top, Inches(2.1), Inches(1.75), fill_color=col)
        txt(s, label,
            Inches(0.35), top, Inches(2.1), Inches(1.75),
            size=Pt(14), bold=True, color=BG, align=PP_ALIGN.CENTER)
        bullet_column(s, bullets,
                      Inches(2.6), top + Inches(0.15), Inches(10.15), Inches(1.5),
                      size=Pt(13.5), bullet="▸ ")

    txt(s, "Start A + B in parallel immediately — both build on existing code. Use C as a follow-on if time allows.",
        Inches(0.35), Inches(7.15), Inches(12.6), Inches(0.3),
        size=Pt(13), color=GRAY, italic=True)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    prs = new_prs()

    # ── Title ──────────────────────────────────────────────────────────────
    slide_title(prs)

    # ── Section 1: Background & Methods ───────────────────────────────────
    slide_section(prs, 1, "Background & Methods",
                  "Motivation · Datasets · Pipeline · Models",
                  ["Why adapt VLMs for wildlife behavior?",
                   "Two clip-level datasets (KABR, Baboonland)",
                   "3-stage pipeline: frames → clips → VLM-ready JSONL",
                   "6 VLM backbones + X3D-L / I3D-L specialist video models"])
    slide_motivation(prs)
    slide_datasets(prs)
    slide_dataset_construction(prs)
    slide_models(prs)

    # ── Section 2: Results ────────────────────────────────────────────────
    slide_section(prs, 2, "Results",
                  "Accuracy across models, datasets, and training regimes",
                  ["KABR-Clip: adapted VLMs beat X3D-L",
                   "Baboonland-Clip: VLMs dominate specialist baselines",
                   "Joint cross-dataset LoRA (Contribution A)"])
    slide_kabr_results(prs)
    slide_baboonland_results(prs)
    slide_contributions(prs)

    # ── Section 3: Efficiency ─────────────────────────────────────────────
    slide_section(prs, 3, "Efficiency",
                  "Literature review + planned contribution",
                  ["How all 6 evaluated VLMs handle visual tokens",
                   "Common techniques & what matters for 8-frame clips",
                   "Planned contribution: latency–accuracy Pareto + token budget sweep"])
    slide_litreview_table(prs)
    slide_litreview_common(prs)
    slide_litreview_gaps(prs)

    # ── Section 4: Proposed Work ──────────────────────────────────────────
    slide_section(prs, 4, "Proposed Work",
                  "Three options for an efficiency contribution",
                  ["Overview: effort vs. impact",
                   "Option A — Token budget sweep (2–3 weeks)",
                   "Option B — Add latency to NAS grid (~1 extra day)",
                   "Option C — Adaptive frame selection (4–6 weeks)"])
    slide_options_overview(prs)
    slide_option_a(prs)
    slide_option_b(prs)
    slide_option_c(prs)

    # ── Section 5: Next Steps ─────────────────────────────────────────────
    slide_section(prs, 5, "Next Steps",
                  "Concrete actions to move the project forward",
                  ["Run latency benchmark for all 6 VLMs (zero-shot + LoRA)",
                   "Short-term plan: benchmarks → Option A sweep → Option B timer"])
    slide_next_steps_benchmarks(prs)
    slide_next_steps_plan(prs)

    # ── Summary ───────────────────────────────────────────────────────────
    slide_summary(prs)

    out = "/scratch/general/vast/j.yan/myproj1/wildlife_vlm_presentation.pptx"
    prs.save(out)
    print(f"Saved → {out}  ({len(prs.slides)} slides)")

if __name__ == "__main__":
    main()
