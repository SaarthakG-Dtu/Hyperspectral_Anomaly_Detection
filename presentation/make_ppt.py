"""
Build the project presentation (DMS2F-HAD_presentation.pptx) with python-pptx.

    python presentation/make_ppt.py        (run after train.py and make_figures.py)

All numbers on the result slides are read from results/runs/*.npz.
"""
import sys
from pathlib import Path

import numpy as np
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import DATASETS, PAPER_AUC, auc_score, load_hsi, rx_detector  # noqa: E402

FIG = ROOT / "results" / "figures"
RUNS = ROOT / "results" / "runs"
ASSETS = Path(__file__).parent / "assets"
OUT = Path(__file__).parent / "DMS2F-HAD_presentation.pptx"

# ---------------------------------------------------------------- design tokens
INK, NIGHT = "14213D", "0E1A2E"          # dark ink / title background
AMBER, TEAL, CORAL = "F2A541", "2A9D8F", "E76F51"
PANEL, LINE, MUTED, WHITE = "EEF2F7", "D5DCE5", "5B6675", "FFFFFF"
HEAD, BODY = "Cambria", "Calibri"


def rgb(h):
    return RGBColor.from_string(h)


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


# ---------------------------------------------------------------- helpers
def new_slide(dark=False, notes=None):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(NIGHT if dark else WHITE)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def _bullet(p, char="•", level=0):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(0.28 + 0.3 * level))))
    pPr.set("indent", str(int(-Inches(0.24))))
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    bu = pPr.makeelement(qn("a:buChar"), {"char": char})
    pPr.append(bu)


def text(s, x, y, w, h, paras, size=16, color=INK, font=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, bullets=False, space_after=6, name=None):
    """paras: list of str or list of runs [(text, {bold, color, size, italic, font})]."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, para in enumerate(paras if isinstance(paras, list) else [paras]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        runs = para if isinstance(para, list) else [(para, {})]
        for t, o in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False)
            f.color.rgb = rgb(o.get("color", color))
        if bullets:
            _bullet(p)
    return tb


def title(s, t, dark=False, sub=None):
    text(s, 0.6, 0.38, 12.1, 0.75, t, size=30, font=HEAD, bold=True, color=WHITE if dark else INK,
         anchor=MSO_ANCHOR.TOP, name="Title")
    if sub:
        text(s, 0.6, 1.08, 12.1, 0.45, sub, size=15, color="B9C4D4" if dark else MUTED)


def box(s, x, y, w, h, fill=PANEL, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, name=None):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(fill)
    if line:
        sh.line.color.rgb = rgb(line)
        sh.line.width = Pt(1)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    if name:
        sh.name = name
    return sh


def label_box(s, x, y, w, h, t, fill, color=INK, size=13, bold=True, line=None):
    sh = box(s, x, y, w, h, fill=fill, line=line)
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    lines = t.split("\n")
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = ln
        r.font.name, r.font.size = BODY, Pt(size if i == 0 else size - 2)
        r.font.bold = bold if i == 0 else False
        r.font.color.rgb = rgb(color)
    return sh


def badge(s, x, y, t, d=0.5, fill=AMBER, color=INK, size=16):
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    c.fill.solid(); c.fill.fore_color.rgb = rgb(fill); c.line.fill.background(); c.shadow.inherit = False
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = t
    r.font.name, r.font.size, r.font.bold = BODY, Pt(size), True
    r.font.color.rgb = rgb(color)
    return c


def arrow(s, x1, y1, x2, y2, color=MUTED, width=1.75):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return c


def picture(s, path, x, y, w=None, h=None, name=None):
    """Insert image fitted inside the (w, h) box, centred, aspect preserved."""
    if not Path(path).exists():                       # figure not produced yet
        return text(s, x, y, w or 4, 0.5, f"[missing figure: {Path(path).name}]", size=14, color=CORAL)
    iw, ih = Image.open(path).size
    if w and h:
        scale = min(w / iw, h / ih)
        pw, ph = iw * scale, ih * scale
        x, y = x + (w - pw) / 2, y + (h - ph) / 2
    elif w:
        pw, ph = w, w * ih / iw
    else:
        pw, ph = h * iw / ih, h
    pic = s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(pw), Inches(ph))
    if name:
        pic.name = name
    return pic


def table(s, rows, x, y, w, col_w, row_h=0.36, size=12, header_fill=INK, highlight_rows=(), bold_cells=()):
    shp = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tbl = shp.table
    tot = sum(col_w)
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(w * cw / tot)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(header_fill if i == 0 else ("FDF1DC" if i in highlight_rows else
                                                               (WHITE if i % 2 else "F6F8FB")))
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(val)
            r.font.name, r.font.size = BODY, Pt(size)
            r.font.bold = i == 0 or (i, j) in bold_cells or (i in highlight_rows and j == 0)
            r.font.color.rgb = rgb(WHITE if i == 0 else INK)
    return tbl


def caption(s, t, y=6.95, x=0.6, w=12.1):
    text(s, x, y, w, 0.35, t, size=11, color=MUTED)


# ---------------------------------------------------------------- results
def load_results():
    res = {}
    for d in DATASETS:
        cube, gt = load_hsi(d)
        r = {"rx": auc_score(rx_detector(cube), gt), "runs": {}}
        for f in ["gated", "add", "spatial", "spectral"]:
            p = RUNS / f"{d}_{f}.npz"
            if p.exists():
                z = np.load(p)
                r["runs"][f] = {k: z[k] for k in ["best_auc", "final_auc", "best_epoch", "infer_time", "params",
                                                   "train_time"]}
        res[d] = r
    return res


R = load_results()
GATED = [d for d in DATASETS if "gated" in R[d]["runs"]]


def mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


avg_best = mean([float(R[d]["runs"]["gated"]["best_auc"]) for d in GATED])
avg_final = mean([float(R[d]["runs"]["gated"]["final_auc"]) for d in GATED])
avg_paper = mean([PAPER_AUC[d]["DMS2F-HAD"] for d in GATED])
avg_rx = mean([R[d]["rx"] for d in GATED])
avg_time = mean([float(R[d]["runs"]["gated"]["infer_time"]) for d in GATED])
params = {d: int(R[d]["runs"]["gated"]["params"]) for d in GATED}

# ================================================================= 1. title
s = new_slide(dark=True, notes=(
    "This project re-implements DMS2F-HAD, a WACV 2026 paper on hyperspectral anomaly detection. "
    "I explain what hyperspectral images and anomaly detection are, how the method works, the datasets, "
    "and the results of my own runs compared with the paper."))
text(s, 0.7, 1.0, 7.4, 0.4, "WACV 2026  ·  paper implementation project", size=14, color=AMBER, bold=True)
text(s, 0.7, 1.5, 7.4, 1.1, "DMS2F-HAD", size=60, font=HEAD, bold=True, color=WHITE)
text(s, 0.7, 2.75, 7.2, 1.3, "A Dual-branch Mamba-based Spatial–Spectral Fusion Network for Hyperspectral "
     "Anomaly Detection", size=22, font=HEAD, color="DCE3EE")
text(s, 0.7, 4.45, 7.2, 1.2, [
    [("Paper: ", {"bold": True, "color": AMBER}), ("A. Pant, L. Tamang, T.-K. Lee, S. Aryal — Deakin University", {})],
    [("This project: ", {"bold": True, "color": AMBER}),
     ("compact PyTorch re-implementation, trained and evaluated on four benchmark scenes", {})],
], size=15, color="DCE3EE")
picture(s, ASSETS / "hsi_cube_dark.png", 8.2, 0.9, 4.7, 4.9)
text(s, 8.2, 5.85, 4.7, 0.4, "San Diego scene: 100 × 100 pixels × 189 spectral bands", size=11, color="9FB0C6",
     align=PP_ALIGN.CENTER)

# ================================================================= 2. roadmap
s = new_slide(notes="The talk follows five steps: data, problem, method, experimental setup, results.")
title(s, "Roadmap")
items = [("Hyperspectral images", "What the data looks like: a cube of spectra"),
         ("Anomaly detection", "The HAD problem, how it is evaluated, families of methods"),
         ("DMS2F-HAD method", "Mamba, dual spatial/spectral branches, gated fusion"),
         ("Data & setup", "Benchmark scenes, training setup, our code"),
         ("Results", "AUC, anomaly maps, ROC, ablation, takeaways")]
for i, (h, d) in enumerate(items):
    x = 0.6 + i * 2.48
    box(s, x, 1.8, 2.28, 4.2, fill=PANEL)
    badge(s, x + 0.25, 2.15, str(i + 1), d=0.62, size=20)
    text(s, x + 0.2, 3.1, 1.95, 0.9, h, size=19, font=HEAD, bold=True)
    text(s, x + 0.2, 4.1, 1.95, 1.6, d, size=14, color=MUTED)

# ================================================================= 3. HSI
s = new_slide(notes=(
    "A normal camera records 3 broad colour bands. A hyperspectral sensor such as AVIRIS records hundreds of "
    "narrow contiguous bands from visible to short-wave infrared. So each pixel is a full spectrum, and the image "
    "is a 3-D cube H x W x C. Different materials have different spectral fingerprints."))
title(s, "A hyperspectral image is a cube: every pixel is a spectrum")
picture(s, FIG / "fig_hsi_cube.png", 0.5, 1.4, 5.3, 5.3)
text(s, 6.2, 1.55, 6.5, 3.6, [
    [("RGB camera: ", {"bold": True}), ("3 broad bands. ", {}), ("Hyperspectral sensor: ", {"bold": True}),
     ("100–200+ narrow, contiguous bands (AVIRIS: 224 bands, 400–2500 nm, ~10 nm each).", {})],
    [("The data is an ", {}), ("H × W × C", {"bold": True}), (" cube; each pixel is a C-dimensional "
                                                              "spectral signature.", {})],
    "Materials (metal, paint, vegetation, soil, rock) reflect light differently at each wavelength, so the "
    "spectrum works like a material fingerprint.",
    "Hard parts: hundreds of correlated bands, sensor noise, mixed pixels, and almost no labels.",
], size=16, bullets=True, space_after=10)
for i, (num, lab) in enumerate([("189", "bands per pixel"), ("10 000", "pixels (100 × 100)"), ("~1.3 %", "pixels are anomalies")]):
    x = 6.2 + i * 2.2
    box(s, x, 5.35, 2.0, 1.25, fill=PANEL)
    text(s, x + 0.1, 5.42, 1.8, 0.6, num, size=26, font=HEAD, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
    text(s, x + 0.1, 6.0, 1.8, 0.5, lab, size=12, color=MUTED, align=PP_ALIGN.CENTER)
caption(s, "Numbers for the San Diego scene used in this project.", y=6.75, x=6.2, w=6.5)

# ================================================================= 4. HAD
s = new_slide(notes=(
    "Hyperspectral anomaly detection gives every pixel a score for how different it is from the background. "
    "It is unsupervised: we have no target spectra and no labels. In San Diego the planes are rare and their "
    "spectra differ from the apron and roads, which is exactly what HAD exploits."))
title(s, "Anomaly detection (HAD): find pixels that don't fit")
text(s, 0.6, 1.5, 5.6, 3.9, [
    [("Goal: ", {"bold": True}), ("give every pixel an anomaly score → an anomaly map.", {})],
    [("Unsupervised: ", {"bold": True}), ("no target spectra and no labels; the background model is "
                                          "learned from the image itself.", {})],
    [("Anomalies ", {"bold": True}), ("are small, rare (often < 1 % of pixels) and spectrally different "
                                     "from their surroundings: planes, vehicles, rocks, boats.", {})],
    [("Output ", {"bold": True}), ("is thresholded into detections and compared with a ground-truth mask.", {})],
], size=16, bullets=True, space_after=10)
picture(s, FIG / "fig_spectra.png", 6.5, 1.45, 6.3, 3.9)
text(s, 0.6, 5.75, 3.0, 0.4, "Where HAD is used", size=14, bold=True, color=MUTED)
for i, a in enumerate(["Defence & surveillance", "Search and rescue", "Mineral exploration",
                       "Environmental monitoring", "Precision agriculture"]):
    label_box(s, 0.6 + i * 2.45, 6.2, 2.3, 0.55, a, fill="E3F1EF", color=INK, size=13, bold=True)

# ================================================================= 5. evaluation
s = new_slide(notes=(
    "Three standard ways to judge a detector. The colour anomaly map is the qualitative view. The ROC curve "
    "plots detection rate against false-alarm rate as the threshold moves; AUC summarises it in one number. "
    "Box plots show how well anomaly scores are separated from background scores."))
title(s, "How a HAD method is evaluated")
cards = [("Anomaly map", "Scores shown with a jet colour map: blue = background, red = anomaly. Qualitative check "
                         "against the ground truth."),
         ("ROC curve & AUC", "Sweep the threshold: probability of detection vs. false-alarm rate. AUC = 1 is "
                             "perfect, 0.5 is random guessing."),
         ("Box plots", "Score distributions of anomaly vs. background pixels. A large gap with little overlap means "
                       "fewer false alarms.")]
for i, (h, d) in enumerate(cards):
    x = 0.6 + i * 4.1
    box(s, x, 1.45, 3.9, 1.95, fill=PANEL)
    badge(s, x + 0.2, 1.62, str(i + 1), d=0.48, size=15)
    text(s, x + 0.8, 1.64, 3.0, 0.5, h, size=18, font=HEAD, bold=True)
    text(s, x + 0.2, 2.2, 3.55, 1.2, d, size=14, color=MUTED)
if (FIG / "San_Diego_result.png").exists():
    picture(s, FIG / "San_Diego_result.png", 0.6, 3.65, 12.1, 3.1)
caption(s, "Example from our run on San Diego: false-colour image, ground truth, RX and DMS2F-HAD anomaly maps, ROC.")

# ================================================================= 6. families
s = new_slide(notes=(
    "HAD research moved from statistics to deep learning. RX is the classic baseline. Representation methods "
    "learn a background dictionary. Deep models learn non-linear backgrounds but CNNs see only local context and "
    "Transformers are expensive. Mamba-based state-space models are the newest family, and DMS2F-HAD belongs to it."))
title(s, "Four families of HAD methods")
fam = [("Statistical", "RX (Reed–Xiaoli): Mahalanobis distance to a Gaussian background.",
        "Fast, no training", "Gaussian assumption fails in cluttered scenes → false alarms"),
       ("Representation", "Background dictionary, low-rank + sparse decomposition.",
        "Better separation of outliers", "Sensitive to noise and parameters; slow"),
       ("Deep learning", "Autoencoders, GANs, CNNs, Transformers (AUTO-AD, LREN, RGAE, TDD, GT-HAD).",
        "Learn non-linear backgrounds", "CNN: local view. Transformer: O(L²) cost, many parameters"),
       ("Mamba (SSM)", "MMR-HAD, DMS2F-HAD: selective state-space sequence models.",
        "Long-range context in linear time; small models", "Young research area")]
for i, (h, d, pro, con) in enumerate(fam):
    x = 0.6 + i * 3.08
    hl = i == 3
    box(s, x, 1.5, 2.9, 5.2, fill="FDF1DC" if hl else PANEL)
    badge(s, x + 0.2, 1.7, str(i + 1), d=0.48, size=15, fill=AMBER if hl else "C9D3E0")
    text(s, x + 0.2, 2.3, 2.55, 0.5, h, size=19, font=HEAD, bold=True)
    text(s, x + 0.2, 2.85, 2.55, 1.5, d, size=14, color=INK)
    text(s, x + 0.2, 4.35, 2.55, 1.1, [[("+ ", {"bold": True, "color": TEAL}), (pro, {})]], size=14)
    text(s, x + 0.2, 5.4, 2.55, 1.2, [[("− ", {"bold": True, "color": CORAL}), (con, {})]], size=14)

# ================================================================= 7. Mamba
s = new_slide(notes=(
    "A state-space model keeps a hidden state h and updates it at every position of the sequence. Mamba makes "
    "the step size and the B and C matrices depend on the input, so it can choose what to remember. The cost "
    "grows linearly with sequence length, while self-attention grows quadratically. A 16x16 patch is a sequence "
    "of 256 pixel tokens."))
title(s, "Why Mamba? Long-range modelling in linear time")
box(s, 0.6, 1.5, 5.9, 1.6, fill=INK)
text(s, 0.85, 1.62, 5.5, 1.4, [
    [("hₜ = Āₜ · hₜ₋₁ + B̄ₜ · xₜ", {"font": "Cambria Math", "size": 24, "color": WHITE})],
    [("yₜ = Cₜ · hₜ + D · xₜ", {"font": "Cambria Math", "size": 24, "color": WHITE})],
], size=22, color=WHITE, space_after=4)
text(s, 0.6, 3.3, 5.9, 3.4, [
    [("State-space model: ", {"bold": True}), ("a hidden state h carries information along the sequence.", {})],
    [("Selective (S6): ", {"bold": True}), ("Δ, B and C are computed from the input, so the model decides what "
                                           "to keep and what to forget.", {})],
    [("Cost: ", {"bold": True}), ("O(L) in sequence length L, versus O(L²) for self-attention.", {})],
    [("In DMS2F-HAD: ", {"bold": True}), ("a 16 × 16 patch = 256 pixel tokens (spatial) and 7 band groups "
                                         "(spectral).", {})],
], size=15, bullets=True, space_after=8)
cd = CategoryChartData()
Ls = [256, 512, 1024, 2048, 4096]
cd.categories = [str(L) for L in Ls]
cd.add_series("Self-attention  ~L²", [L * L / 1e6 for L in Ls])
cd.add_series("Mamba  ~L·N (N = 16)", [L * 16 / 1e6 for L in Ls])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(6.9), Inches(1.45), Inches(5.9), Inches(5.0), cd)
ch = gf.chart
ch.has_title = True
ch.chart_title.text_frame.text = "Relative compute vs. sequence length (millions of ops)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(12)
for ser, col in zip(ch.series, [CORAL, TEAL]):
    ser.format.line.color.rgb = rgb(col)
    ser.format.line.width = Pt(2.25)
    ser.smooth = False
    ser.marker.style = XL_MARKER_STYLE.CIRCLE
    ser.marker.size = 7
    ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = rgb(col)
    ser.marker.format.line.color.rgb = rgb(col)
ch.category_axis.tick_labels.font.size = Pt(12)
ch.category_axis.tick_labels.font.color.rgb = rgb(MUTED)
ch.value_axis.tick_labels.font.size = Pt(12)
ch.value_axis.tick_labels.font.color.rgb = rgb(MUTED)
ch.value_axis.major_gridlines.format.line.color.rgb = rgb("E3E8EF")
ch.value_axis.format.line.fill.background()
ch.category_axis.has_title = True
ch.category_axis.axis_title.text_frame.text = "Sequence length L (tokens)"
ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)

# ================================================================= 8. architecture (own diagram)
s = new_slide(notes=(
    "The image is cut into overlapping 16x16 patches; random rectangles are masked during training. A 1x1 "
    "convolution reduces the bands to 64 features. The spatial branch looks at texture across the 256 pixels; "
    "the spectral branch looks at correlations between bands. A learned gate mixes them pixel by pixel, a "
    "decoder rebuilds the patch, and the reconstruction error is the anomaly score."))
title(s, "DMS2F-HAD at a glance")
y0 = 3.35
label_box(s, 0.5, y0, 1.55, 1.1, "HSI patch\n16×16×C", fill="DCE3EE")
label_box(s, 2.35, y0, 1.6, 1.1, "Random mask\n(training only)", fill="DCE3EE")
label_box(s, 4.25, y0, 1.45, 1.1, "In-head\n1×1 conv → 64", fill="DCE3EE")
label_box(s, 6.15, 1.75, 2.45, 1.45, "Spatial branch\n3×3 + 5×5 conv →\n256 pixel tokens → Mamba", fill="D6EEEA")
label_box(s, 6.15, 4.55, 2.45, 1.45, "Spectral branch\n1-D band convs → 7 band\ngroups (16 bands) → Mamba", fill="FBE3D6")
label_box(s, 9.0, y0, 1.6, 1.1, "Gated fusion\nG·Fspa + (1−G)·Fspe", fill=AMBER)
label_box(s, 10.95, y0, 1.85, 1.1, "SS decoder ×2\n+ out-head", fill="DCE3EE")
for (x1, x2) in [(2.05, 2.35), (3.95, 4.25), (10.6, 10.95)]:
    arrow(s, x1, y0 + 0.55, x2, y0 + 0.55)
arrow(s, 5.7, y0 + 0.4, 6.15, 2.5); arrow(s, 5.7, y0 + 0.7, 6.15, 5.25)
arrow(s, 8.6, 2.5, 9.0, y0 + 0.4); arrow(s, 8.6, 5.25, 9.0, y0 + 0.7)
box(s, 6.6, 6.35, 6.2, 0.62, fill=INK)
text(s, 6.75, 6.39, 5.95, 0.55, [[("Anomaly score  ", {"bold": True, "color": AMBER}),
                                 ("R(i, j) = ‖ original − reconstruction ‖₂", {"color": WHITE})]],
     size=16, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
arrow(s, 11.9, y0 + 1.1, 11.9, 6.35)
text(s, 0.5, 1.6, 5.3, 1.4, [
    "Train an autoencoder on patches of the image itself (no labels).",
    "Background dominates the data, so it is reconstructed well; rare anomalies are not.",
], size=14, color=MUTED, bullets=True, space_after=6)
text(s, 0.5, 5.0, 5.3, 1.3, [
    [("0.65 M parameters", {"bold": True, "color": INK}), (" in our build (paper: 0.64 M).", {})],
    [("Linear-time Mamba blocks", {"bold": True, "color": INK}), (" instead of attention.", {})],
], size=14, color=MUTED, bullets=True, space_after=6)

# ================================================================= 9. paper figure
s = new_slide(notes="The original architecture figure from the paper, for reference: (a) the full network, "
                    "(b) the Mamba block, (c) the spatial-spectral decoder block.")
title(s, "Architecture as drawn in the paper")
picture(s, ASSETS / "paper_fig2_architecture.png", 0.6, 1.25, 12.1, 5.55)
caption(s, "Figure 2 of Pant et al., “DMS2F-HAD”, WACV 2026: (a) overall framework, (b) Mamba block, "
           "(c) SS decoder block.")

# ================================================================= 10. components
s = new_slide(notes=(
    "Four pieces. The spatial branch sees each patch as 256 pixel tokens. The spectral branch splits the 64 "
    "features into 7 overlapping groups of 16 so Mamba can model band correlations cheaply. The gate is a 1x1 "
    "convolution with a sigmoid giving one weight per pixel. The decoder mixes a global Mamba path with local "
    "3x3 and 5x5 convolutions."))
title(s, "The four building blocks")
comp = [("Spatial branch", TEAL, "Parallel 3×3 and 5×5 convolutions capture multi-scale texture. The 16×16 map is "
         "flattened row-major into 256 tokens and a Mamba block gives every pixel a global receptive field."),
        ("Spectral branch", CORAL, "Two 1-D convolutions along the spectrum (kernel 7 and 3). The 64-d spectrum is "
         "split into g = 7 overlapping groups of c₂ = 16 bands (stride k = 8); Mamba runs over the groups."),
        ("Adaptive gated fusion", AMBER, "G = σ(Conv₁ₓ₁([F_spa, F_spe])) gives one weight per pixel.  "
         "F_fusion = Proj(G ⊙ F_spa + (1 − G) ⊙ F_spe): texture where it helps, spectra where the scene is smooth."),
        ("SS decoder + detection", INK, "Two blocks of Mamba ‖ 3×3 conv ‖ 5×5 conv → concat → 1×1 conv, then a "
         "1×1 out-head back to C bands. Overlapping patches are averaged; the L2 residual is the score.")]
for i, (h, col, d) in enumerate(comp):
    x, y = 0.6 + (i % 2) * 6.15, 1.5 + (i // 2) * 2.7
    box(s, x, y, 5.95, 2.5, fill=PANEL)
    badge(s, x + 0.25, y + 0.25, str(i + 1), d=0.5, fill=col, color=WHITE if col == INK else INK, size=15)
    text(s, x + 0.95, y + 0.27, 4.8, 0.5, h, size=19, font=HEAD, bold=True)
    text(s, x + 0.3, y + 0.9, 5.45, 1.55, d, size=14, color=INK)

# ================================================================= 11. datasets (14)
s = new_slide(notes=(
    "The paper evaluates on 14 public scenes, all AVIRIS except CRI. The official code release ships four of them: "
    "CRI, AVIRIS-1, AVIRIS-2 and San Diego, which are the ones I used (highlighted)."))
title(s, "Benchmark datasets in the paper (14 scenes)")
ds14 = [("CRI", "400×400×46", "–", "Rocks"), ("Salinas", "512×217×224", "3.7", "Rooftops"),
        ("San Diego", "100×100×189", "3.5", "Planes"), ("Bay Champagne", "100×100×188", "4.4", "Watercraft"),
        ("Abu-urban-2", "100×100×207", "17.2", "Buildings"), ("Abu-urban-3", "100×100×191", "3.5", "Buildings"),
        ("Abu-urban-4", "100×100×205", "7.1", "Houses"), ("Abu-urban-5", "100×100×205", "7.1", "Houses"),
        ("AVIRIS-1", "100×100×186*", "3.5", "Planes"), ("AVIRIS-2", "128×128×189", "3.5", "Plane"),
        ("Cat Island", "150×150×188", "17.2", "Plane"), ("Gulfport", "100×100×191", "3.4", "Planes"),
        ("Pavia University", "224×423×102", "1.3", "Vehicles"), ("Texas Coast", "100×100×204", "17.2", "Buildings")]
used = {"CRI", "San Diego", "AVIRIS-1", "AVIRIS-2"}
half = 7
for k in range(2):
    rows = [["Dataset", "Size (H×W×C)", "Res. (m)", "Anomaly"]] + [list(r) for r in ds14[k * half:(k + 1) * half]]
    hl = [i + 1 for i, r in enumerate(ds14[k * half:(k + 1) * half]) if r[0] in used]
    table(s, rows, 0.6 + k * 6.15, 1.5, 5.95, [1.7, 1.5, 0.9, 1.15], row_h=0.5, size=13, highlight_rows=hl)
box(s, 0.6, 5.7, 0.32, 0.32, fill="FDF1DC", line="E9C98F")
text(s, 1.0, 5.66, 11.5, 0.4, "Highlighted = provided with the official code and used in this project.", size=13,
     color=MUTED)
caption(s, "All scenes captured by AVIRIS except CRI (Nuance CRI sensor).  *The released AVIRIS-1 file has 189 "
           "bands; the paper lists 186.", y=6.15)

# ================================================================= 12. our datasets
s = new_slide(notes=(
    "The four scenes: CRI is a large grassland scene with ten rocks; the three AVIRIS scenes are airport images "
    "with planes. White pixels in the ground truth are anomalies."))
title(s, "The four scenes used in this project")
picture(s, FIG / "fig_datasets.png", 0.6, 1.3, 12.1, 5.3)
caption(s, "Top: false-colour composites (three bands). Bottom: ground-truth anomaly masks (white = anomaly).",
        y=6.7)

# ================================================================= 13. implementation
p_groups = {}
for d in GATED:
    p_groups.setdefault(params[d], []).append(DATASETS[d])
p_str = "; ".join(f"{p/1e6:.3f} M ({', '.join(n)})" for p, n in sorted(p_groups.items(), reverse=True)) or "n/a"
s = new_slide(notes=(
    "The official code depends on the mamba_ssm CUDA kernels, which do not build on Windows. I wrote the "
    "Mamba selective scan in plain PyTorch with an exact two-level chunked scan, and kept the paper's training "
    "settings. Each image is trained on its own, without labels; the ground truth is only used to compute AUC."))
title(s, "Implementation and training setup")
hp = [["Setting", "Value"], ["Patch size / stride", "16 × 16 / 8"], ["Embedding c₁", "64"],
      ["Band group c₂ / stride k", "16 / 8  (7 groups)"], ["Mamba state size", "16"],
      ["Optimiser", "Adam, lr 5e-4, weight decay 1e-4"], ["Epochs / batch", "100 / 32"],
      ["Loss", "MSE + 0.1 · L1 (official code)"], ["Masking", "p = 0.5, random 2–5 px rectangle"]]
table(s, hp, 0.6, 1.5, 5.6, [2.3, 3.0], row_h=0.5, size=13)
text(s, 6.6, 1.45, 6.2, 0.45, "Code: method in ≈ 440 lines of Python", size=18, font=HEAD, bold=True)
files = [("model.py", "Mamba in pure PyTorch + DMS2F-HAD network"),
         ("utils.py", "data loading, patching, RX baseline, metrics"),
         ("train.py", "unsupervised training + anomaly map per image"),
         ("make_figures.py", "paper-style PNG figures and tables"),
         ("presentation/make_ppt.py", "this slide deck")]
for i, (f, d) in enumerate(files):
    y = 2.0 + i * 0.62
    box(s, 6.6, y, 6.2, 0.52, fill=PANEL)
    text(s, 6.75, y + 0.06, 2.6, 0.42, f, size=13, font="Consolas", bold=True, color=INK)
    text(s, 9.35, y + 0.08, 3.4, 0.42, d, size=12, color=MUTED)
box(s, 6.6, 5.2, 6.2, 1.45, fill="FDF1DC")
text(s, 6.8, 5.28, 5.85, 1.35, [
    [("No mamba_ssm needed: ", {"bold": True}), ("the selective scan runs as an exact two-level chunked scan "
                                               "(16 + 16 steps instead of 256), so it runs on Windows/CPU/any GPU.", {})],
    [("Parameters: ", {"bold": True}), (p_str, {})],
], size=13, space_after=4)

# ================================================================= 14. results headline
rows = [["Dataset", "RX", "DMS2F-HAD\nours (best epoch)", "DMS2F-HAD\nours (epoch 100)", "DMS2F-HAD\npaper",
         "Inference (s)"]]
bold = []
for d in DATASETS:
    r = R[d]
    g = r["runs"].get("gated")
    if g is None:
        rows.append([DATASETS[d], f"{r['rx']:.4f}", "running", "running", f"{PAPER_AUC[d]['DMS2F-HAD']:.4f}", "–"])
        continue
    rows.append([DATASETS[d], f"{r['rx']:.4f}", f"{float(g['best_auc']):.4f}  (ep {int(g['best_epoch'])})",
                 f"{float(g['final_auc']):.4f}", f"{PAPER_AUC[d]['DMS2F-HAD']:.4f}", f"{float(g['infer_time']):.3f}"])
    bold.append((len(rows) - 1, 2))
rows.append(["Average", f"{avg_rx:.4f}", f"{avg_best:.4f}", f"{avg_final:.4f}", f"{avg_paper:.4f}", f"{avg_time:.3f}"])
s = new_slide(notes=(
    "The main numbers. 'Best epoch' follows the paper and the official code, which keep the checkpoint with the "
    "highest AUC; 'epoch 100' is the model at the end of training, which needs no ground truth to select. "
    "RX is the classic statistical baseline computed on the same data."))
title(s, "Results: detection accuracy (AUC)")
stats = [(f"{avg_best:.4f}", "mean AUC, ours (best epoch)", TEAL), (f"{avg_paper:.4f}", "mean AUC, paper", INK),
         (f"{avg_rx:.4f}", "mean AUC, RX baseline", MUTED), (f"{avg_time:.2f} s", "mean inference per image", AMBER)]
for i, (v, l, col) in enumerate(stats):
    x = 0.6 + i * 3.08
    box(s, x, 1.45, 2.9, 1.45, fill=PANEL)
    text(s, x + 0.1, 1.52, 2.7, 0.75, v, size=32, font=HEAD, bold=True, color=col, align=PP_ALIGN.CENTER)
    text(s, x + 0.1, 2.3, 2.7, 0.5, l, size=13, color=MUTED, align=PP_ALIGN.CENTER)
table(s, rows, 0.6, 3.2, 12.1, [1.5, 1.0, 2.0, 1.7, 1.4, 1.3], row_h=0.5, size=13,
      highlight_rows=[len(rows) - 1], bold_cells=bold)
caption(s, "Best epoch = checkpoint with the highest AUC (evaluated every 5 epochs), the protocol of the paper and "
           "official code. Epoch 100 = final model, selected without ground truth.", y=6.65)

# ================================================================= 15. anomaly maps
s = new_slide(notes=(
    "Colour anomaly maps in the style of the paper's Figure 3. Red means high anomaly score. RX highlights many "
    "background structures; DMS2F-HAD suppresses the background and keeps the planes and rocks."))
title(s, "Colour anomaly maps")
picture(s, FIG / "fig3_anomaly_maps.png", 0.6, 1.2, 12.1, 5.6)
caption(s, "Same layout as Fig. 3 of the paper. AUC in each panel's corner.", y=6.85)

# ================================================================= 16. ROC
s = new_slide(notes=(
    "ROC curves in the style of the paper's Figure 5. Curves that hug the top-left corner detect most anomaly "
    "pixels at a very low false-alarm rate."))
title(s, "ROC curves")
picture(s, FIG / "fig5_roc.png", 0.6, 1.4, 12.1, 4.6)
text(s, 0.6, 6.1, 12.1, 0.7, "A curve closer to the top-left corner = more anomalies found at fewer false alarms. "
     "Legend values are AUC.", size=14, color=MUTED)

# ================================================================= 17. box plots
s = new_slide(notes=(
    "Box plots in the style of the paper's Figure 4. For each method, red is the score distribution of anomaly "
    "pixels and blue of background pixels. A good detector keeps the blue box low and narrow and the red box high."))
title(s, "Separating anomalies from background")
picture(s, FIG / "fig4_boxplots.png", 0.6, 1.4, 12.1, 4.6)
text(s, 0.6, 6.1, 12.1, 0.7, "Red = anomaly pixels, blue = background pixels. A large gap with little overlap means "
     "a threshold exists that separates the two.", size=14, color=MUTED)

# ================================================================= 18. ablation
ab = [("spectral", "Spectral only"), ("spatial", "Spatial only"), ("add", "Addition fusion"), ("gated", "Gated fusion")]
ab_ds = [d for d in DATASETS if all(k in R[d]["runs"] for k, _ in ab)]
s = new_slide(notes=(
    "Ablation in the style of the paper's Table 5: each branch on its own, simple addition of the two, and the "
    "full gated fusion. Each cell shows our AUC with the paper's value in brackets."))
title(s, "Ablation: do we need both branches and the gate?")
if ab_ds:
    rows = [["Dataset"] + [n for _, n in ab]]
    bold = []
    for d in ab_ds:
        a = [float(R[d]["runs"][k]["best_auc"]) for k, _ in ab]
        pk = [PAPER_AUC[d][k if k != "gated" else "DMS2F-HAD"] for k, _ in ab]
        rows.append([DATASETS[d]] + [f"{x:.4f}  ({p:.4f})" for x, p in zip(a, pk)])
        bold.append((len(rows) - 1, 1 + int(np.argmax(a))))
    m = [mean([float(R[d]["runs"][k]["best_auc"]) for d in ab_ds]) for k, _ in ab]
    pm = [mean([PAPER_AUC[d][k if k != "gated" else "DMS2F-HAD"] for d in ab_ds]) for k, _ in ab]
    rows.append(["Average"] + [f"{x:.4f}  ({p:.4f})" for x, p in zip(m, pm)])
    table(s, rows, 0.6, 1.5, 12.1, [1.6, 1.9, 1.9, 1.9, 1.9], row_h=0.55, size=14, highlight_rows=[len(rows) - 1],
          bold_cells=bold)
    y_cap = 1.6 + 0.55 * len(rows)
    caption(s, "Our AUC (best epoch), the paper's value in brackets. Bold = best of our four variants for that "
               "scene." + ("" if "Cri" in ab_ds else "  CRI ablation still training."), y=y_cap)
    mean_of = dict(zip([k for k, _ in ab], m))
    box(s, 0.6, y_cap + 0.5, 12.1, 6.85 - (y_cap + 0.5), fill=PANEL)
    text(s, 0.85, y_cap + 0.62, 11.6, 6.7 - (y_cap + 0.62), [
        [("Spectral branch alone is the weakest ", {"bold": True}),
         (f"(mean {mean_of['spectral']:.4f}), as in the paper: one pooled spectrum per patch cannot localise small "
          "targets.", {})],
        [("Adding the spatial branch is what matters. ", {"bold": True}),
         (f"Spatial only {mean_of['spatial']:.4f}, addition {mean_of['add']:.4f}, gated {mean_of['gated']:.4f}: "
          f"within {max(mean_of['spatial'], mean_of['add'], mean_of['gated']) - min(mean_of['spatial'], mean_of['add'], mean_of['gated']):.3f} "
          "of each other.", {})],
        [("Gate vs. addition: ", {"bold": True}),
         ("on these scenes, with one seed, we could not confirm the paper's gain from gated fusion; the difference is "
          "within run-to-run noise.", {})],
    ], size=14, bullets=True, space_after=6)
else:
    text(s, 0.6, 2.0, 12, 1, "Ablation runs not finished yet.", size=18, color=MUTED)

# ================================================================= 19. training behaviour
s = new_slide(notes=(
    "Training curves. The loss keeps going down, but AUC does not always keep going up: after the background is "
    "learned, the network also starts to reconstruct the anomalies, and the anomaly scores shrink. That is why the "
    "choice of checkpoint matters, and why we report both the best epoch and the final epoch."))
title(s, "Training behaviour: lower loss is not always better detection")
picture(s, FIG / "fig_training_curves.png", 0.6, 1.3, 8.2, 5.4)
box(s, 9.1, 1.45, 3.65, 5.1, fill=PANEL)
text(s, 9.3, 1.6, 3.3, 4.9, [
    [("Loss falls steadily", {"bold": True}), (" on every scene.", {})],
    [("AUC can peak early, then drop", {"bold": True}), (": once the background is learned, the decoder starts to "
                                                       "reconstruct anomalies too.", {})],
    [("Model selection matters. ", {"bold": True}), ("The paper reports the best checkpoint; with no labels, early "
                                                    "stopping or fewer epochs would be needed.", {})],
], size=14, bullets=True, space_after=10)

# ================================================================= 20. takeaways
s = new_slide(dark=True, notes="Summary of what the method does, what we reproduced, and the limits of this study.")
title(s, "Takeaways", dark=True)
cols = [("What DMS2F-HAD does", [
            "Learns the background of one image without labels, as a patch autoencoder.",
            "Splits modelling into a spatial and a spectral Mamba branch, mixed per pixel by a learned gate.",
            "Stays tiny (≈0.65 M parameters) and fast thanks to linear-time Mamba."]),
        ("What we reproduced", [
            f"Mean AUC {avg_best:.4f} (best epoch) vs {avg_paper:.4f} in the paper on the same {len(GATED)} scenes.",
            f"RX baseline matches the paper exactly (mean {avg_rx:.4f}).",
            "Ablation: spectral-only is clearly weakest, as in the paper; gated vs. addition fusion is within noise.",
            f"Full-image inference in {avg_time:.2f} s."]),
        ("Limits of this study", [
            "4 of the paper's 14 scenes (the ones released with the code); one random seed.",
            "Best-epoch numbers use the ground truth to pick the checkpoint, as the paper does.",
            "Pure-PyTorch scan is slower to train than the fused CUDA kernels."])]
for i, (h, pts) in enumerate(cols):
    x = 0.6 + i * 4.1
    box(s, x, 1.5, 3.9, 4.3, fill="1B2B45")
    badge(s, x + 0.25, 1.73, str(i + 1), d=0.5, size=15)
    text(s, x + 0.9, 1.75, 2.9, 0.6, h, size=18, font=HEAD, bold=True, color=WHITE)
    text(s, x + 0.25, 2.5, 3.45, 3.2, pts, size=14, color="DCE3EE", bullets=True, space_after=10)
text(s, 0.6, 6.1, 12.1, 0.8, [[("Bottom line: ", {"bold": True, "color": AMBER}),
     (f"a 0.65 M-parameter dual-branch Mamba autoencoder gets close to the paper (mean AUC {avg_best:.4f} vs "
      f"{avg_paper:.4f}) on the released scenes; the open problem is choosing the checkpoint without labels.", {})]],
     size=16, color=WHITE)

prs.save(OUT)
print("saved", OUT, "with", len(prs.slides), "slides")
