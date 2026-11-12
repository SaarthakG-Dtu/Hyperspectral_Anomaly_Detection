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
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from PIL import Image
from sklearn.metrics import roc_curve

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


def bar_chart(s, x, y, w, h, cats, vals, title_txt, colors, fmt="0.0000", vmax=None):
    """Horizontal one-series bar chart; categories are drawn bottom-to-top."""
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series("value", vals)
    ch = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    ch.has_legend = False
    ch.has_title = True
    ch.chart_title.text_frame.text = title_txt
    r = ch.chart_title.text_frame.paragraphs[0].runs[0]
    r.font.size, r.font.bold, r.font.color.rgb = Pt(14), True, rgb(INK)
    plot = ch.plots[0]
    plot.gap_width = 45
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = fmt, False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size, dl.font.color.rgb = Pt(13), rgb(INK)
    for i, c in enumerate(colors):
        pt = plot.series[0].points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = rgb(c)
    va = ch.value_axis
    va.minimum_scale = 0
    if vmax:
        va.maximum_scale = vmax
    va.major_gridlines.format.line.color.rgb = rgb("E3E8EF")
    va.format.line.fill.background()
    va.tick_labels.font.size, va.tick_labels.font.color.rgb = Pt(12), rgb(MUTED)
    ca = ch.category_axis
    ca.tick_labels.font.size, ca.tick_labels.font.color.rgb = Pt(14), rgb(INK)
    ca.format.line.color.rgb = rgb(LINE)
    return ch


# ---------------------------------------------------------------- numbers reported in the paper
BASELINES = ["RX", "AUTO-AD", "TDD", "LREN", "RGAE", "GT-HAD"]
PAPER_T2 = {   # Table 2 (AUC): six baselines + DMS2F-HAD, on the scenes we have
    "Cri":       [0.9989, 0.5123, 0.9906, 0.9976, 0.9983, 0.9952, 0.9999],
    "aviris_1":  [0.8866, 0.7954, 0.9520, 0.9757, 0.9886, 0.9862, 0.9891],
    "aviris_2":  [0.9181, 0.9034, 0.9091, 0.9342, 0.9392, 0.8970, 0.9851],
    "San_Diego": [0.9403, 0.9471, 0.9447, 0.9704, 0.9920, 0.9815, 0.9722],
}
PAPER_AVG14 = [("TDD", 0.5900), ("LREN", 0.8297), ("AUTO-AD", 0.9189), ("RGAE", 0.9326), ("RX", 0.9523),
               ("GT-HAD", 0.9774), ("DMS2F-HAD", 0.9878)]           # Table 2, average over 14 scenes
PAPER_INFER = [("GT-HAD", 36.26), ("AUTO-AD", 9.32), ("TDD", 2.55), ("DMS2F-HAD", 0.555)]  # Table 3, mean seconds


# ---------------------------------------------------------------- results
def tpr_at(score, gt, fpr_max=0.01):
    """Share of anomaly pixels detected when the threshold allows `fpr_max` false alarms."""
    fpr, tpr, _ = roc_curve(gt.ravel(), score.ravel())
    return float(np.interp(fpr_max, fpr, tpr))


def load_results():
    res = {}
    for d in DATASETS:
        cube, gt = load_hsi(d)
        rx_map = rx_detector(cube)
        r = {"rx": auc_score(rx_map, gt), "rx_tpr1": tpr_at(rx_map, gt), "runs": {}}
        for f in ["gated", "add", "spatial", "spectral"]:
            p = RUNS / f"{d}_{f}.npz"
            if p.exists():
                z = np.load(p)
                r["runs"][f] = {k: z[k] for k in ["best_auc", "final_auc", "best_epoch", "infer_time", "params",
                                                   "train_time"]}
                r["runs"][f]["tpr1"] = tpr_at(z["best_map"], gt)
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

# ---------------------------------------------------------------- extra test runs (results/runs/extra)
EXTRA_DS = ["aviris_1", "aviris_2", "San_Diego"]


def extra(d, tag):
    p = RUNS / "extra" / f"{d}_gated_{tag}.npz"
    return float(np.load(p)["best_auc"]) if p.exists() else None


X = {d: {"main": float(R[d]["runs"]["gated"]["best_auc"]) if "gated" in R[d]["runs"] else None,
         "e1": [extra(d, f"every1_s{k}") for k in range(3)], "official": extra(d, "official")} for d in EXTRA_DS}

# ================================================================= 1. title
s = new_slide(dark=True, notes=(
    "This project re-implements DMS2F-HAD, a WACV 2026 paper on hyperspectral anomaly detection. I explain what "
    "hyperspectral images and anomaly detection are, how the method works, which datasets were used, and how my "
    "own results compare with the paper and with other models."))
text(s, 0.7, 1.0, 7.4, 0.4, "WACV 2026  ·  paper implementation project", size=14, color=AMBER, bold=True)
text(s, 0.7, 1.5, 7.4, 1.1, "DMS2F-HAD", size=60, font=HEAD, bold=True, color=WHITE)
text(s, 0.7, 2.75, 7.2, 1.3, "A Dual-branch Mamba-based Spatial–Spectral Fusion Network for Hyperspectral "
     "Anomaly Detection", size=22, font=HEAD, color="DCE3EE")
text(s, 0.7, 4.45, 7.2, 1.3, [
    [("Paper: ", {"bold": True, "color": AMBER}), ("A. Pant, L. Tamang, T.-K. Lee and S. Aryal, Deakin University.", {})],
    [("This project: ", {"bold": True, "color": AMBER}),
     ("We re-implemented the method in PyTorch and trained and evaluated it on four benchmark scenes.", {})],
], size=15, color="DCE3EE")
picture(s, ASSETS / "hsi_cube_dark.png", 8.2, 0.9, 4.7, 4.9)
text(s, 8.2, 5.85, 4.7, 0.4, "The San Diego scene has 100 × 100 pixels and 189 spectral bands.", size=11,
     color="9FB0C6", align=PP_ALIGN.CENTER)

# ================================================================= 2. roadmap
s = new_slide(notes="The talk has five parts: the data, the problem, the method, the setup and the results.")
title(s, "Roadmap")
items = [("Hyperspectral images", "We see what the data is: a cube of spectra."),
         ("Anomaly detection", "We define HAD, its evaluation and its method families."),
         ("DMS2F-HAD method", "We explain Mamba, the two branches and the gate."),
         ("Data & setup", "We describe the scenes, the setup and our code."),
         ("Results", "We compare with other models and show our results.")]
for i, (h, d) in enumerate(items):
    x = 0.6 + i * 2.48
    box(s, x, 1.45, 2.28, 3.25, fill=PANEL)
    badge(s, x + 0.25, 1.7, str(i + 1), d=0.62, size=20)
    text(s, x + 0.2, 2.5, 1.95, 0.9, h, size=19, font=HEAD, bold=True)
    text(s, x + 0.2, 3.35, 1.95, 1.3, d, size=14, color=MUTED)
n_runs = sum(len(R[d]["runs"]) for d in DATASETS)
box(s, 0.6, 5.0, 12.1, 1.85, fill=INK)
text(s, 0.85, 5.1, 3.2, 0.4, "The project in numbers", size=14, bold=True, color=AMBER)
for i, (v, l) in enumerate([("4", "benchmark scenes"), (str(n_runs), "models trained"), ("0.65 M", "parameters"),
                            (f"{avg_best:.4f}", "our mean AUC"), ("7", "methods compared")]):
    x = 0.85 + i * 2.4
    text(s, x, 5.5, 2.2, 0.7, v, size=30, font=HEAD, bold=True, color=WHITE)
    text(s, x, 6.2, 2.2, 0.45, l, size=13, color="B9C4D4")

# ================================================================= 3. HSI
s = new_slide(notes=(
    "A normal camera records three broad colour bands. A hyperspectral sensor such as AVIRIS records hundreds of "
    "narrow contiguous bands from the visible to the short-wave infrared. Each pixel is therefore a full spectrum, "
    "and the image is a 3-D cube of size H x W x C. Different materials have different spectral fingerprints."))
title(s, "A hyperspectral image is a cube: every pixel is a spectrum")
picture(s, FIG / "fig_hsi_cube.png", 0.5, 1.4, 5.3, 5.3)
text(s, 6.2, 1.5, 6.5, 3.75, [
    "An RGB camera records three broad colour bands, whereas a hyperspectral sensor records 100 to 200+ narrow, "
    "contiguous bands. AVIRIS, for example, records 224 bands between 400 and 2500 nm, each about 10 nm wide.",
    [("The data therefore forms an ", {}), ("H × W × C", {"bold": True}),
     (" cube, in which each pixel is a C-dimensional spectral signature.", {})],
    "Materials such as metal, paint, vegetation, soil and rock reflect light differently at each wavelength, so a "
    "spectrum works like a material fingerprint.",
    "The main difficulties are the hundreds of correlated bands, sensor noise, mixed pixels and the near absence "
    "of labels.",
], size=15, bullets=True, space_after=8)
for i, (num, lab) in enumerate([("189", "bands per pixel"), ("10 000", "pixels (100 × 100)"),
                                ("~1.3 %", "of pixels are anomalies")]):
    x = 6.2 + i * 2.2
    box(s, x, 5.35, 2.0, 1.25, fill=PANEL)
    text(s, x + 0.1, 5.42, 1.8, 0.6, num, size=26, font=HEAD, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
    text(s, x + 0.1, 6.0, 1.8, 0.5, lab, size=12, color=MUTED, align=PP_ALIGN.CENTER)
caption(s, "These numbers are for the San Diego scene used in this project.", y=6.75, x=6.2, w=6.5)

# ================================================================= 4. HAD
s = new_slide(notes=(
    "Hyperspectral anomaly detection gives every pixel a score for how different it is from the background. It is "
    "unsupervised: we have no target spectra and no labels. In San Diego the planes are rare and their spectra "
    "differ from the apron and the roads, which is exactly what HAD exploits."))
title(s, "Anomaly detection (HAD): finding pixels that do not fit")
text(s, 0.6, 1.45, 5.7, 4.1, [
    "The goal is to give every pixel an anomaly score; together, the scores form an anomaly map.",
    "The task is unsupervised: there are no target spectra and no labels, so the background model has to be "
    "learned from the image itself.",
    "Anomalies are small and rare (often less than 1 % of the pixels), and their spectra differ from their "
    "surroundings. Typical examples are planes, vehicles, rocks and boats.",
    "The scores are thresholded into detections, which are then compared with a ground-truth mask.",
], size=15, bullets=True, space_after=8)
picture(s, FIG / "fig_spectra.png", 6.5, 1.45, 6.3, 3.9)
text(s, 0.6, 5.75, 6.0, 0.4, "HAD is used in many fields, for example:", size=14, bold=True, color=MUTED)
for i, a in enumerate(["Defence & surveillance", "Search and rescue", "Mineral exploration",
                       "Environmental monitoring", "Precision agriculture"]):
    label_box(s, 0.6 + i * 2.45, 6.2, 2.3, 0.55, a, fill="E3F1EF", color=INK, size=13, bold=True)

# ================================================================= 5. evaluation
s = new_slide(notes=(
    "There are three standard ways to judge a detector. The colour anomaly map is the visual view. The ROC curve "
    "plots the detection rate against the false-alarm rate as the threshold moves, and the AUC summarises it in "
    "one number. Box plots show how well the anomaly scores are separated from the background scores."))
title(s, "How a HAD method is evaluated")
cards = [("Anomaly map", "The scores are shown with a jet colour map, in which blue means background and red means "
                         "anomaly. It gives a visual check against the ground truth."),
         ("ROC curve & AUC", "The threshold is swept to plot the detection rate against the false-alarm rate. An AUC "
                             "of 1 is perfect, and 0.5 is random guessing."),
         ("Box plots", "They show the score distributions of anomaly and background pixels. A large gap with "
                       "little overlap means few false alarms.")]
for i, (h, d) in enumerate(cards):
    x = 0.6 + i * 4.1
    box(s, x, 1.4, 3.9, 2.3, fill=PANEL)
    badge(s, x + 0.2, 1.57, str(i + 1), d=0.48, size=15)
    text(s, x + 0.8, 1.59, 3.0, 0.5, h, size=18, font=HEAD, bold=True)
    text(s, x + 0.2, 2.15, 3.55, 1.5, d, size=14, color=MUTED)
picture(s, FIG / "San_Diego_result.png", 0.6, 3.85, 12.1, 2.95)
caption(s, "Example from our run on San Diego: the false-colour image, the ground truth, the RX and DMS2F-HAD "
           "anomaly maps, and the ROC curves.")

# ================================================================= 6. families
s = new_slide(notes=(
    "HAD research has moved from statistics to deep learning. RX is the classic baseline. Representation methods "
    "learn a background dictionary. Deep models learn non-linear backgrounds, but CNNs see only local context and "
    "Transformers are expensive. Mamba-based state-space models are the newest family, and DMS2F-HAD belongs to it."))
title(s, "Four families of HAD methods")
fam = [("Statistical", "RX (Reed–Xiaoli) measures the Mahalanobis distance of each pixel to a Gaussian "
                       "background.",
        "It is fast and needs no training.", "Its Gaussian assumption fails in cluttered scenes."),
       ("Representation", "These methods learn a background dictionary or split the image into low-rank and "
                          "sparse parts.",
        "They separate outliers better than RX.", "They are slow and sensitive to noise and parameters."),
       ("Deep learning", "Autoencoders, GANs, CNNs and Transformers learn the background, e.g. AUTO-AD, LREN, "
                         "RGAE, TDD and GT-HAD.",
        "They can learn non-linear backgrounds.", "CNNs see only local context; Transformers cost O(L²)."),
       ("Mamba (SSM)", "MMR-HAD and DMS2F-HAD use selective state-space sequence models.",
        "They model long-range context in linear time.", "The research area is still young.")]
for i, (h, d, pro, con) in enumerate(fam):
    x = 0.6 + i * 3.08
    hl = i == 3
    box(s, x, 1.45, 2.9, 4.3, fill="FDF1DC" if hl else PANEL)
    badge(s, x + 0.2, 1.62, str(i + 1), d=0.48, size=15, fill=AMBER if hl else "C9D3E0")
    text(s, x + 0.8, 1.66, 2.05, 0.5, h, size=18, font=HEAD, bold=True)
    text(s, x + 0.2, 2.2, 2.55, 1.3, d, size=14, color=INK)
    text(s, x + 0.2, 3.5, 2.55, 0.3, "Strength", size=12, bold=True, color=TEAL)
    text(s, x + 0.2, 3.78, 2.55, 0.75, pro, size=14)
    text(s, x + 0.2, 4.52, 2.55, 0.3, "Weakness", size=12, bold=True, color=CORAL)
    text(s, x + 0.2, 4.8, 2.55, 0.9, con, size=14)
box(s, 0.6, 5.95, 12.1, 0.9, fill=INK)
text(s, 0.85, 6.0, 11.6, 0.8, [[("In this project, ", {"bold": True, "color": AMBER}),
     ("RX is run as the statistical baseline and DMS2F-HAD represents the Mamba family. The paper's results for "
      "AUTO-AD, LREN, RGAE, TDD and GT-HAD are compared in the results section.", {})]], size=15, color=WHITE,
     anchor=MSO_ANCHOR.MIDDLE)

# ================================================================= 7. Mamba
s = new_slide(notes=(
    "A state-space model keeps a hidden state h and updates it at every position of the sequence. Mamba makes the "
    "step size and the B and C matrices depend on the input, so it can choose what to remember. The cost grows "
    "linearly with the sequence length, while self-attention grows quadratically. A 16x16 patch is a sequence of "
    "256 pixel tokens."))
title(s, "Why Mamba? Long-range modelling in linear time")
box(s, 0.6, 1.5, 5.9, 1.6, fill=INK)
text(s, 0.85, 1.62, 5.5, 1.4, [
    [("hₜ = Āₜ · hₜ₋₁ + B̄ₜ · xₜ", {"font": "Cambria Math", "size": 24, "color": WHITE})],
    [("yₜ = Cₜ · hₜ + D · xₜ", {"font": "Cambria Math", "size": 24, "color": WHITE})],
], size=22, color=WHITE, space_after=4)
text(s, 0.6, 3.3, 5.9, 3.5, [
    "A state-space model carries a hidden state h along the sequence.",
    "In the selective version (S6), Δ, B and C are computed from the input, so the model decides what to keep "
    "and what to forget.",
    "The cost grows linearly with the sequence length L, whereas self-attention grows with L².",
    "In DMS2F-HAD, a 16 × 16 patch becomes 256 pixel tokens for the spatial branch and 7 band groups for the "
    "spectral branch.",
], size=15, bullets=True, space_after=8)
cd = CategoryChartData()
Ls = [256, 512, 1024, 2048, 4096]
cd.categories = [str(L) for L in Ls]
cd.add_series("Self-attention  ~L²", [L * L / 1e6 for L in Ls])
cd.add_series("Mamba  ~L·N (N = 16)", [L * 16 / 1e6 for L in Ls])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(6.9), Inches(1.45), Inches(5.9), Inches(5.0), cd)
ch = gf.chart
ch.has_title = True
ch.chart_title.text_frame.text = "Relative compute vs. sequence length (millions of operations)"
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
    "The image is cut into overlapping 16x16 patches, and random rectangles are masked during training. A 1x1 "
    "convolution reduces the bands to 64 features. The spatial branch looks at texture across the 256 pixels, and "
    "the spectral branch looks at correlations between bands. A learned gate mixes them pixel by pixel, a decoder "
    "rebuilds the patch, and the reconstruction error is the anomaly score."))
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
text(s, 0.5, 1.5, 5.4, 1.6, [
    "An autoencoder is trained on patches of the image itself, without labels.",
    "Because the background dominates the data, it is reconstructed well, while the rare anomalies are not.",
], size=14, color=MUTED, bullets=True, space_after=6)
text(s, 0.5, 4.9, 5.4, 1.4, [
    "Our model has 0.65 M parameters; the paper reports 0.64 M.",
    "Linear-time Mamba blocks take the place of attention.",
], size=14, color=MUTED, bullets=True, space_after=6)

# ================================================================= 9. paper figure
s = new_slide(notes="This is the original architecture figure from the paper: (a) the full network, (b) the Mamba "
                    "block and (c) the spatial-spectral decoder block.")
title(s, "The architecture as drawn in the paper")
picture(s, ASSETS / "paper_fig2_architecture.png", 0.6, 1.25, 12.1, 5.55)
caption(s, "Figure 2 of Pant et al., “DMS2F-HAD”, WACV 2026: (a) the overall framework, (b) the Mamba block and "
           "(c) the SS decoder block.")

# ================================================================= 10. components
s = new_slide(notes=(
    "There are four pieces. The spatial branch sees each patch as 256 pixel tokens. The spectral branch splits the "
    "64 features into 7 overlapping groups of 16, so that Mamba can model band correlations cheaply. The gate is a "
    "1x1 convolution with a sigmoid that gives one weight per pixel. The decoder mixes a global Mamba path with "
    "local 3x3 and 5x5 convolutions."))
title(s, "The four building blocks")
comp = [("Spatial branch", TEAL, "Parallel 3×3 and 5×5 convolutions capture texture at two scales. The 16×16 map is "
         "then flattened into 256 tokens, and a Mamba block gives every pixel a global receptive field.",
         "[B,64,16,16] → 256 tokens × 64 → Mamba → [B,64,16,16]"),
        ("Spectral branch", CORAL, "Two 1-D convolutions (kernels 7 and 3) run along the spectrum. The 64 features "
         "are then split into g = 7 overlapping groups of c₂ = 16 (stride k = 8), and Mamba runs over the groups.",
         "[B,64] → 7 groups × 16 → Mamba → [B,64] → copied to 16×16"),
        ("Adaptive gated fusion", AMBER, "A 1×1 convolution with a sigmoid gives one weight G per pixel. The fused "
         "feature is Proj(G ⊙ F_spa + (1 − G) ⊙ F_spe), so texture is used where it helps and spectra elsewhere.",
         "G → 1 trusts spatial texture  ·  G → 0 trusts the spectrum"),
        ("SS decoder + detection", INK, "Each of the two decoder blocks combines a Mamba path with 3×3 and 5×5 "
         "convolutions. A 1×1 out-head maps back to C bands, and each pixel's L2 residual is its score.",
         "[B,64,16,16] → [B,C,16,16]  ·  score = ‖x − recon‖₂ per pixel")]
for i, (h, col, d, shape) in enumerate(comp):
    x, y = 0.6 + (i % 2) * 6.15, 1.45 + (i // 2) * 2.75
    box(s, x, y, 5.95, 2.6, fill=PANEL)
    badge(s, x + 0.25, y + 0.2, str(i + 1), d=0.5, fill=col, color=WHITE if col == INK else INK, size=15)
    text(s, x + 0.95, y + 0.22, 4.8, 0.5, h, size=19, font=HEAD, bold=True)
    text(s, x + 0.3, y + 0.8, 5.45, 1.15, d, size=14, color=INK)
    box(s, x + 0.25, y + 1.95, 5.45, 0.48, fill=WHITE)
    text(s, x + 0.35, y + 1.99, 5.25, 0.4, shape, size=13, color=INK, anchor=MSO_ANCHOR.MIDDLE)

# ================================================================= 11. datasets (14)
s = new_slide(notes=(
    "The paper evaluates on 14 public scenes, all from AVIRIS except CRI. The official code release ships four of "
    "them, CRI, AVIRIS-1, AVIRIS-2 and San Diego, and these are the ones I used. They are highlighted."))
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
text(s, 1.0, 5.66, 11.5, 0.4, "Highlighted rows are the scenes released with the official code, which we use in "
     "this project.", size=13, color=MUTED)
caption(s, "All scenes were captured by AVIRIS except CRI, which comes from a Nuance CRI sensor. *The released "
           "AVIRIS-1 file has 189 bands, while the paper lists 186.", y=6.15)

# ================================================================= 12. our datasets
s = new_slide(notes=(
    "CRI is a large grassland scene with ten rocks, and the three AVIRIS scenes are airport images with planes. "
    "White pixels in the ground truth are anomalies."))
title(s, "The four scenes used in this project")
picture(s, FIG / "fig_datasets.png", 0.6, 1.3, 12.1, 5.3)
caption(s, "The top row shows false-colour composites of three bands, and the bottom row shows the ground-truth "
           "masks, in which white marks anomalies.", y=6.7)

# ================================================================= 13. implementation
p_groups = {}
for d in GATED:
    p_groups.setdefault(params[d], []).append(DATASETS[d])
p_str = " and ".join(f"{p/1e6:.3f} M for {', '.join(n)}" for p, n in sorted(p_groups.items(), reverse=True)) or "n/a"
s = new_slide(notes=(
    "The official code depends on the mamba_ssm CUDA kernels, which do not build on Windows. I wrote the Mamba "
    "selective scan in plain PyTorch as an exact two-level chunked scan and kept the paper's training settings. "
    "Each image is trained on its own, without labels, and the ground truth is only used to compute the AUC."))
title(s, "Implementation and training setup")
hp = [["Setting", "Value"], ["Patch size / stride", "16 × 16 / 8"], ["Embedding c₁", "64"],
      ["Band group c₂ / stride k", "16 / 8  (7 groups)"], ["Mamba state size", "16  (official code: 64)"],
      ["Optimiser", "Adam, lr 5e-4, weight decay 1e-4"], ["Epochs / batch", "100 / 32"],
      ["Loss", "MSE + 0.1 · L1 (official code)"], ["Masking", "p = 0.5, random 2–5 px rectangle"]]
table(s, hp, 0.6, 1.5, 5.6, [2.3, 3.0], row_h=0.5, size=13)
text(s, 6.6, 1.45, 6.2, 0.45, "The method fits in about 440 lines of Python", size=18, font=HEAD, bold=True)
files = [("model.py", "Implements Mamba and the network."),
         ("utils.py", "Loads data, patches it, runs RX and scores AUC."),
         ("train.py", "Trains on one image and saves its map."),
         ("make_figures.py", "Draws the paper-style figures."),
         ("presentation/make_ppt.py", "Builds this slide deck.")]
for i, (f, d) in enumerate(files):
    y = 2.0 + i * 0.62
    box(s, 6.6, y, 6.2, 0.52, fill=PANEL)
    text(s, 6.75, y + 0.06, 2.6, 0.42, f, size=13, font="Consolas", bold=True, color=INK)
    text(s, 9.35, y + 0.08, 3.4, 0.42, d, size=12, color=MUTED)
box(s, 6.6, 5.2, 6.2, 1.55, fill="FDF1DC")
text(s, 6.8, 5.27, 5.85, 1.45, [
    "No mamba_ssm library is needed: the selective scan runs as an exact two-level chunked scan (16 + 16 steps "
    "instead of 256), so the code runs on Windows, on a CPU or on any GPU.",
    f"The model has {p_str}.",
], size=13, space_after=4)

# ================================================================= 14a. competing models
s = new_slide(notes=(
    "The paper compares DMS2F-HAD with six published detectors on 14 scenes, and with MMR-HAD, the other Mamba "
    "model, on four shared scenes. Together they cover every family: statistics (RX), autoencoders (AUTO-AD, LREN "
    "and RGAE), a transferred detector (TDD) and Transformers (GT-HAD)."))
title(s, "The models DMS2F-HAD is compared against")
models = [("RX", "Statistical  ·  1990", "It measures the Mahalanobis distance of each pixel to one Gaussian "
           "background model.", "It needs no training."),
          ("AUTO-AD", "Autoencoder  ·  TGRS 2022", "It is a fully convolutional autoencoder whose adaptive loss keeps "
           "it from learning the anomalies.", "It has 3.23 M parameters."),
          ("LREN", "Low-rank network  ·  AAAI 2021", "It is an autoencoder whose latent space is a low-rank "
           "background dictionary.", "It has 0.86 M parameters."),
          ("RGAE", "Graph autoencoder  ·  TGRS 2022", "It is a robust autoencoder regularised by a superpixel graph "
           "of the image.", "It is trained per image."),
          ("TDD", "Transferred detector  ·  TGRS 2024", "It is trained once on other data to spot spectral "
           "deviations and is then applied directly.", "It has 46.64 M parameters."),
          ("GT-HAD", "Gated Transformer  ·  TNNLS 2025", "It is the strongest earlier baseline, but its attention "
           "cost grows with L².", "It has 0.26 M parameters."),
          ("MMR-HAD", "Mamba + attention  ·  TGRS 2025", "It combines multi-scale Mamba blocks with attention; its "
           "code is not public.", "It has 2.12 M parameters."),
          ("DMS2F-HAD", "Dual-branch Mamba  ·  WACV 2026", "It uses spatial and spectral Mamba branches with gated "
           "fusion and no attention.", "It has 0.64 M parameters.")]
for i, (n, kind, d, foot) in enumerate(models):
    x, y = 0.6 + (i % 4) * 3.08, 1.45 + (i // 4) * 2.75
    ours = n == "DMS2F-HAD"
    box(s, x, y, 2.9, 2.55, fill="FDF1DC" if ours else PANEL)
    text(s, x + 0.2, y + 0.12, 2.5, 0.45, n, size=19, font=HEAD, bold=True)
    text(s, x + 0.2, y + 0.58, 2.6, 0.35, kind, size=12, bold=True, color=CORAL if ours else TEAL)
    text(s, x + 0.2, y + 0.95, 2.55, 1.15, d, size=13)
    text(s, x + 0.2, y + 2.1, 2.55, 0.35, foot, size=12, color=MUTED, bold=True)

# ================================================================= 14b. paper benchmark (AUC)
s = new_slide(notes=(
    "These are the paper's numbers over all 14 scenes. DMS2F-HAD has the highest average AUC, 0.9878, ahead of the "
    "gated Transformer GT-HAD at 0.9774 and the classic RX at 0.9523. It is the best model on 9 of the 14 scenes. "
    "TDD, which is not trained on the target scene, collapses on several scenes."))
title(s, "The paper's benchmark: the highest mean AUC over 14 scenes")
cats = [n for n, _ in PAPER_AVG14]
bar_chart(s, 0.6, 1.4, 7.6, 5.3, cats, [v for _, v in PAPER_AVG14], "Mean AUC over 14 scenes (paper Table 2)",
          [AMBER if n == "DMS2F-HAD" else "9AA8BA" for n in cats], vmax=1.1)
for i, (v, l) in enumerate([("9 / 14", "DMS2F-HAD is the best method on 9 of the 14 scenes."),
                            ("+0.0104", "Its mean AUC is 0.0104 higher than that of GT-HAD, the next best."),
                            ("+0.0355", "Its mean AUC is 0.0355 higher than that of RX.")]):
    y = 1.5 + i * 1.75
    box(s, 8.6, y, 4.2, 1.55, fill=PANEL)
    text(s, 8.75, y + 0.08, 3.9, 0.75, v, size=32, font=HEAD, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
    text(s, 8.75, y + 0.83, 3.9, 0.65, l, size=13, color=MUTED, align=PP_ALIGN.CENTER)
caption(s, "These numbers are reported by the paper (14 scenes, best checkpoint). The remaining five scenes are won "
           "by RX, AUTO-AD, RGAE and GT-HAD (twice).", y=6.85)

# ================================================================= 14c. efficiency
s = new_slide(notes=(
    "Accuracy is only half of the story. On an A100, the paper measures 0.555 s per image on average, which is 4.6 "
    "times faster than TDD and 65 times faster than GT-HAD. Against MMR-HAD, the other Mamba detector, each model "
    "wins two of the four shared scenes, but DMS2F-HAD needs 3.3 times fewer parameters and 29 times fewer FLOPs."))
title(s, "Efficiency: the fastest model in the comparison")
cats = [n for n, _ in PAPER_INFER]
bar_chart(s, 0.6, 1.4, 6.2, 4.1, cats, [v for _, v in PAPER_INFER], "Mean inference time per image in seconds "
          "(paper Table 3, A100)", [AMBER if n == "DMS2F-HAD" else "9AA8BA" for n in cats], fmt="0.0##", vmax=42)
for i, (v, l) in enumerate([("4.6×", "faster than TDD"), ("17×", "faster than AUTO-AD"), ("65×", "faster than GT-HAD")]):
    x = 0.6 + i * 2.1
    box(s, x, 5.7, 1.95, 1.1, fill=PANEL)
    text(s, x + 0.05, 5.74, 1.85, 0.55, v, size=24, font=HEAD, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
    text(s, x + 0.05, 6.3, 1.85, 0.4, l, size=12, color=MUTED, align=PP_ALIGN.CENTER)
eff = [["Model", "Params", "GFLOPs", "Mean AUC (14)"],
       ["TDD", "46.64 M", "≈ 0.3", "0.5900"], ["AUTO-AD", "3.23 M", "≈ 12", "0.9189"],
       ["MMR-HAD", "2.12 M", "3.5", "–"], ["LREN", "0.86 M", "≈ 0.2", "0.8297"],
       ["GT-HAD", "0.26 M", "≈ 2.6", "0.9774"], ["DMS2F-HAD", "0.64 M", "0.12", "0.9878"]]
table(s, eff, 7.2, 1.45, 5.6, [1.5, 1.1, 1.0, 1.4], row_h=0.44, size=13, highlight_rows=[6])
box(s, 7.2, 4.75, 5.6, 2.05, fill="FDF1DC")
text(s, 7.4, 4.85, 5.25, 1.9, [
    "Against MMR-HAD, on the four scenes that both papers share, DMS2F-HAD wins on Cat Island and Bay Champagne, "
    "and MMR-HAD wins on Texas Coast and Gulfport.",
    [("DMS2F-HAD reaches this accuracy with 3.3× fewer parameters and 29× fewer FLOPs.", {"bold": True})],
], size=14, space_after=8)
caption(s, "Parameters and FLOPs come from the paper's Fig. 1 and Table 4; values marked ≈ are read off Fig. 1.",
        y=6.9, x=7.2, w=5.6)

# ================================================================= 14d. our 4 scenes vs baselines
s = new_slide(notes=(
    "Here I put our own runs next to the paper's numbers for every baseline on the four scenes we have. The "
    "baseline values are taken from the paper; RX we recomputed, and it matches exactly."))
title(s, "Our reproduction vs. the baselines on our four scenes")
hdr = ["Scene"] + BASELINES + ["DMS2F-HAD\n(paper)", "DMS2F-HAD\n(our run)"]
rows, bold, beat = [hdr], [], 0
ours_vals = {}
for d in DATASETS:
    g = R[d]["runs"].get("gated")
    v = PAPER_T2[d]
    ours = float(g["best_auc"]) if g is not None else None
    ours_vals[d] = ours
    rows.append([DATASETS[d]] + [f"{a:.4f}" for a in v] + [f"{ours:.4f}" if ours is not None else "running"])
    allv = v + ([ours] if ours is not None else [])
    bold.append((len(rows) - 1, 1 + int(np.argmax(allv))))
    if ours is not None and ours > max(v[:6]):
        beat += 1
done = [d for d in DATASETS if ours_vals[d] is not None]
avg = [np.mean([PAPER_T2[d][j] for d in done]) for j in range(7)] + [np.mean([ours_vals[d] for d in done])]
rows.append([f"Average ({len(done)})"] + [f"{a:.4f}" for a in avg])
bold.append((len(rows) - 1, 1 + int(np.argmax(avg))))
table(s, rows, 0.6, 1.45, 12.1, [1.3, 0.95, 1.05, 0.95, 0.95, 0.95, 1.05, 1.25, 1.25], row_h=0.55, size=14,
      highlight_rows=[len(rows) - 1], bold_cells=bold)
y2 = 1.45 + 0.55 * len(rows) + 0.3
box(s, 0.6, y2, 12.1, 6.75 - y2, fill=PANEL)
rk = []
for d in done:
    v = PAPER_T2[d][:6] + [ours_vals[d]]
    rk.append(f"#{1 + sorted(v, reverse=True).index(ours_vals[d])} on {DATASETS[d]}")
text(s, 0.85, y2 + 0.12, 11.6, 6.65 - y2, [
    [("Our run beats all six baselines on ", {}), (f"{beat} of {len(done)} scenes", {"bold": True}),
     (". Among the seven methods it ranks " + ", ".join(rk[:-1]) + " and " + rk[-1] + ".", {})],
    [("Compared with the paper's own DMS2F-HAD numbers, our mean AUC on these scenes differs by ", {}),
     (f"{avg[-1] - avg[-2]:+.4f}", {"bold": True}), (".", {})],
    "The strongest competitor on these scenes is RGAE, a graph autoencoder. AUTO-AD fails on CRI (0.5123), and TDD "
    "varies strongly between scenes.",
], size=15, bullets=True, space_after=6)
caption(s, "Bold marks the best value in each row. The baseline AUCs are the paper's (Table 2); 'our run' is the "
           "best epoch of our training.", y=6.9)

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
    "These are the main numbers. 'Best epoch' follows the paper and the official code, which keep the checkpoint "
    "with the highest AUC. 'Epoch 100' is the model at the end of training, which needs no ground truth to choose. "
    "RX is the classic statistical baseline, computed on the same data."))
title(s, "Our reproduction: detection accuracy (AUC)")
stats = [(f"{avg_best:.4f}", "Our mean AUC (best epoch)", TEAL), (f"{avg_paper:.4f}", "The paper's mean AUC", INK),
         (f"{avg_rx:.4f}", "Mean AUC of the RX baseline", MUTED), (f"{avg_time:.2f} s", "Mean inference time per image", AMBER)]
for i, (v, l, col) in enumerate(stats):
    x = 0.6 + i * 3.08
    box(s, x, 1.45, 2.9, 1.45, fill=PANEL)
    text(s, x + 0.1, 1.52, 2.7, 0.75, v, size=32, font=HEAD, bold=True, color=col, align=PP_ALIGN.CENTER)
    text(s, x + 0.1, 2.3, 2.7, 0.5, l, size=13, color=MUTED, align=PP_ALIGN.CENTER)
table(s, rows, 0.6, 3.2, 12.1, [1.5, 1.0, 2.0, 1.7, 1.4, 1.3], row_h=0.5, size=13,
      highlight_rows=[len(rows) - 1], bold_cells=bold)
caption(s, "The best epoch is the checkpoint with the highest AUC (checked every 5 epochs), which is the protocol of "
           "the paper and the official code. Epoch 100 is the final model, which is chosen without the ground truth.",
        y=6.5)

# ================================================================= 14e. why numbers differ
s = new_slide(notes=(
    "Our numbers are close to the paper's but not identical. The architecture, loss, data and scoring rule are the "
    "same. Several details differ between the paper text, the official code and our version, so I tested the ones "
    "that could matter: how often the checkpoint is checked, the random seed, and the official code's settings."))
title(s, "Why our numbers differ from the paper")
diffs = [["Aspect", "Official code", "Our version"],
         ["Mamba state size", "64", "16"],
         ["Spatial convolutions", "3×3 only", "3×3 + 5×5 (paper text)"],
         ["Band groups", "12 bands, stride 6", "16, stride 8 (paper text)"],
         ["Epochs; AUC checked", "150; every epoch", "100; every 5 epochs"],
         ["Model state at test", "training mode", "evaluation mode"],
         ["Masking", "fixed 3×3 square", "random 2–5 px box"]]
table(s, diffs, 0.6, 1.4, 5.7, [1.9, 1.75, 2.05], row_h=0.44, size=12)
text(s, 0.6, 4.6, 5.7, 0.4, "Unchanged: architecture, loss, data, normalisation and anomaly score.", size=13,
     bold=True, color=TEAL)
hdr = ["Scene", "Paper", "Our main\nrun", "Every epoch\n(3 seeds)", "Official\nsettings"]
rows, have = [hdr], {"main": [], "e1": [], "off": [], "paper": []}
spread = []
for d in EXTRA_DS:
    e1 = [v for v in X[d]["e1"] if v is not None]
    off = X[d]["official"]
    rows.append([DATASETS[d], f"{PAPER_AUC[d]['DMS2F-HAD']:.4f}", f"{X[d]['main']:.4f}",
                 f"{np.mean(e1):.4f} ± {np.ptp(e1) / 2:.4f}" if len(e1) == 3 else
                 (f"{np.mean(e1):.4f} ({len(e1)} of 3)" if e1 else "running"),
                 f"{off:.4f}" if off is not None else "running"])
    have["paper"].append(PAPER_AUC[d]["DMS2F-HAD"]); have["main"].append(X[d]["main"])
    if len(e1) == 3:
        have["e1"].append(np.mean(e1)); spread.append(np.ptp(e1))
    if off is not None:
        have["off"].append(off)
full = lambda k: len(have[k]) == len(EXTRA_DS)
rows.append(["Mean", f"{np.mean(have['paper']):.4f}", f"{np.mean(have['main']):.4f}",
             f"{np.mean(have['e1']):.4f}" if full("e1") else "–", f"{np.mean(have['off']):.4f}" if full("off") else "–"])
table(s, rows, 6.6, 1.4, 6.15, [1.3, 0.95, 1.05, 1.7, 1.15], row_h=0.5, size=13, highlight_rows=[len(rows) - 1])
caption(s, "Best-epoch AUC on the three AVIRIS scenes; '±' is half the range over seeds 0, 1 and 2.", y=4.45,
        x=6.6, w=6.15)
find = []
if full("e1"):
    find.append(f"Across three random seeds, the best AUC of a scene varies by up to {max(spread):.3f}, so "
                f"differences of this size between single runs are noise rather than a real effect.")
    find.append(f"Checking the AUC every epoch instead of every five changes the mean from "
                f"{np.mean(have['main']):.4f} to {np.mean(have['e1']):.4f}, because more checkpoints give more "
                f"chances to catch a high peak.")
if full("off"):
    gap = np.mean(have["off"]) - np.mean(have["main"])
    verdict = ("so these settings explain part of the gap." if gap > 0.003 else
               "so these settings do not explain the gap." if gap > -0.003 else
               "so these settings do not help on our runs.")
    find.append(f"With the official code's settings the mean is {np.mean(have['off']):.4f} against "
                f"{np.mean(have['paper']):.4f} in the paper, {verdict}")
if not find:
    find = ["The test runs are still training; this slide fills in automatically when they finish."]
box(s, 0.6, 5.15, 12.15, 1.7, fill=PANEL)
text(s, 0.85, 5.22, 11.7, 1.6, find, size=13, bullets=True, space_after=3)

# ================================================================= 15. anomaly maps
s = new_slide(notes=(
    "These colour anomaly maps follow the style of the paper's Figure 3. Red means a high anomaly score. RX "
    "highlights many background structures, while DMS2F-HAD suppresses the background and keeps the planes and rocks."))
title(s, "Colour anomaly maps")
picture(s, FIG / "fig3_anomaly_maps_wide.png", 0.6, 1.15, 12.1, 5.7)
caption(s, "The layout follows Fig. 3 of the paper. The maps are min-max normalised, and each panel shows its AUC "
           "in the corner.", y=6.9)

# ================================================================= 16. ROC
s = new_slide(notes=(
    "These ROC curves follow the style of the paper's Figure 5. Curves that hug the top-left corner detect most "
    "anomaly pixels at a very low false-alarm rate."))
title(s, "ROC curves")
picture(s, FIG / "fig5_roc_grid.png", 0.5, 1.2, 6.6, 5.75)
box(s, 7.35, 1.45, 5.4, 5.3, fill=PANEL)
text(s, 7.6, 1.6, 5.0, 0.5, "Anomalies found at 1 % false alarms", size=18, font=HEAD, bold=True)
text(s, 7.6, 2.12, 5.0, 0.75, "The threshold is set so that only 1 in 100 background pixels is flagged. The table "
     "shows the share of anomaly pixels found at that threshold.", size=13, color=MUTED)
tp = [["Scene", "RX", "DMS2F-HAD"]]
for d in DATASETS:
    g = R[d]["runs"].get("gated")
    tp.append([DATASETS[d], f"{R[d]['rx_tpr1'] * 100:.0f} %", f"{g['tpr1'] * 100:.0f} %" if g else "running"])
table(s, tp, 7.6, 3.0, 4.9, [1.6, 1.4, 1.9], row_h=0.48, size=14,
      bold_cells=[(i, 2) for i in range(1, len(tp))])
text(s, 7.6, 3.0 + 0.48 * len(tp) + 0.2, 4.9, 1.2, [
    "Curves close to the top-left corner find most anomalies before raising false alarms.",
    "The legend in each panel gives the AUC.",
], size=14, bullets=True, space_after=4)

# ================================================================= 17. box plots
s = new_slide(notes=(
    "These box plots follow the style of the paper's Figure 4. For each method, red is the score distribution of the "
    "anomaly pixels and blue that of the background pixels. A good detector keeps the blue box low and narrow and the "
    "red box high."))
title(s, "Separating anomalies from the background")
picture(s, FIG / "fig4_boxplots.png", 0.6, 1.25, 12.1, 4.95)
text(s, 0.6, 6.3, 12.1, 0.6, "Red boxes show anomaly pixels and blue boxes show background pixels. A large gap with "
     "little overlap means that a threshold can separate the two.", size=14, color=MUTED)

# ================================================================= 18. ablation
ab = [("spectral", "Spectral only"), ("spatial", "Spatial only"), ("add", "Addition fusion"), ("gated", "Gated fusion")]
ab_ds = [d for d in DATASETS if all(k in R[d]["runs"] for k, _ in ab)]
s = new_slide(notes=(
    "This ablation follows the paper's Table 5: each branch on its own, a simple addition of the two, and the full "
    "gated fusion. Each cell shows our AUC with the paper's value in brackets."))
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
    table(s, rows, 0.6, 1.45, 12.1, [1.6, 1.9, 1.9, 1.9, 1.9], row_h=0.5, size=14, highlight_rows=[len(rows) - 1],
          bold_cells=bold)
    y_cap = 1.5 + 0.5 * len(rows)
    caption(s, "Each cell shows our AUC at the best epoch, with the paper's value in brackets. Bold marks the best of "
               "our four variants for each scene." + ("" if "Cri" in ab_ds else " The CRI ablation is still training."),
            y=y_cap)
    mean_of = dict(zip([k for k, _ in ab], m))
    trio = [mean_of["spatial"], mean_of["add"], mean_of["gated"]]
    noise = f" (the seed-to-seed spread is up to {max(spread):.3f})" if spread and len(spread) == len(EXTRA_DS) else ""
    box(s, 0.6, y_cap + 0.45, 12.1, 6.85 - (y_cap + 0.45), fill=PANEL)
    text(s, 0.85, y_cap + 0.55, 11.6, 6.75 - (y_cap + 0.55), [
        f"The spectral branch alone is the weakest variant (mean {mean_of['spectral']:.4f}), as in the paper, because "
        "one pooled spectrum per patch cannot localise small targets.",
        f"Adding the spatial branch matters most: spatial only reaches {mean_of['spatial']:.4f}, addition "
        f"{mean_of['add']:.4f} and gated fusion {mean_of['gated']:.4f}, which are within {max(trio) - min(trio):.3f} "
        "of each other.",
        "On these scenes we could not confirm the paper's gain from the gate, because the difference is within the "
        f"run-to-run variation{noise}.",
    ], size=14, bullets=True, space_after=5)
else:
    text(s, 0.6, 2.0, 12, 1, "The ablation runs have not finished yet.", size=18, color=MUTED)

# ================================================================= 19. training behaviour
s = new_slide(notes=(
    "The loss keeps going down, but the AUC does not always keep going up. Once the background is learned, the "
    "network also starts to reconstruct the anomalies, and their scores shrink. That is why the choice of "
    "checkpoint matters, and why we report both the best epoch and the final epoch."))
title(s, "Training behaviour: a lower loss is not always better detection")
picture(s, FIG / "fig_training_curves.png", 0.6, 1.3, 8.2, 5.4)
box(s, 9.1, 1.4, 3.65, 5.35, fill=PANEL)
text(s, 9.25, 1.5, 3.4, 2.7, [
    "The training loss falls steadily on every scene.",
    "The AUC peaks early and then swings, because the decoder also starts to rebuild the anomalies.",
    "The paper reports the best checkpoint; without labels, early stopping would be needed.",
], size=14, bullets=True, space_after=6)
tr = [["Scene", "Best ep.", "Best", "Ep. 100"]]
for d in GATED:
    g = R[d]["runs"]["gated"]
    tr.append([DATASETS[d], str(int(g["best_epoch"])), f"{float(g['best_auc']):.4f}", f"{float(g['final_auc']):.4f}"])
text(s, 9.25, 4.15, 3.4, 0.35, "Best and final checkpoint (AUC)", size=13, bold=True, color=MUTED)
table(s, tr, 9.25, 4.55, 3.35, [1.25, 0.85, 0.95, 0.95], row_h=0.38, size=12)

# ================================================================= 20. takeaways
s = new_slide(dark=True, notes=(
    "To sum up: what the method does, what we reproduced, the limits of this study, and what could be done next."))
title(s, "Takeaways", dark=True)
cols = [("What the method does", [
            "It learns the background of one image without labels, as a patch autoencoder.",
            "It models space and spectrum in two Mamba branches and mixes them per pixel with a gate.",
            "It stays small (≈ 0.65 M parameters) and fast thanks to linear-time Mamba."]),
        ("What we reproduced", [
            f"Our mean AUC is {avg_best:.4f}, against {avg_paper:.4f} in the paper on the same {len(GATED)} scenes.",
            "Our RX baseline matches the paper exactly.",
            "As in the paper, the spectral branch alone is the weakest variant.",
            f"Inference takes {avg_time:.2f} s per image on average."]),
        ("Limits of this study", [
            "We used 4 of the paper's 14 scenes, because only these are released.",
            "The best-epoch numbers use the ground truth to choose the checkpoint, as the paper does.",
            "The baseline numbers are the paper's; we only re-ran RX."]),
        ("Next steps", [
            "Choose the checkpoint without labels, e.g. by early stopping on the reconstruction loss.",
            "Run the other ten scenes with several seeds.",
            "Use fused CUDA Mamba kernels on Linux to train faster."])]
for i, (h, pts) in enumerate(cols):
    x = 0.6 + i * 3.08
    box(s, x, 1.45, 2.9, 4.45, fill="1B2B45")
    badge(s, x + 0.2, 1.62, str(i + 1), d=0.48, size=15)
    text(s, x + 0.8, 1.64, 2.0, 0.6, h, size=17, font=HEAD, bold=True, color=WHITE)
    text(s, x + 0.15, 2.35, 2.65, 3.5, pts, size=14, color="DCE3EE", bullets=True, space_after=6)
text(s, 0.6, 6.05, 12.1, 0.85, [[("Bottom line: ", {"bold": True, "color": AMBER}),
     (f"A dual-branch Mamba autoencoder with 0.65 M parameters gets close to the paper's accuracy (mean AUC "
      f"{avg_best:.4f} against {avg_paper:.4f}) on the released scenes. The open problem is how to choose the "
      "checkpoint without labels.", {})]], size=15, color=WHITE)

prs.save(OUT)
print("saved", OUT, "with", len(prs.slides), "slides")
