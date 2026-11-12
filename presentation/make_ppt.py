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
        # "gated16" = the earlier full-model run (state size 16, AUC every 5 epochs), kept for the ablation,
        # whose other variants were trained with those settings
        for f, p in [("gated", RUNS / f"{d}_gated.npz"), ("gated16", RUNS / "extra" / f"{d}_gated_state16.npz")] + \
                    [(f, RUNS / f"{d}_{f}.npz") for f in ["add", "spatial", "spectral"]]:
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
P_MAIN = f"{max(params.values()) / 1e6:.2f} M" if params else "n/a"     # size of the 189-band model

# ---------------------------------------------------------------- extra test runs (results/runs/extra)
EXTRA_DS = ["aviris_1", "aviris_2", "San_Diego"]


def extra(d, tag):
    p = RUNS / "extra" / f"{d}_gated_{tag}.npz"
    return float(np.load(p)["best_auc"]) if p.exists() else None


# main = current run (state 64, AUC every epoch); s16 = earlier run (state 16, every 5 epochs);
# e1 = state 16 with AUC every epoch, seeds 0-2; official = all settings of the official code
X = {d: {"main": float(R[d]["runs"]["gated"]["best_auc"]) if "gated" in R[d]["runs"] else None,
         "s16": extra(d, "state16"), "e1": [extra(d, f"every1_s{k}") for k in range(3)],
         "official": extra(d, "official")} for d in EXTRA_DS}
seed_spread = [np.ptp(X[d]["e1"]) for d in EXTRA_DS if all(v is not None for v in X[d]["e1"])]
n_runs = len(list(RUNS.glob("*.npz"))) + len(list((RUNS / "extra").glob("*.npz")))
# largest AUC swing between checkpoints after epoch 10 of the main runs (training instability)
swing = max(float(np.ptp(h[h[:, 0] > 10, 1])) for h in
            (np.load(RUNS / f"{d}_gated.npz")["auc_hist"] for d in GATED)) if GATED else 0.0

# ================================================================= 1. title
s = new_slide(dark=True, notes=(
    "This project re-implements DMS2F-HAD, a WACV 2026 paper on hyperspectral anomaly detection. I explain what "
    "hyperspectral images and anomaly detection are, how the method works, which datasets were used and why, and how "
    "my own results compare with the paper and with other models."))
text(s, 0.7, 0.85, 7.4, 0.4, "WACV 2026  ·  paper implementation project", size=14, color=AMBER, bold=True)
text(s, 0.7, 1.3, 7.4, 1.1, "DMS2F-HAD", size=60, font=HEAD, bold=True, color=WHITE)
text(s, 0.7, 2.5, 7.2, 1.3, "A Dual-branch Mamba-based Spatial–Spectral Fusion Network for Hyperspectral "
     "Anomaly Detection", size=22, font=HEAD, color="DCE3EE")
text(s, 0.7, 3.95, 7.2, 1.3, [
    [("Paper: ", {"bold": True, "color": AMBER}), ("A. Pant, L. Tamang, T.-K. Lee and S. Aryal, Deakin University.", {})],
    [("This project: ", {"bold": True, "color": AMBER}),
     ("We re-implemented the method in PyTorch and trained and evaluated it on four benchmark scenes.", {})],
], size=15, color="DCE3EE")
picture(s, ASSETS / "hsi_cube_dark.png", 8.3, 0.6, 4.5, 4.6)
box(s, 0.6, 5.45, 12.15, 1.55, fill="1B2B45")
for i, (v, l) in enumerate([("4", "benchmark scenes"), (str(n_runs), "models trained"), (P_MAIN, "parameters"),
                            (f"{avg_best:.4f}", "our mean AUC"), ("7", "methods compared")]):
    x = 0.85 + i * 2.4
    text(s, x, 5.55, 2.2, 0.7, v, size=30, font=HEAD, bold=True, color=WHITE)
    text(s, x, 6.25, 2.2, 0.45, l, size=13, color="B9C4D4")

# ================================================================= 2. HSI + HAD
s = new_slide(notes=(
    "A normal camera records three broad colour bands, while a hyperspectral sensor records hundreds of narrow bands, "
    "so every pixel is a full spectrum. Anomaly detection gives every pixel a score for how different it is from the "
    "background, without any labels. In San Diego the planes are rare and their spectra differ from the apron and the "
    "roads, which is exactly what HAD exploits."))
title(s, "Hyperspectral images and anomaly detection")
text(s, 0.6, 1.35, 6.0, 0.45, "A hyperspectral image is a cube of spectra", size=18, font=HEAD, bold=True, color=TEAL)
text(s, 0.6, 1.8, 6.0, 1.9, [
    "A hyperspectral sensor records 100 to 200+ narrow, contiguous bands instead of three colours. AVIRIS, for "
    "example, records 224 bands between 370 and 2510 nm.",
    "The image is an H × W × C cube, and each pixel is a spectrum that acts like a material fingerprint.",
], size=14, bullets=True, space_after=6)
text(s, 0.6, 3.75, 6.0, 0.45, "Anomaly detection (HAD) finds pixels that do not fit", size=18, font=HEAD, bold=True,
     color=CORAL)
text(s, 0.6, 4.2, 6.0, 1.75, [
    "Every pixel gets an anomaly score; together, the scores form an anomaly map.",
    "The task is unsupervised: there are no labels and no target spectra, so the background has to be learned "
    "from the image itself.",
    "Anomalies are small, rare (often under 1 % of pixels) and spectrally different, such as planes or rocks.",
], size=14, bullets=True, space_after=5)
picture(s, FIG / "fig_spectra.png", 6.8, 1.3, 6.0, 4.6)
text(s, 0.6, 6.05, 3.0, 0.35, "HAD is used in many fields:", size=13, bold=True, color=MUTED)
for i, a in enumerate(["Defence & surveillance", "Search and rescue", "Mineral exploration",
                       "Environmental monitoring", "Precision agriculture"]):
    label_box(s, 0.6 + i * 2.45, 6.4, 2.3, 0.5, a, fill="E3F1EF", color=INK, size=13, bold=True)

# ================================================================= 3. families + evaluation
s = new_slide(notes=(
    "HAD research has moved from statistics to deep learning. RX is the classic baseline, representation methods learn "
    "a background dictionary, and deep models learn non-linear backgrounds. Mamba-based models are the newest family. "
    "Detectors are judged with anomaly maps, ROC curves and the AUC, and box plots of the scores."))
title(s, "Families of HAD methods and how they are evaluated")
fam = [("Statistical", "RX scores each pixel by its Mahalanobis distance to a Gaussian background.",
        "It is fast and needs no training.", "It fails in cluttered scenes."),
       ("Representation", "These methods learn a background dictionary or a low-rank plus sparse split.",
        "They separate outliers better than RX.", "They are slow and sensitive to settings."),
       ("Deep learning", "Autoencoders, CNNs and Transformers (AUTO-AD, LREN, RGAE, TDD, GT-HAD) learn the background.",
        "They learn non-linear backgrounds.", "CNNs see little context; attention costs O(L²)."),
       ("Mamba (SSM)", "MMR-HAD and DMS2F-HAD use selective state-space models.",
        "They see long-range context in linear time.", "The research area is still young.")]
for i, (h, d, pro, con) in enumerate(fam):
    x = 0.6 + i * 3.08
    hl = i == 3
    box(s, x, 1.35, 2.9, 3.35, fill="FDF1DC" if hl else PANEL)
    badge(s, x + 0.18, 1.5, str(i + 1), d=0.44, size=14, fill=AMBER if hl else "C9D3E0")
    text(s, x + 0.72, 1.52, 2.1, 0.45, h, size=17, font=HEAD, bold=True)
    text(s, x + 0.18, 2.02, 2.6, 1.05, d, size=13, color=INK)
    text(s, x + 0.18, 3.08, 2.6, 0.28, "Strength", size=12, bold=True, color=TEAL)
    text(s, x + 0.18, 3.33, 2.6, 0.5, pro, size=13)
    text(s, x + 0.18, 3.85, 2.6, 0.28, "Weakness", size=12, bold=True, color=CORAL)
    text(s, x + 0.18, 4.1, 2.6, 0.55, con, size=13)
ev = [("Anomaly map", "Scores are drawn with a jet colour map, in which blue is background and red is anomaly, and "
                      "compared with the ground truth by eye."),
      ("ROC curve and AUC", "Sweeping the threshold gives the detection rate against the false-alarm rate. An AUC of 1 "
                            "is perfect and 0.5 is guessing."),
      ("Score box plots", "The score distributions of anomaly and background pixels are compared. Little overlap "
                          "means few false alarms.")]
for i, (h, d) in enumerate(ev):
    x = 0.6 + i * 4.1
    box(s, x, 4.9, 3.95, 2.0, fill=INK)
    text(s, x + 0.2, 5.0, 3.6, 0.45, h, size=17, font=HEAD, bold=True, color=AMBER)
    text(s, x + 0.2, 5.48, 3.6, 1.35, d, size=14, color="DCE3EE")

# ================================================================= 4. Mamba
s = new_slide(notes=(
    "A state-space model keeps a hidden state h and updates it at every position of the sequence. Mamba makes the "
    "step size and the B and C matrices depend on the input, so it can choose what to remember. The cost grows "
    "linearly with the sequence length, while self-attention grows quadratically."))
title(s, "Why Mamba? Long-range modelling in linear time")
box(s, 0.6, 1.4, 5.9, 1.55, fill=INK)
text(s, 0.85, 1.5, 5.5, 1.4, [
    [("hₜ = Āₜ · hₜ₋₁ + B̄ₜ · xₜ", {"font": "Cambria Math", "size": 24, "color": WHITE})],
    [("yₜ = Cₜ · hₜ + D · xₜ", {"font": "Cambria Math", "size": 24, "color": WHITE})],
], size=22, color=WHITE, space_after=4)
text(s, 0.6, 3.15, 5.9, 3.8, [
    "A state-space model carries a hidden state h along a sequence of tokens.",
    "In the selective version (S6), Δ, B and C are computed from the input, so the model decides what to keep and "
    "what to forget.",
    "The cost grows linearly with the sequence length L, whereas self-attention grows with L².",
    "In DMS2F-HAD, a 16 × 16 patch becomes 256 pixel tokens for the spatial branch and 7 band groups for the "
    "spectral branch.",
    "Our version runs the scan in plain PyTorch as an exact chunked scan, so no special CUDA library is needed.",
], size=15, bullets=True, space_after=7)
cd = CategoryChartData()
Ls = [256, 512, 1024, 2048, 4096]
cd.categories = [str(L) for L in Ls]
cd.add_series("Self-attention  ~L²", [L * L / 1e6 for L in Ls])
cd.add_series("Mamba  ~L·N (state size N = 64)", [L * 64 / 1e6 for L in Ls])
ch = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(6.9), Inches(1.35), Inches(5.9), Inches(5.55), cd).chart
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
for ax in (ch.category_axis, ch.value_axis):
    ax.tick_labels.font.size = Pt(12)
    ax.tick_labels.font.color.rgb = rgb(MUTED)
ch.value_axis.major_gridlines.format.line.color.rgb = rgb("E3E8EF")
ch.value_axis.format.line.fill.background()
ch.category_axis.has_title = True
ch.category_axis.axis_title.text_frame.text = "Sequence length L (tokens)"
ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)

# ================================================================= 5. architecture
s = new_slide(notes=(
    "The image is cut into overlapping 16x16 patches, and random rectangles are masked during training. A 1x1 "
    "convolution reduces the bands to 64 features. The spatial branch looks at texture, the spectral branch at "
    "correlations between bands, a learned gate mixes them pixel by pixel, and a decoder rebuilds the patch. "
    "The reconstruction error is the anomaly score."))
title(s, "The DMS2F-HAD architecture")
picture(s, ASSETS / "paper_fig2_architecture.png", 0.45, 1.3, 7.45, 5.3)
caption(s, "Figure 2 of the paper: (a) the framework, (b) the Mamba block and (c) the decoder block.", y=6.65,
        x=0.6, w=7.2)
box(s, 8.05, 1.3, 4.75, 5.65, fill=PANEL)
text(s, 8.25, 1.38, 4.4, 0.8, "An autoencoder learns to rebuild the background of one image; anomalies are rebuilt "
     "badly and stand out.", size=13, bold=True, color=INK)
blocks = [("1", TEAL, "Spatial branch", "3×3 and 5×5 convolutions, then Mamba over the 256 pixels of a patch, give "
           "every pixel global context."),
          ("2", CORAL, "Spectral branch", "The 64 features are split into 7 overlapping groups of 16 bands, and Mamba "
           "models how the groups relate."),
          ("3", AMBER, "Gated fusion", "A 1×1 convolution with a sigmoid gives each pixel a weight G that mixes "
           "the two branches."),
          ("4", INK, "Decoder and score", "Two Mamba-plus-convolution blocks rebuild the patch; a pixel's score is "
           "its error ‖x − x̂‖₂.")]
for i, (n, col, h, d) in enumerate(blocks):
    y = 2.25 + i * 1.17
    badge(s, 8.25, y + 0.03, n, d=0.38, fill=col, color=WHITE if col == INK else INK, size=12)
    text(s, 8.75, y, 3.9, 0.35, h, size=14, bold=True)
    text(s, 8.75, y + 0.33, 3.95, 0.8, d, size=13, color=INK)

# ================================================================= 6. datasets in detail
s = new_slide(notes=(
    "These are the four scenes released with the official code. CRI is a large close-up of grassland with ten rocks, "
    "taken with a Nuance CRI tunable-filter camera in the near infrared. The three AVIRIS scenes are airborne images "
    "of airports with three planes each, at 3.5 m per pixel. AVIRIS-2 and San Diego come from the same flight and "
    "overlap, but each has its own ground-truth mask."))
title(s, "The four benchmark scenes in detail")
order4 = ["Cri", "aviris_1", "aviris_2", "San_Diego"]
for i, d in enumerate(order4):
    picture(s, FIG / f"thumb_{d}.png", 2.65 + i * 2.53, 1.25, 2.3, 1.6)
facts = [["", "CRI", "AVIRIS-1", "AVIRIS-2", "San Diego"],
         ["Sensor", "Nuance CRI tunable-filter camera", "AVIRIS (NASA/JPL), airborne", "AVIRIS (NASA/JPL), airborne",
          "AVIRIS (NASA/JPL), airborne"],
         ["Scene", "Grassland with rocks", "Airport with planes", "Airport with planes", "Airport with planes"],
         ["Size (H × W × bands)", "400 × 400 × 46", "100 × 100 × 189", "128 × 128 × 189", "100 × 100 × 189"],
         ["Spectral range", "650–1100 nm (10 nm steps)", "370–2510 nm; 189 of 224 bands", "370–2510 nm; 189 of 224 bands",
          "370–2510 nm; 189 of 224 bands"],
         ["Pixel size", "not reported", "3.5 m", "3.5 m", "3.5 m"],
         ["Anomalies", "10 rocks, 15–180 px each", "3 planes, 20–22 px each", "3 planes, 34–44 px each",
          "3 planes, 38–56 px each"],
         ["Anomaly pixels", "860 (0.54 %)", "64 (0.64 %)", "120 (0.73 %)", "134 (1.34 %)"]]
table(s, facts, 0.6, 2.95, 12.15, [2.05, 2.53, 2.53, 2.53, 2.53], row_h=0.44, size=12)
caption(s, "Red outlines in the thumbnails mark the ground-truth anomalies. AVIRIS bands affected by water absorption "
           "and noise are removed. AVIRIS-2 and San Diego come from the same flight: 80 % of San Diego's pixels also "
           "appear in AVIRIS-2.", y=6.5)

# ================================================================= 7. what each scene tests
s = new_slide(notes=(
    "Each scene stands for a different use case. CRI is a natural scene with small objects in vegetation, close to "
    "search and rescue. The AVIRIS scenes are airborne surveillance of airfields, where the targets are aircraft and "
    "the background is a mix of concrete, roofs, roads and soil. The paper uses 14 scenes in total, but only these "
    "four are released with the code."))
title(s, "What each scene is used for and why it is hard")
uses = {"Cri": ("Finding small natural objects, such as rocks or debris, on vegetated ground, as in search and rescue.",
                "The grass texture is strong, the bands cover only red and near-infrared light (650–1100 nm), and "
                "the image has 160 000 pixels."),
        "aviris_1": ("Airborne surveillance: spotting small aircraft parked on an airfield.",
                     f"The planes are tiny (about 20 pixels each). At 1 % false alarms RX finds only "
                     f"{R['aviris_1']['rx_tpr1'] * 100:.0f} % of their pixels."),
        "aviris_2": ("Airborne surveillance over a larger and busier part of the airport.",
                     "Buildings, roofs and vehicles also look unusual and compete with the planes for the top scores."),
        "San_Diego": ("The classic HAD benchmark: aircraft on an airport apron, imaged from the air.",
                      "Bright roofs and building edges produce false alarms, especially for the RX baseline.")}
for i, d in enumerate(order4):
    x = 0.6 + i * 3.08
    box(s, x, 1.35, 2.9, 5.55, fill=PANEL)
    text(s, x + 0.2, 1.45, 2.5, 0.45, DATASETS[d], size=19, font=HEAD, bold=True)
    text(s, x + 0.2, 1.95, 2.5, 0.3, "Use case", size=12, bold=True, color=TEAL)
    text(s, x + 0.2, 2.22, 2.55, 1.45, uses[d][0], size=14)
    text(s, x + 0.2, 3.7, 2.5, 0.3, "Challenge", size=12, bold=True, color=CORAL)
    text(s, x + 0.2, 3.97, 2.55, 1.5, uses[d][1], size=14)
    g = R[d]["runs"].get("gated")
    box(s, x + 0.15, 5.6, 2.6, 1.15, fill=WHITE)
    text(s, x + 0.25, 5.65, 2.45, 1.05, [
        [("AUC  ", {"bold": True, "color": MUTED, "size": 12})],
        [("Ours ", {"size": 13}), (f"{float(g['best_auc']):.4f}" if g else "running", {"bold": True, "size": 13}),
         ("  ·  Paper ", {"size": 13}), (f"{PAPER_AUC[d]['DMS2F-HAD']:.4f}", {"bold": True, "size": 13})],
        [("RX ", {"size": 13}), (f"{R[d]['rx']:.4f}", {"bold": True, "size": 13})],
    ], size=13, space_after=1)
caption(s, "The paper evaluates 14 scenes (urban ABU scenes, Salinas, Pavia, Gulfport and others); only these four are "
           "released with the official code.", y=6.95)

# ================================================================= 8. implementation
p_groups = {}
for d in GATED:
    p_groups.setdefault(params[d], []).append(DATASETS[d])
p_str = " and ".join(f"{p/1e6:.3f} M for {', '.join(n)}" for p, n in sorted(p_groups.items(), reverse=True)) or "n/a"
s = new_slide(notes=(
    "The official code depends on the mamba_ssm CUDA kernels, which do not build on Windows. I wrote the Mamba "
    "selective scan in plain PyTorch as an exact two-level chunked scan and kept the paper's training settings. Each "
    "image is trained on its own, without labels, and the ground truth is only used to compute the AUC."))
title(s, "Implementation and training setup")
hp = [["Setting", "Value"], ["Patch size / stride", "16 × 16 / 8"], ["Embedding c₁", "64"],
      ["Band group c₂ / stride k", "16 / 8  (7 groups)"], ["Mamba state size", "64 (official code)"],
      ["Optimiser", "Adam, lr 5e-4, weight decay 1e-4"], ["Epochs / batch", "100 / 32"],
      ["Loss", "MSE + 0.1 · L1 (official code)"], ["Masking", "p = 0.5, random 2–5 px rectangle"],
      ["Model selection", "best AUC, checked every epoch (code)"], ["Border padding", "zeros (official code)"]]
table(s, hp, 0.6, 1.4, 5.7, [2.3, 3.0], row_h=0.5, size=13)
text(s, 6.6, 1.35, 6.2, 0.45, "The method fits in about 440 lines of Python", size=18, font=HEAD, bold=True)
files = [("model.py", "Implements Mamba and the network."),
         ("utils.py", "Loads data, patches it, runs RX and scores AUC."),
         ("train.py", "Trains on one image and saves its map."),
         ("make_figures.py", "Draws the paper-style figures."),
         ("presentation/make_ppt.py", "Builds this slide deck.")]
for i, (f, d) in enumerate(files):
    y = 1.9 + i * 0.6
    box(s, 6.6, y, 6.2, 0.5, fill=PANEL)
    text(s, 6.75, y + 0.06, 2.6, 0.4, f, size=13, font="Consolas", bold=True, color=INK)
    text(s, 9.35, y + 0.07, 3.4, 0.4, d, size=12, color=MUTED)
box(s, 6.6, 5.0, 6.2, 1.9, fill="FDF1DC")
text(s, 6.8, 5.08, 5.85, 1.8, [
    [("Rule: ", {"bold": True}), ("we follow the paper where it is explicit and the official code where the paper is "
                                 "silent.", {})],
    "The selective scan runs as an exact chunked scan in plain PyTorch, so no mamba_ssm library is needed and the "
    "code runs on Windows, on a CPU or on any GPU.",
    f"The model has {p_str}; each image is trained on its own, without labels.",
], size=13, space_after=4)

# ================================================================= 9. models compared (paper)
s = new_slide(notes=(
    "The paper compares DMS2F-HAD with six detectors on 14 scenes and with MMR-HAD on four shared scenes. DMS2F-HAD "
    "has the highest mean AUC, 0.9878, and is the best model on 9 of the 14 scenes. It is also the fastest: 0.555 s "
    "per image on an A100, 4.6 times faster than TDD and 65 times faster than GT-HAD."))
title(s, "The models it beats: the paper's benchmark")
cats = [n for n, _ in PAPER_AVG14]
bar_chart(s, 0.6, 1.3, 5.7, 5.6, cats, [v for _, v in PAPER_AVG14], "Mean AUC over 14 scenes (paper Table 2)",
          [AMBER if n == "DMS2F-HAD" else "9AA8BA" for n in cats], vmax=1.12)
mt = [["Model", "Type", "Params", "Time (s)", "AUC (14)"],
      ["RX", "Statistical, 1990", "–", "–", "0.9523"],
      ["AUTO-AD", "Autoencoder, 2022", "3.23 M", "9.32", "0.9189"],
      ["LREN", "Low-rank AE, 2021", "0.86 M", "–", "0.8297"],
      ["RGAE", "Graph AE, 2022", "–", "–", "0.9326"],
      ["TDD", "Transfer, 2024", "46.64 M", "2.55", "0.5900"],
      ["GT-HAD", "Transformer, 2025", "0.26 M", "36.26", "0.9774"],
      ["MMR-HAD", "Mamba + attn., 2025", "2.12 M", "–", "–"],
      ["DMS2F-HAD", "Dual Mamba, 2026", "0.64 M", "0.555", "0.9878"]]
table(s, mt, 6.55, 1.35, 6.25, [1.35, 1.85, 1.0, 0.95, 1.1], row_h=0.4, size=12, highlight_rows=[8])
box(s, 6.55, 5.1, 6.25, 1.8, fill=PANEL)
text(s, 6.75, 5.17, 5.9, 1.7, [
    "DMS2F-HAD is best on 9 of the 14 scenes, 0.0104 AUC above GT-HAD, the next best.",
    "It is 4.6× faster than TDD, 17× faster than AUTO-AD and 65× faster than GT-HAD.",
    "Against MMR-HAD each wins two of four shared scenes, but DMS2F-HAD has 3.3× fewer parameters and 29× fewer FLOPs.",
], size=13, bullets=True, space_after=3)

# ================================================================= 10. our results vs paper and baselines
s = new_slide(notes=(
    "Here are our own runs next to the paper's numbers for every baseline on the four scenes we have. 'Best' follows "
    "the paper's protocol and keeps the checkpoint with the highest AUC; 'epoch 100' is the model at the end of "
    "training, which needs no labels to choose."))
title(s, "Our results vs. the paper and the baselines")
stats = [(f"{avg_best:.4f}", "Our mean AUC (best epoch)", TEAL), (f"{avg_paper:.4f}", "The paper's mean AUC", INK),
         (f"{avg_final:.4f}", "Our mean AUC (epoch 100)", MUTED), (f"{avg_rx:.4f}", "Mean AUC of RX", MUTED)]
for i, (v, l, col) in enumerate(stats):
    x = 0.6 + i * 3.08
    box(s, x, 1.3, 2.9, 1.2, fill=PANEL)
    text(s, x + 0.1, 1.33, 2.7, 0.7, v, size=30, font=HEAD, bold=True, color=col, align=PP_ALIGN.CENTER)
    text(s, x + 0.1, 2.0, 2.7, 0.4, l, size=13, color=MUTED, align=PP_ALIGN.CENTER)
hdr = ["Scene"] + BASELINES + ["Paper", "Ours\n(best)", "Ours\n(ep. 100)"]
rows, bold, beat, ranks = [hdr], [], 0, []
for d in DATASETS:
    g = R[d]["runs"].get("gated")
    v = PAPER_T2[d]
    if g is None:
        rows.append([DATASETS[d]] + [f"{a:.4f}" for a in v] + ["running", "running"]); continue
    ours, fin = float(g["best_auc"]), float(g["final_auc"])
    rows.append([DATASETS[d]] + [f"{a:.4f}" for a in v] + [f"{ours:.4f}", f"{fin:.4f}"])
    allv = v + [ours]
    bold.append((len(rows) - 1, 1 + int(np.argmax(allv))))
    beat += ours > max(v[:6])
    ranks.append(f"#{1 + sorted(v[:6] + [ours], reverse=True).index(ours)} on {DATASETS[d]}")
avg = [np.mean([PAPER_T2[d][j] for d in GATED]) for j in range(7)] + [avg_best, avg_final]
rows.append(["Mean"] + [f"{a:.4f}" for a in avg])
bold.append((len(rows) - 1, 1 + int(np.argmax(avg[:8]))))
table(s, rows, 0.6, 2.7, 12.15, [1.25, 0.95, 1.05, 0.95, 0.95, 0.95, 1.05, 1.0, 1.0, 1.1], row_h=0.47, size=14,
      highlight_rows=[len(rows) - 1], bold_cells=bold)
text(s, 0.6, 2.7 + 0.47 * len(rows) + 0.15, 12.15, 1.2, [
    f"Our run beats all six baselines on {beat} of {len(GATED)} scenes and ranks " + ", ".join(ranks[:-1]) +
    f" and {ranks[-1]} among the seven methods." if ranks else "Runs are still training.",
    f"Our mean AUC is {abs(avg_best - avg_paper):.4f} {'below' if avg_best < avg_paper else 'above'} the paper's. "
    f"The AUC swings during training, so the model after 100 epochs scores {avg_final:.4f}; the paper, like us, reports "
    "the best checkpoint.",
], size=14, bullets=True, space_after=6)
caption(s, "Bold marks the best value in each row. Baseline AUCs are the paper's (Table 2); only RX was re-run, and it "
           "matches the paper exactly.", y=6.95)

# ================================================================= 11. anomaly maps
s = new_slide(notes=(
    "These colour anomaly maps follow the style of the paper's Figure 3. Red means a high anomaly score. RX "
    "highlights many background structures, while DMS2F-HAD suppresses the background and keeps the planes and rocks."))
title(s, "Colour anomaly maps")
picture(s, FIG / "fig3_anomaly_maps_wide.png", 0.6, 1.15, 12.1, 5.7)
caption(s, "The layout follows Fig. 3 of the paper. Maps are min-max normalised, each panel shows its AUC, and the three "
           "ablation variants use state size 16.", y=6.9)

# ================================================================= 12. ROC
s = new_slide(notes=(
    "These ROC curves follow the style of the paper's Figure 5. Curves that hug the top-left corner detect most "
    "anomaly pixels at a very low false-alarm rate. The table on the right reads off one point of each curve."))
title(s, "ROC curves: detection rate against false alarms")
picture(s, FIG / "fig5_roc_grid.png", 0.5, 1.2, 6.6, 5.75)
box(s, 7.35, 1.35, 5.4, 5.5, fill=PANEL)
text(s, 7.6, 1.48, 5.0, 0.5, "Anomalies found at 1 % false alarms", size=18, font=HEAD, bold=True)
text(s, 7.6, 2.0, 5.0, 0.75, "The threshold is set so that only 1 in 100 background pixels is flagged. The table "
     "shows the share of anomaly pixels found at that threshold.", size=13, color=MUTED)
tp = [["Scene", "RX", "DMS2F-HAD"]]
for d in DATASETS:
    g = R[d]["runs"].get("gated")
    tp.append([DATASETS[d], f"{R[d]['rx_tpr1'] * 100:.0f} %", f"{g['tpr1'] * 100:.0f} %" if g else "running"])
table(s, tp, 7.6, 2.9, 4.9, [1.6, 1.4, 1.9], row_h=0.46, size=14, bold_cells=[(i, 2) for i in range(1, len(tp))])
text(s, 7.6, 2.9 + 0.46 * len(tp) + 0.15, 4.9, 1.7, [
    "Curves close to the top-left corner find most anomalies before raising false alarms.",
    (lambda r: f"On the AVIRIS scenes, DMS2F-HAD finds {min(r):.1f} to {max(r):.0f} times as many anomaly pixels as RX."
     if r else "The DMS2F-HAD runs are still training.")(
        [R[d]["runs"]["gated"]["tpr1"] / R[d]["rx_tpr1"] for d in EXTRA_DS if "gated" in R[d]["runs"]]),
    "The legend in each panel gives the AUC.",
], size=13, bullets=True, space_after=3)

# ================================================================= 13. ablation
ab = [("spectral", "Spectral only"), ("spatial", "Spatial only"), ("add", "Addition fusion"), ("gated16", "Gated fusion")]
pkey = lambda k: "DMS2F-HAD" if k.startswith("gated") else k
ab_ds = [d for d in DATASETS if all(k in R[d]["runs"] for k, _ in ab)]
s = new_slide(notes=(
    "This ablation follows the paper's Table 5: each branch on its own, a simple addition of the two, and the full "
    "gated fusion. Each cell shows our AUC with the paper's value in brackets. All four variants were trained with the "
    "earlier settings, state size 16 and the AUC checked every 5 epochs, so they are comparable with each other."))
title(s, "Ablation: do we need both branches and the gate?")
if ab_ds:
    rows = [["Scene"] + [n for _, n in ab]]
    bold = []
    for d in ab_ds:
        a = [float(R[d]["runs"][k]["best_auc"]) for k, _ in ab]
        pk = [PAPER_AUC[d][pkey(k)] for k, _ in ab]
        rows.append([DATASETS[d]] + [f"{x:.4f}  ({p:.4f})" for x, p in zip(a, pk)])
        bold.append((len(rows) - 1, 1 + int(np.argmax(a))))
    m = [mean([float(R[d]["runs"][k]["best_auc"]) for d in ab_ds]) for k, _ in ab]
    pm = [mean([PAPER_AUC[d][pkey(k)] for d in ab_ds]) for k, _ in ab]
    rows.append(["Mean"] + [f"{x:.4f}  ({p:.4f})" for x, p in zip(m, pm)])
    table(s, rows, 0.6, 1.35, 12.15, [1.6, 1.9, 1.9, 1.9, 1.9], row_h=0.55, size=15, highlight_rows=[len(rows) - 1],
          bold_cells=bold)
    y_cap = 1.4 + 0.55 * len(rows)
    caption(s, "Our AUC at the best epoch, with the paper's value in brackets; bold marks our best variant per scene. All "
               "four variants use the earlier settings (state size 16, AUC checked every 5 epochs).", y=y_cap)
    mo = dict(zip([k for k, _ in ab], m))
    mo["gated"] = mo["gated16"]
    trio = [mo["spatial"], mo["add"], mo["gated"]]
    noise = f" (seed-to-seed spread up to {max(seed_spread):.3f})" if len(seed_spread) == len(EXTRA_DS) else ""
    box(s, 0.6, y_cap + 0.45, 12.15, 6.9 - (y_cap + 0.45), fill=PANEL)
    text(s, 0.85, y_cap + 0.55, 11.7, 6.8 - (y_cap + 0.55), [
        f"The spectral branch alone is the weakest variant (mean {mo['spectral']:.4f}), as in the paper, because one "
        "pooled spectrum per patch cannot localise small targets.",
        f"Adding the spatial branch matters most: spatial only reaches {mo['spatial']:.4f}, addition {mo['add']:.4f} "
        f"and gated fusion {mo['gated']:.4f}, which are within {max(trio) - min(trio):.3f} of each other.",
        "On these scenes we could not confirm the paper's gain from the gate, because the difference is within the "
        f"run-to-run variation{noise}.",
    ], size=14, bullets=True, space_after=6)
else:
    text(s, 0.6, 2.0, 12, 1, "The ablation runs have not finished yet.", size=18, color=MUTED)

# ================================================================= 14. why results differ
s = new_slide(notes=(
    "We follow the paper where it is explicit and the official code where the paper is silent, so our state size, "
    "checkpoint checks and padding now match the code. The remaining differences are places where the paper's text "
    "and the code disagree, where we keep the paper. The extra runs show how much each source of difference matters."))
title(s, "Why our results differ from the paper")
box(s, 0.6, 1.3, 12.15, 0.68, fill=INK)
text(s, 0.85, 1.33, 11.7, 0.62, [[("Rule: ", {"bold": True, "color": AMBER}),
     ("we follow the paper where it is explicit and the official code where the paper is silent. Our PyTorch Mamba "
      "scan matches the CUDA kernel to within 10⁻¹⁵, so it is not a source of difference.", {})]],
     size=13, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
diffs = [["Aspect", "Our version", "Official code"],
         ["Spatial convolutions", "3×3 and 5×5 (paper)", "3×3 only"],
         ["Band groups", "16 bands, stride 8 (paper)", "12 bands, stride 6"],
         ["Epochs", "100 (paper)", "150"],
         ["Mask size", "random per sample (paper)", "fixed 3×3 square"],
         ["Model mode when scoring", "evaluation", "training (masks on)"],
         ["State size; AUC checks; padding", "64; every epoch; zeros", "same (code)"]]
table(s, diffs, 0.6, 2.15, 5.85, [2.15, 2.05, 1.65], row_h=0.38, size=12, highlight_rows=[6])
hdr = ["Scene", "Paper", "Our run\n(state 64)", "State 16\n(3 seeds)", "Official\nsettings"]
rows, have, ranges = [hdr], {"main": [], "s16": [], "e1": [], "off": [], "paper": []}, {}
for d in EXTRA_DS:
    e1 = [v for v in X[d]["e1"] if v is not None]
    off = X[d]["official"]
    rows.append([DATASETS[d], f"{PAPER_AUC[d]['DMS2F-HAD']:.4f}",
                 f"{X[d]['main']:.4f}" if X[d]["main"] is not None else "running",
                 f"{np.mean(e1):.4f} ± {np.ptp(e1) / 2:.4f}" if len(e1) == 3 else "–",
                 f"{off:.4f}" if off is not None else "–"])
    have["paper"].append(PAPER_AUC[d]["DMS2F-HAD"])
    for k, v in [("main", X[d]["main"]), ("s16", X[d]["s16"]), ("off", off)]:
        if v is not None:
            have[k].append(v)
    if len(e1) == 3:
        have["e1"].append(np.mean(e1))
    ours_all = [v for v in [X[d]["main"], X[d]["s16"], off] + e1 if v is not None]
    ranges[d] = (min(ours_all), max(ours_all))
full = lambda k: len(have[k]) == len(EXTRA_DS)
rows.append(["Mean", f"{np.mean(have['paper']):.4f}", f"{np.mean(have['main']):.4f}" if full("main") else "–",
             f"{np.mean(have['e1']):.4f}" if full("e1") else "–", f"{np.mean(have['off']):.4f}" if full("off") else "–"])
table(s, rows, 6.65, 2.15, 6.1, [1.3, 0.95, 1.15, 1.6, 1.1], row_h=0.46, size=13, highlight_rows=[len(rows) - 1])
caption(s, "Best-epoch AUC on the AVIRIS scenes. 'State 16' checks the AUC every epoch; '±' is half the range over "
           "seeds 0, 1 and 2.", y=4.55, x=6.65, w=6.1)
reasons = [f"Checkpoint choice: the AUC swings by up to {swing:.2f} between checkpoints, and the paper reports the best "
           "one, so how often it is checked matters"
           + (f"; checking every epoch instead of every five raised our earlier mean from {np.mean(have['s16']):.4f} "
              f"to {np.mean(have['e1']):.4f}." if full("s16") and full("e1") else ".")]
if full("e1"):
    reasons.append(f"Randomness: with three seeds, the best AUC of one scene varies by up to {max(seed_spread):.3f}.")
if full("main") and full("e1") and full("off"):
    reasons.append(f"Settings: state size 64 gives a mean of {np.mean(have['main']):.4f}, state 16 gives "
                   f"{np.mean(have['e1']):.4f} and the full official settings {np.mean(have['off']):.4f}, against "
                   f"{np.mean(have['paper']):.4f} in the paper; these differences are within the seed variation."
                   if max(np.mean(have['main']), np.mean(have['e1']), np.mean(have['off']))
                   - min(np.mean(have['main']), np.mean(have['e1']), np.mean(have['off'])) <= max(seed_spread) else
                   f"Settings: state size 64 gives a mean of {np.mean(have['main']):.4f}, state 16 gives "
                   f"{np.mean(have['e1']):.4f} and the full official settings {np.mean(have['off']):.4f}, against "
                   f"{np.mean(have['paper']):.4f} in the paper.")
missed = [d for d in EXTRA_DS if ranges[d][1] < PAPER_AUC[d]["DMS2F-HAD"] - 0.005]
r4 = ("Not recorded: the released code cannot run as published (two files are missing and its Mamba library is "
      "Linux-only), and the settings and seed behind each paper number are not documented.")
for d in missed:
    r4 += (f" {DATASETS[d]} is not reproduced: all our runs give {ranges[d][0]:.4f}–{ranges[d][1]:.4f} against "
           f"{PAPER_AUC[d]['DMS2F-HAD']:.4f}.")
reasons.append(r4)
box(s, 0.6, 4.95, 12.15, 1.95, fill=PANEL)
text(s, 0.85, 5.0, 11.7, 1.88, reasons, size=13, bullets=True, space_after=2)

# ================================================================= 15. takeaways
s = new_slide(dark=True, notes=(
    "To sum up: what the method does, what we reproduced, the limits of this study, and what could be done next."))
title(s, "Takeaways", dark=True)
cols = [("What the method does", [
            "It learns the background of one image without labels, as a patch autoencoder.",
            "It models space and spectrum in two Mamba branches and mixes them per pixel with a gate.",
            f"It stays small (≈ {P_MAIN} parameters in our version) and fast thanks to linear-time Mamba."]),
        ("What we reproduced", [
            f"Our mean AUC is {avg_best:.4f}, against {avg_paper:.4f} in the paper on the same {len(GATED)} scenes.",
            "Our RX baseline matches the paper exactly.",
            "As in the paper, the spectral branch alone is the weakest variant.",
            f"Inference takes {avg_time:.2f} s per image on average."]),
        ("Limits of this study", [
            "We used 4 of the paper's 14 scenes, because only these are released.",
            "The best-epoch numbers use the ground truth to choose the checkpoint, as the paper does.",
            "The baseline numbers are the paper's; we only re-ran RX."]
            + [f"{DATASETS[d]} is not reproduced ({ranges[d][0]:.3f}–{ranges[d][1]:.3f} against "
               f"{PAPER_AUC[d]['DMS2F-HAD']:.3f})." for d in missed]),
        ("Next steps", [
            "Choose the checkpoint without labels, e.g. by early stopping on the reconstruction loss.",
            "Run the other ten scenes with several seeds.",
            "Use fused CUDA Mamba kernels on Linux to train faster."])]
for i, (h, pts) in enumerate(cols):
    x = 0.6 + i * 3.08
    box(s, x, 1.35, 2.9, 4.2, fill="1B2B45")
    badge(s, x + 0.2, 1.52, str(i + 1), d=0.48, size=15)
    text(s, x + 0.8, 1.54, 2.0, 0.6, h, size=17, font=HEAD, bold=True, color=WHITE)
    text(s, x + 0.15, 2.25, 2.65, 3.25, pts, size=14, color="DCE3EE", bullets=True, space_after=6)
box(s, 0.6, 5.75, 12.15, 1.15, fill=AMBER)
text(s, 0.85, 5.8, 11.7, 1.05, [[("Bottom line: ", {"bold": True}),
     (f"A dual-branch Mamba autoencoder with {P_MAIN} parameters "
      + ("matches" if avg_best >= avg_paper else "gets close to")
      + f" the paper's accuracy on the released scenes (mean AUC {avg_best:.4f} against {avg_paper:.4f})"
      + (f", although {' and '.join(DATASETS[d] for d in missed)} stays below it" if missed else "")
      + ". The open problem is how to choose the checkpoint without labels.", {})]],
     size=16, color=INK, anchor=MSO_ANCHOR.MIDDLE)

prs.save(OUT)
print("saved", OUT, "with", len(prs.slides), "slides")
