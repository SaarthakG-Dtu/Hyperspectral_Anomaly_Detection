"""
Make paper-style PNG figures and tables from results/runs/*.npz.

    python make_figures.py

Writes to results/figures/ :
  fig_datasets.png         false-colour image + ground truth of each dataset
  fig3_anomaly_maps.png    colour anomaly maps (paper Fig. 3)
  fig4_boxplots.png        anomaly vs background score distributions (paper Fig. 4)
  fig5_roc.png             ROC curves (paper Fig. 5)
  table2_auc.png           AUC + inference time vs. paper (paper Tables 2 & 3)
  table5_ablation.png      branch / fusion ablation (paper Table 5)
  fig_auc_vs_paper.png     our AUC vs. the paper's reported AUC
  fig_training_curves.png  loss and AUC during training
  <dataset>_result.png     one summary figure per dataset
plus results/auc_table.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import DATASETS, PAPER_AUC, auc_score, false_color, load_hsi, normalize, roc, rx_detector

ROOT = Path(__file__).parent
RUNS, FIGS = ROOT / "results" / "runs", ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
# fixed categorical order (one colour per method, never re-assigned)
METHODS = ["DMS2F-HAD", "RX", "Spatial only", "Spectral only", "Addition fusion"]
COLORS = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
FUSION_KEY = {"DMS2F-HAD": "gated", "Spatial only": "spatial", "Spectral only": "spectral", "Addition fusion": "add"}
ANOM, BACK = "#e34948", "#2a78d6"
INK, MUTED = "#0b0b0b", "#52514e"


# ----------------------------------------------------------------------------- load
def load_results():
    """res[dataset] = {'cube','gt','maps':{method: map}, 'runs':{fusion: npz}}  (maps use the
    checkpoint with the best AUC during training, as in the paper / official code)."""
    res = {}
    for d in DATASETS:
        cube, gt = load_hsi(d)
        r = {"cube": cube, "gt": gt, "maps": {"RX": rx_detector(cube)}, "runs": {}}
        for m, f in FUSION_KEY.items():
            p = RUNS / f"{d}_{f}.npz"
            if p.exists():
                run = dict(np.load(p))
                r["runs"][f] = run
                r["maps"][m] = run["best_map"]
        res[d] = r
    return res


def methods_present(res):
    """Methods that have results for at least one dataset (missing ones are shown as n/a)."""
    return [m for m in METHODS if any(m in r["maps"] for r in res.values())]


def clean(a):
    """Image panel: no ticks, no frame."""
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)


def save(fig, name):
    fig.savefig(FIGS / name, facecolor="white")
    plt.close(fig)
    print("saved", FIGS / name)


# ----------------------------------------------------------------------------- figures
def fig_datasets(res):
    fig, ax = plt.subplots(2, len(res), figsize=(2.6 * len(res), 5.4))
    for j, (d, r) in enumerate(res.items()):
        H, W, C = r["cube"].shape
        ax[0, j].imshow(false_color(r["cube"]))
        ax[0, j].set_title(f"{DATASETS[d]}\n{H}×{W}×{C}", fontsize=10)
        ax[1, j].imshow(r["gt"], cmap="gray")
        ax[1, j].set_title(f"Ground truth ({int(r['gt'].sum())} anomaly px)", fontsize=8, color=MUTED)
    for a in ax.ravel():
        clean(a)
    save(fig, "fig_datasets.png")


def fig3_anomaly_maps(res, wide=False):
    """Paper Fig. 3 layout. wide=True: vertical colour bar on the right (fits a 16:9 slide better)."""
    methods = ["RX", "Spectral only", "Spatial only", "Addition fusion", "DMS2F-HAD"]
    methods = [m for m in methods if m in methods_present(res)]
    cols = ["False colour"] + methods + ["Ground truth"]
    size = (1.75 * len(cols) + 0.9, 1.75 * len(res) + 0.45) if wide else (1.75 * len(cols), 1.8 * len(res) + 0.6)
    fig, ax = plt.subplots(len(res), len(cols), figsize=size)
    for i, (d, r) in enumerate(res.items()):
        for j, c in enumerate(cols):
            a = ax[i, j]
            if c == "False colour":
                a.imshow(false_color(r["cube"]))
            elif c == "Ground truth":
                a.imshow(r["gt"], cmap="gray")
            elif c not in r["maps"]:
                a.imshow(np.full(r["gt"].shape, 0.95), cmap="gray", vmin=0, vmax=1)
                a.text(0.5, 0.5, "not run", ha="center", va="center", fontsize=7, color=MUTED, transform=a.transAxes)
            else:
                im = a.imshow(normalize(r["maps"][c]), cmap="jet", vmin=0, vmax=1)
                a.text(0.03, 0.03, f"{auc_score(r['maps'][c], r['gt']):.4f}", transform=a.transAxes,
                       fontsize=6.5, color="white", bbox=dict(fc="black", alpha=0.55, lw=0, pad=1.2))
            clean(a)
            if i == 0:
                a.set_title(c, fontsize=8.5, fontweight="bold")
            if j == 0:
                a.set_ylabel(DATASETS[d], fontsize=9, fontweight="bold")
    if wide:
        fig.subplots_adjust(top=0.93, bottom=0.01, left=0.03, right=0.9, wspace=0.05, hspace=0.06)
        cax = fig.add_axes([0.92, 0.2, 0.013, 0.6])
        cb = fig.colorbar(im, cax=cax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
        cb.ax.tick_params(labelsize=7)
        fig.text(0.926, 0.83, "Anomaly\npresent", ha="center", va="bottom", fontsize=8)
        fig.text(0.926, 0.17, "Anomaly\nabsent", ha="center", va="top", fontsize=8)
        save(fig, "fig3_anomaly_maps_wide.png")
        return
    fig.subplots_adjust(top=0.86, bottom=0.03, wspace=0.05, hspace=0.08)
    cax = fig.add_axes([0.35, 0.94, 0.3, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    cb.ax.tick_params(labelsize=7)
    fig.text(0.34, 0.947, "Anomaly absent", ha="right", va="center", fontsize=8)
    fig.text(0.66, 0.947, "Anomaly present", ha="left", va="center", fontsize=8)
    fig.text(0.5, -0.01, "Numbers in the corner = AUC.  Maps min-max normalised to [0, 1].",
             ha="center", fontsize=7, color=MUTED)
    save(fig, "fig3_anomaly_maps.png")


def fig4_boxplots(res):
    methods = [m for m in ["RX", "Spectral only", "Spatial only", "Addition fusion", "DMS2F-HAD"]
               if m in methods_present(res)]
    fig, axs = plt.subplots(1, len(res), figsize=(3.6 * len(res), 4.8), sharey=True)
    for a, (d, r) in zip(axs, res.items()):
        gt = r["gt"].ravel().astype(bool)
        for k, m in enumerate(methods):
            if m not in r["maps"]:
                a.text(k, 0.5, "not run", ha="center", fontsize=6.5, color=MUTED, rotation=90)
                continue
            s = normalize(r["maps"][m]).ravel()
            for off, data, col in [(-0.18, s[gt], ANOM), (0.18, s[~gt], BACK)]:
                bp = a.boxplot(data, positions=[k + off], widths=0.3, whis=(5, 95), showfliers=False,
                               patch_artist=True, medianprops=dict(color=INK, lw=1),
                               whiskerprops=dict(color=col, ls="--", lw=1), capprops=dict(color=col, lw=1))
                bp["boxes"][0].set(facecolor=col, alpha=0.75, edgecolor=col)
        a.set_xticks(range(len(methods)))
        short = {"Spectral only": "Spectral\nonly", "Spatial only": "Spatial\nonly", "Addition fusion": "Addition\nfusion",
                 "DMS2F-HAD": "DMS2F-\nHAD"}
        a.set_xticklabels([short.get(m, m) for m in methods], fontsize=6.5)
        a.set_title(DATASETS[d], fontweight="bold")
        a.set_ylim(-0.02, 1.02)
        a.grid(axis="y", alpha=0.3)
    axs[0].set_ylabel("Normalised anomaly score")
    axs[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=ANOM, alpha=0.75),
                           plt.Rectangle((0, 0), 1, 1, fc=BACK, alpha=0.75)],
                  labels=["Anomaly", "Background"], loc="upper left", fontsize=7, frameon=False)
    fig.text(0.5, -0.02, "Boxes = 25-75 %, whiskers = 5-95 % of pixel scores.", ha="center", fontsize=7, color=MUTED)
    save(fig, "fig4_boxplots.png")


def fig5_roc(res, grid=False):
    """Paper Fig. 5. grid=True: 2 x 2 layout (used on the slide)."""
    methods = methods_present(res)
    if grid:
        fig, axs = plt.subplots(2, 2, figsize=(7.6, 7.0), sharex=True, sharey=True)
        axs = axs.ravel()
    else:
        fig, axs = plt.subplots(1, len(res), figsize=(3.4 * len(res), 3.8), sharey=True)
    for a, (d, r) in zip(axs, res.items()):
        for m in reversed([m for m in methods if m in r["maps"]]):   # draw ours last (on top)
            fpr, tpr = roc(r["maps"][m], r["gt"])
            a.plot(fpr, tpr, color=COLORS[m], lw=2.2 if m == "DMS2F-HAD" else 1.4,
                   label=f"{m} ({auc_score(r['maps'][m], r['gt']):.4f})")
        a.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
        a.set_title(DATASETS[d], fontweight="bold")
        a.grid(alpha=0.3)
        h, l = a.get_legend_handles_labels()
        a.legend(h[::-1], l[::-1], loc="lower right", fontsize=6.5, frameon=False, title="Method (AUC)",
                 title_fontsize=6.5)
    if grid:
        for a in axs[2:]:
            a.set_xlabel("False positive rate")
        for a in axs[::2]:
            a.set_ylabel("True positive rate")
        fig.tight_layout()
        save(fig, "fig5_roc_grid.png")
        return
    for a in axs:
        a.set_xlabel("False positive rate")
    axs[0].set_ylabel("True positive rate")
    save(fig, "fig5_roc.png")


def draw_table(rows, header, name, title, col_w=None, note=None, bold_best=None):
    n = len(rows) + 2                                    # header counts as two rows
    fig, ax = plt.subplots(figsize=(sum(col_w) if col_w else 1.6 * len(header), 0.32 * n + 0.5))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, cellLoc="center", bbox=[0, 0, 1, 1],
                 colWidths=[w / sum(col_w) for w in col_w] if col_w else None)
    t.auto_set_font_size(False); t.set_fontsize(8.5)
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        cell.set_height(2 / n if i == 0 else 1 / n)
        if i == 0:
            cell.set_facecolor("#1f3b57"); cell.set_text_props(color="white", fontweight="bold")
        elif rows[i - 1][0] in ("Average", "Mean"):
            cell.set_facecolor("#eef2f6"); cell.set_text_props(fontweight="bold")
        if bold_best and i > 0 and (i - 1, j) in bold_best:
            cell.set_text_props(fontweight="bold", color="#1f6f3f")
    ax.set_title(title, fontweight="bold", fontsize=10, pad=8)
    if note:
        ax.text(0.5, -0.04, note, ha="center", va="top", fontsize=7, color=MUTED, transform=ax.transAxes)
    save(fig, name)


def tables(res):
    # ---- Table 2 + 3 style: AUC and inference time
    rows, csv = [], ["dataset,rx_auc,rx_auc_paper,dms2f_best_auc,dms2f_final_auc,best_epoch,dms2f_auc_paper,"
                     "infer_time_s,params"]
    vals = {k: [] for k in ["rx", "rxp", "best", "final", "paper", "time"]}
    for d, r in res.items():
        run = r["runs"].get("gated")
        rx = auc_score(r["maps"]["RX"], r["gt"])
        vals["rx"].append(rx); vals["rxp"].append(PAPER_AUC[d]["RX"]); vals["paper"].append(PAPER_AUC[d]["DMS2F-HAD"])
        if run is None:
            rows.append([DATASETS[d], f"{rx:.4f}", f"{PAPER_AUC[d]['RX']:.4f}", "-", "-", f"{PAPER_AUC[d]['DMS2F-HAD']:.4f}", "-"])
            continue
        b, f, t = float(run["best_auc"]), float(run["final_auc"]), float(run["infer_time"])
        vals["best"].append(b); vals["final"].append(f); vals["time"].append(t)
        rows.append([DATASETS[d], f"{rx:.4f}", f"{PAPER_AUC[d]['RX']:.4f}", f"{b:.4f} (ep {int(run['best_epoch'])})",
                     f"{f:.4f}", f"{PAPER_AUC[d]['DMS2F-HAD']:.4f}", f"{t:.3f}"])
        csv.append(f"{d},{rx:.4f},{PAPER_AUC[d]['RX']},{b:.4f},{f:.4f},{int(run['best_epoch'])},"
                   f"{PAPER_AUC[d]['DMS2F-HAD']},{t:.3f},{int(run['params'])}")
    if len(vals["best"]) == len(res):
        rows.append(["Average", f"{np.mean(vals['rx']):.4f}", f"{np.mean(vals['rxp']):.4f}", f"{np.mean(vals['best']):.4f}",
                     f"{np.mean(vals['final']):.4f}", f"{np.mean(vals['paper']):.4f}", f"{np.mean(vals['time']):.3f}"])
    draw_table(rows, ["Dataset", "RX\n(ours)", "RX\n(paper)", "DMS2F-HAD\nbest epoch (ours)", "DMS2F-HAD\nlast epoch (ours)",
                      "DMS2F-HAD\n(paper)", "Inference\ntime (s)"], "table2_auc.png",
               "AUC of anomaly detection: our run vs. paper", col_w=[1.2, 0.85, 0.85, 1.75, 1.75, 1.2, 1.0],
               note="'Best epoch' = checkpoint with highest AUC during training (paper / official-code protocol); "
                    "'last epoch' = model after 100 epochs.")
    (ROOT / "results" / "auc_table.csv").write_text("\n".join(csv) + "\n")

    # ---- Table 5 style: ablation
    order = [("spectral", "Spectral only"), ("spatial", "Spatial only"), ("add", "Addition fusion"), ("gated", "Gated fusion")]
    have = [(k, n) for k, n in order if any(k in r["runs"] for r in res.values())]
    res = {d: r for d, r in res.items() if all(k in r["runs"] for k, _ in have)}   # datasets with every variant
    if len(have) < 2 or not res:
        return
    rows, best_cells, means = [], [], {k: [] for k, _ in have}
    for i, (d, r) in enumerate(res.items()):
        aucs = [float(r["runs"][k]["best_auc"]) for k, _ in have]
        paper = [PAPER_AUC[d][k if k != "gated" else "DMS2F-HAD"] for k, _ in have]
        for (k, _), a in zip(have, aucs):
            means[k].append(a)
        rows.append([DATASETS[d]] + [f"{a:.4f}  ({p:.4f})" for a, p in zip(aucs, paper)])
        best_cells.append((i, 1 + int(np.argmax(aucs))))
    pm = [np.mean([PAPER_AUC[d][k if k != "gated" else "DMS2F-HAD"] for d in res]) for k, _ in have]
    rows.append(["Average"] + [f"{np.mean(means[k]):.4f}  ({p:.4f})" for (k, _), p in zip(have, pm)])
    draw_table(rows, ["Dataset"] + [n for _, n in have], "table5_ablation.png",
               "Ablation: branches and fusion  —  ours (paper)", col_w=[1.2] + [1.7] * len(have),
               note="Each cell: our AUC (paper's reported AUC in brackets). Green = best of our variants per dataset.",
               bold_best=best_cells)


def fig_auc_vs_paper(res):
    ds = [d for d in res if "gated" in res[d]["runs"]]
    if not ds:
        return
    fig, ax = plt.subplots(figsize=(6.2, 0.55 * len(ds) + 1.2))
    for i, d in enumerate(ds):
        pts = [("RX (paper)", PAPER_AUC[d]["RX"], COLORS["RX"], "o", False),
               ("RX (ours)", auc_score(res[d]["maps"]["RX"], res[d]["gt"]), COLORS["RX"], "o", True),
               ("DMS2F-HAD (paper)", PAPER_AUC[d]["DMS2F-HAD"], COLORS["DMS2F-HAD"], "D", False),
               ("DMS2F-HAD (ours)", float(res[d]["runs"]["gated"]["best_auc"]), COLORS["DMS2F-HAD"], "D", True)]
        for lab, v, c, mk, filled in pts:
            ax.scatter(v, i + (0.12 if "ours" in lab else -0.12), s=55, marker=mk, color=c if filled else "white",
                       edgecolor=c, lw=1.6, zorder=3, label=lab if i == 0 else None)
    ax.set_yticks(range(len(ds))); ax.set_yticklabels([DATASETS[d] for d in ds]); ax.invert_yaxis()
    ax.set_xlabel("AUC (higher is better)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=7, frameon=False)
    ax.set_title("Reproduced AUC vs. reported in the paper", fontweight="bold")
    save(fig, "fig_auc_vs_paper.png")


def fig_training_curves(res):
    ds = [d for d in res if "gated" in res[d]["runs"]]
    if not ds:
        return
    fig, axs = plt.subplots(2, len(ds), figsize=(3.2 * len(ds), 5), sharex=True)
    axs = np.array(axs).reshape(2, -1)
    for j, d in enumerate(ds):
        run = res[d]["runs"]["gated"]
        axs[0, j].plot(np.arange(1, len(run["loss_hist"]) + 1), run["loss_hist"], color=COLORS["DMS2F-HAD"], lw=1.6)
        axs[0, j].set_yscale("log"); axs[0, j].set_title(DATASETS[d], fontweight="bold")
        h = run["auc_hist"]
        axs[1, j].plot(h[:, 0], h[:, 1], color=COLORS["DMS2F-HAD"], lw=1.6, marker="o", ms=3)
        axs[1, j].axhline(PAPER_AUC[d]["DMS2F-HAD"], color=MUTED, ls="--", lw=1)
        axs[1, j].text(100, PAPER_AUC[d]["DMS2F-HAD"], "paper ", va="bottom", ha="right", fontsize=7, color=MUTED)
        axs[1, j].set_xlabel("Epoch")
        for a in axs[:, j]:
            a.grid(alpha=0.3)
    axs[0, 0].set_ylabel("Training loss (MSE + 0.1·L1)")
    axs[1, 0].set_ylabel("AUC")
    save(fig, "fig_training_curves.png")


def per_dataset(res):
    for d, r in res.items():
        if "DMS2F-HAD" not in r["maps"]:
            continue
        fig, ax = plt.subplots(1, 5, figsize=(15, 3.2), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.15]})
        ax[0].imshow(false_color(r["cube"])); ax[0].set_title("False-colour HSI")
        ax[1].imshow(r["gt"], cmap="gray"); ax[1].set_title("Ground truth")
        for a, m in [(ax[2], "RX"), (ax[3], "DMS2F-HAD")]:
            a.imshow(normalize(r["maps"][m]), cmap="jet", vmin=0, vmax=1)
            a.set_title(f"{m}  (AUC {auc_score(r['maps'][m], r['gt']):.4f})")
        for a in ax[:4]:
            clean(a)
        for m in ["RX", "DMS2F-HAD"]:
            fpr, tpr = roc(r["maps"][m], r["gt"])
            ax[4].plot(fpr, tpr, color=COLORS[m], lw=2, label=m)
        ax[4].plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
        ax[4].set_xlabel("False positive rate"); ax[4].set_ylabel("True positive rate")
        ax[4].legend(frameon=False, loc="lower right"); ax[4].grid(alpha=0.3); ax[4].set_title("ROC")
        fig.subplots_adjust(wspace=0.32)
        fig.suptitle(DATASETS[d], fontweight="bold", y=1.02)
        save(fig, f"{d}_result.png")


def fig_hsi_cube(d="San_Diego", out=None, dark=False):
    """Illustration of a hyperspectral cube: image on top, spectra along the sides."""
    cube, _ = load_hsi(d)
    H, W, C = cube.shape
    ink = "white" if dark else INK
    fig = plt.figure(figsize=(5, 4.4))
    ax = fig.add_subplot(projection="3d")
    zc = C / 2.2                                                  # height of the band axis
    X, Y = np.meshgrid(np.arange(W + 1), np.arange(H + 1))
    ax.plot_surface(X, Y, np.full_like(X, zc, dtype=float), facecolors=false_color(cube)[::-1],
                    rstride=1, cstride=1, shade=False, linewidth=0)
    cmap = plt.get_cmap("turbo")
    xs, zs = np.meshgrid(np.arange(W + 1), np.linspace(0, zc, C + 1))
    ax.plot_surface(xs, np.zeros_like(xs, dtype=float), zs, facecolors=cmap(normalize(cube[-1, :, ::-1].T)),
                    rstride=1, cstride=1, shade=False, linewidth=0)
    ys, zs = np.meshgrid(np.arange(H + 1), np.linspace(0, zc, C + 1))
    ax.plot_surface(np.full_like(ys, W, dtype=float), ys, zs,
                    facecolors=cmap(normalize(cube[::-1, -1, ::-1].T)), rstride=1, cstride=1, shade=False, linewidth=0)
    ax.set_box_aspect((W, H, zc)); ax.view_init(elev=24, azim=-58); ax.set_axis_off()
    ax.text(W / 2, -18, 0, "W (x)", fontsize=9, ha="center", color=ink)
    ax.text(W + 14, H / 2, 0, "H (y)", fontsize=9, ha="center", color=ink)
    fig.text(0.21, 0.45, f"C = {C}\nbands", fontsize=9, ha="right", color=ink)
    if out:                                            # transparent version for dark slides
        fig.savefig(out, transparent=True)
        plt.close(fig)
        print("saved", out)
    else:
        save(fig, "fig_hsi_cube.png")


def fig_spectra(d="San_Diego"):
    """Background vs anomaly spectral signatures: why anomalies stand out."""
    cube, gt = load_hsi(d)
    X = cube.reshape(-1, cube.shape[2]); g = gt.ravel().astype(bool)
    b = np.arange(1, X.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    lo, hi = np.percentile(X[~g], [10, 90], axis=0)
    ax.fill_between(b, lo, hi, color=BACK, alpha=0.18, lw=0, label="Background (10-90 %)")
    ax.plot(b, X[~g].mean(0), color=BACK, lw=2, label="Background mean")
    for k, i in enumerate(np.random.default_rng(0).choice(np.where(g)[0], 6, replace=False)):
        ax.plot(b, X[i], color=ANOM, lw=0.9, alpha=0.6, label="Anomaly pixels (planes)" if k == 0 else None)
    ax.plot(b, X[g].mean(0), color=ANOM, lw=2, label="Anomaly mean")
    ax.set_xlabel("Spectral band index"); ax.set_ylabel("Normalised reflectance")
    ax.set_title(f"{DATASETS[d]}: spectral signatures", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.grid(alpha=0.3)
    save(fig, "fig_spectra.png")


def update_readme(res):
    """Rewrite the block between the RESULTS markers in README.md with the current numbers."""
    readme = ROOT / "README.md"
    start, end = "<!-- RESULTS START -->", "<!-- RESULTS END -->"
    txt = readme.read_text(encoding="utf-8")
    if start not in txt:
        return
    L = ["| Scene | RX (ours / paper) | DMS2F-HAD ours, best epoch | DMS2F-HAD ours, epoch 100 | DMS2F-HAD paper | Inference (s) |",
         "|---|---|---|---|---|---|"]
    best, final, paper, rx = [], [], [], []
    for d, r in res.items():
        a_rx = auc_score(r["maps"]["RX"], r["gt"])
        g = r["runs"].get("gated")
        p = PAPER_AUC[d]["DMS2F-HAD"]
        if g is None:
            L.append(f"| {DATASETS[d]} | {a_rx:.4f} / {PAPER_AUC[d]['RX']:.4f} | *training* | *training* | {p:.4f} | – |")
            continue
        best.append(float(g["best_auc"])); final.append(float(g["final_auc"])); paper.append(p); rx.append(a_rx)
        L.append(f"| {DATASETS[d]} | {a_rx:.4f} / {PAPER_AUC[d]['RX']:.4f} | **{float(g['best_auc']):.4f}** "
                 f"(epoch {int(g['best_epoch'])}) | {float(g['final_auc']):.4f} | {p:.4f} | {float(g['infer_time']):.3f} |")
    if best:
        L.append(f"| **Average ({len(best)} scenes)** | {np.mean(rx):.4f} | **{np.mean(best):.4f}** | {np.mean(final):.4f} "
                 f"| {np.mean(paper):.4f} | |")
    order = [("spectral", "Spectral only"), ("spatial", "Spatial only"), ("add", "Addition fusion"), ("gated", "Gated fusion (full)")]
    rows = [d for d, r in res.items() if all(k in r["runs"] for k, _ in order)]
    if rows:
        L += ["", "**Ablation** (paper Table 5) — our AUC at the best epoch, paper value in brackets:", "",
              "| Scene | " + " | ".join(n for _, n in order) + " |", "|---|" + "---|" * len(order)]
        for d in rows:
            a = [float(res[d]["runs"][k]["best_auc"]) for k, _ in order]
            cells = [(f"**{x:.4f}**" if x == max(a) else f"{x:.4f}") +
                     f" ({PAPER_AUC[d][k if k != 'gated' else 'DMS2F-HAD']:.4f})" for x, (k, _) in zip(a, order)]
            L.append(f"| {DATASETS[d]} | " + " | ".join(cells) + " |")
        means = [np.mean([float(res[d]["runs"][k]["best_auc"]) for d in rows]) for k, _ in order]
        pmeans = [np.mean([PAPER_AUC[d][k if k != "gated" else "DMS2F-HAD"] for d in rows]) for k, _ in order]
        L.append("| **Average** | " + " | ".join((f"**{x:.4f}**" if x == max(means) else f"{x:.4f}") + f" ({p:.4f})"
                                              for x, p in zip(means, pmeans)) + " |")
        spread = max(means[1:]) - min(means[1:])
        L += ["", f"Reading: the spectral branch alone is clearly the weakest variant, as in the paper. Spatial-only, "
                  f"addition and gated fusion are within {spread:.3f} AUC of each other on average, so with one seed per "
                  f"run we could not confirm the paper's gain from the gate on these scenes."]
    # ---- test runs that explain the gap to the paper (results/runs/extra)
    def extra_auc(d, tag):
        p = RUNS / "extra" / f"{d}_gated_{tag}.npz"
        return float(np.load(p)["best_auc"]) if p.exists() else None
    eds = ["aviris_1", "aviris_2", "San_Diego"]
    if (RUNS / "extra").exists():
        L += ["", "**Why our numbers differ from the paper.** Best-epoch AUC of extra test runs on the three AVIRIS "
                  "scenes: the main run, the same settings with the AUC checked every epoch for seeds 0, 1 and 2, and "
                  "the official code's settings (`--official`).", "",
              "| Scene | Paper | Our main run | AUC checked every epoch, seeds 0 / 1 / 2 | Official settings |",
              "|---|---|---|---|---|"]
        cols = {"paper": [], "main": [], "e1": [], "off": []}
        spreads = []
        for d in eds:
            e1 = [extra_auc(d, f"every1_s{k}") for k in range(3)]
            off = extra_auc(d, "official")
            main = float(res[d]["runs"]["gated"]["best_auc"])
            e1s = " / ".join(f"{v:.4f}" if v is not None else "running" for v in e1)
            offs = f"{off:.4f}" if off is not None else "running"
            L.append(f"| {DATASETS[d]} | {PAPER_AUC[d]['DMS2F-HAD']:.4f} | {main:.4f} | {e1s} | {offs} |")
            cols["paper"].append(PAPER_AUC[d]["DMS2F-HAD"]); cols["main"].append(main)
            if all(v is not None for v in e1):
                cols["e1"].append(np.mean(e1)); spreads.append(max(e1) - min(e1))
            if off is not None:
                cols["off"].append(off)
        n = len(eds)
        L.append(f"| **Mean** | {np.mean(cols['paper']):.4f} | {np.mean(cols['main']):.4f} | "
                 + (f"{np.mean(cols['e1']):.4f} (mean of seeds)" if len(cols["e1"]) == n else "–")
                 + (f" | {np.mean(cols['off']):.4f} |" if len(cols["off"]) == n else " | – |"))
        if len(cols["e1"]) == n:
            L += ["", f"* Across three seeds, the best AUC of one scene varies by up to {max(spreads):.3f}, so "
                      f"differences of this size between single runs are run-to-run noise.",
                  f"* Checking the AUC every epoch instead of every five epochs changes the mean of seed 0 from "
                  f"{np.mean(cols['main']):.4f} to {np.mean([extra_auc(d, 'every1_s0') for d in eds]):.4f}."]
        if len(cols["off"]) == n:
            gap = np.mean(cols["off"]) - np.mean(cols["main"])
            verdict = ("so these settings explain part of the gap" if gap > 0.003 else
                       "so these settings do not explain the gap" if gap > -0.003 else
                       "so these settings do not help in our runs")
            L.append(f"* With the official code's settings the mean is {np.mean(cols['off']):.4f}, against "
                     f"{np.mean(cols['paper']):.4f} in the paper and {np.mean(cols['main']):.4f} for our main run, "
                     f"{verdict}.")
    missing = [f"{DATASETS[d]} / {n}" for d, r in res.items() for k, n in order if k not in r["runs"]]
    if missing:
        L += ["", f"*Not finished or not run yet:* {', '.join(missing)}."]
    new = start + "\n" + "\n".join(L) + "\n" + end
    readme.write_text(txt[:txt.index(start)] + new + txt[txt.index(end) + len(end):], encoding="utf-8")
    print("updated", readme)


if __name__ == "__main__":
    res = load_results()
    update_readme(res)
    fig_hsi_cube()
    fig_hsi_cube(out=ROOT / "presentation" / "assets" / "hsi_cube_dark.png", dark=True)
    fig_spectra()
    fig_datasets(res)
    fig3_anomaly_maps(res)
    fig3_anomaly_maps(res, wide=True)
    fig4_boxplots(res)
    fig5_roc(res)
    fig5_roc(res, grid=True)
    tables(res)
    fig_auc_vs_paper(res)
    fig_training_curves(res)
    per_dataset(res)
