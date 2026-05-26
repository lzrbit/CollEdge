#!/usr/bin/env python3
"""
Generate two additional result figures for the JSTSP submission.

  fig5_bubble_matrix.{pdf,png}
      DCFCL-style "client x task" bubble matrix on EMNIST-Letters.
      Two columns of subplots compare the per-client per-task accuracy
      before and after the catastrophic-forgetting stream:
          (a) "After-task" accuracy: the value measured immediately after
              the algorithm finished training task t (per-client diagonal of
              `all_accs`, exposed in results.json as `per_client_diag_acc`).
          (b) "End-of-stream" accuracy: the value measured after the
              continual stream + directed coalition aggregation has run for
              the full T tasks (`per_client_final_acc`).
      Bubble area + colour both encode accuracy in [0, 100].

  fig6_radar_metrics.{pdf,png}
      Five-axis radar chart comparing DynDFCL (gradient mode, HP-tuned)
      against representative baselines on:
          1. EMNIST-Letters average accuracy
          2. EMNIST-Letters anti-forgetting (1 - forget)
          3. CIFAR-100 average accuracy
          4. CIFAR-100 anti-forgetting
          5. Collective intelligence emergence (mean accuracy on previously
             seen tasks at end-of-stream, i.e. mean(per_task_acc[-1][:-1]))

Run from the repo root:
    python paper/generate_extra_figures.py
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "paper_experiments"
FIG_DIR = REPO / "paper" / "figures"
DATA_DIR = REPO / "paper" / "figure_data"

FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Unified font sizes (kept in sync with plot_figures.py)
# All-9pt across the bubble-matrix figure per user request (Fig 7 font
# consistency); the radar figure uses its own local font overrides in
# plot_radar() and is unaffected.
FONT_BASE       = 9.0
FONT_TITLE      = 9.0
FONT_AXIS_LABEL = 9.0
FONT_TICK       = 9.0
FONT_LEGEND     = 9.0

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":        FONT_BASE,
    "axes.titlesize":   FONT_TITLE,
    "axes.labelsize":   FONT_AXIS_LABEL,
    "xtick.labelsize":  FONT_TICK,
    "ytick.labelsize":  FONT_TICK,
    "legend.fontsize":  FONT_LEGEND,
    "legend.title_fontsize": FONT_LEGEND,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest(pattern: str) -> dict | None:
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        return None
    return json.load(open(files[-1]))


def _avg_acc(d: dict) -> float:
    a = d.get("all_accuracies", [])
    return 100.0 * (sum(a) / len(a)) if a else 0.0


def _anti_forget(d: dict) -> float:
    return 100.0 - 100.0 * float(d.get("forgetting_rate", 0.0))


def _past_task_retention(d: dict) -> float:
    """Mean accuracy on previously seen tasks at the end of the stream.

    `per_task_acc[-1]` is the final phase's per-task accuracy vector with
    length T; the leading T-1 entries are old tasks, the trailing entry is
    the just-trained current task.  We average over the leading T-1 entries
    to capture how much of the past collective knowledge is retained.
    """
    pta = d.get("per_task_acc")
    if not pta or len(pta[-1]) < 2:
        return 0.0
    past = pta[-1][:-1]
    return 100.0 * sum(past) / len(past)


# ---------------------------------------------------------------------------
# fig5: client x task bubble matrix
# ---------------------------------------------------------------------------
# Pick a small set of representative algorithms (those for which we have
# per-client per-task data captured by the supplementary EMNIST run).
PERCLIENT_BASE = RES / "perclient" / "EMNIST-Letters"

# (raw_folder_on_disk, display_name).  Our method's raw experiment folder
# is still named "CollEdge" historically; display label is DynDFCL.
PERCLIENT_ALGORITHMS = [
    ("FedAvg",    "FedAvg"),
    ("FedProx",   "FedProx"),
    ("DCFCL",     "DCFCL"),
    ("CollEdge",  "DynDFCL"),
]


def _load_perclient(folder: str) -> dict | None:
    """Load the most recent run with per-client matrices."""
    pattern = str(PERCLIENT_BASE / folder / "**" / "results.json")
    files = sorted(glob.glob(pattern, recursive=True))
    for f in reversed(files):
        d = json.load(open(f))
        if d.get("per_client_diag_acc") and d.get("per_client_final_acc"):
            return d
    return None


def _client_task_matrix(d: dict, key: str) -> np.ndarray:
    """Return an (N_clients, T_tasks) matrix in percentage.

    Rows are sorted by integer client id; columns are tasks 0..T-1.
    Missing cells are filled with 0.
    """
    m = d.get(key, {})
    if not m:
        return np.zeros((0, 0))
    cids = sorted(m.keys(), key=lambda x: int(x))
    n_tasks = max(len(m[cid]) for cid in cids)
    out = np.zeros((len(cids), n_tasks))
    for i, cid in enumerate(cids):
        row = m[cid]
        for t in range(min(n_tasks, len(row))):
            out[i, t] = 100.0 * float(row[t])
    return out


def plot_bubble_matrix(panels: list[tuple[str, str, np.ndarray]], out_pdf: Path):
    """panels: list of (panel_label, algorithm_label, mat[N x T]) triples.

    Single-row 1xN layout (after-task accuracy only).  The end-of-stream
    row has been removed by request to keep the figure compact.
    """
    n_alg = len(PERCLIENT_ALGORITHMS)
    fig, axes = plt.subplots(
        nrows=1, ncols=n_alg, figsize=(2.85 * n_alg, 2.9),
        sharex=True, sharey=True,
    )
    if n_alg == 1:
        axes = np.array([axes])

    cmap = plt.get_cmap("viridis")
    # Compute vmin from the data so the colourbar isn't dominated by the
    # 0..min(values) range — gives more visible separation among the high
    # accuracies typical of after-task readings.  Rounded down to the next
    # multiple of 5 and clamped to [40, 70] for stability across runs.
    if panels:
        data_min = min(float(mat.min()) for _, _, mat in panels)
        vmin = max(40.0, min(70.0, np.floor(data_min / 5.0) * 5.0))
    else:
        vmin = 50.0
    # Non-linear mapping: most after-task accuracies sit in [90, 100], so a
    # linear colour/size scale leaves them indistinguishable.  PowerNorm
    # with gamma>1 stretches the high end of the dynamic range, and we
    # apply the same gamma to the bubble-area formula below for visual
    # consistency between colour and size.
    GAMMA = 3.0
    norm = PowerNorm(gamma=GAMMA, vmin=vmin, vmax=100)
    vrange = max(1e-9, 100.0 - vmin)

    for idx, (row_label, alg_label, mat) in enumerate(panels):
        ax = axes[idx]
        N, T = mat.shape
        xs, ys, vs = [], [], []
        for i in range(N):
            for t in range(T):
                xs.append(i)
                ys.append(t)
                vs.append(mat[i, t])
        # Non-linear bubble area sharing the colour-norm's gamma so that
        # equal-colour cells also have equal-area bubbles.  Values below
        # vmin are clamped to the minimum visible size.
        sizes = []
        for v in vs:
            t = min(1.0, max(0.0, (v - vmin) / vrange))
            sizes.append(10 + 160 * (t ** GAMMA))
        sc = ax.scatter(xs, ys, s=sizes, c=vs, cmap=cmap, norm=norm,
                        edgecolors="black", linewidths=0.35, alpha=0.92)
        ax.set_xticks(range(N))
        ax.set_xticklabels([f"C{i+1}" for i in range(N)], fontsize=FONT_TICK)
        ax.set_yticks(range(T))
        ax.set_yticklabels([f"T{t+1}" for t in range(T)], fontsize=FONT_TICK)
        ax.set_xlim(-0.7, N - 0.3)
        ax.set_ylim(-0.7, T - 0.3)
        ax.invert_yaxis()  # T1 on top
        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.55)
        ax.set_axisbelow(True)
        ax.set_title(alg_label, fontsize=FONT_TITLE, pad=4)
        ax.set_xlabel("Client", fontsize=FONT_AXIS_LABEL)
        if idx == 0:
            ax.set_ylabel(row_label, fontsize=FONT_AXIS_LABEL)

    # Explicit ticks for the non-linear scale: denser near the top so the
    # 90-100 region (where most after-task values live) gets visible labels.
    cb_ticks = sorted({vmin, 80.0, 90.0, 95.0, 98.0, 100.0})
    cb_ticks = [t for t in cb_ticks if vmin <= t <= 100.0]
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, pad=0.025,
                        location="right", fraction=0.022, ticks=cb_ticks)
    cbar.set_label("Per-client per-task accuracy (%)", fontsize=FONT_AXIS_LABEL)
    cbar.ax.tick_params(labelsize=FONT_TICK)

    # Suptitle removed by request; the LaTeX figure caption carries the
    # equivalent description in main.tex.
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[v] {out_pdf.name}")
    print(f"[v] {out_pdf.with_suffix('.png').name}")


# ---------------------------------------------------------------------------
# fig6: 5-axis radar
# ---------------------------------------------------------------------------
# Fixed numbers below come from the consolidated tables in main.tex
# (Tab.~\ref{tab:main_results} EMNIST avg-acc / forget; CIFAR avg-acc / forget)
# plus past_tasks_avg_at_final from paper/figure_data/fig4_emergence.json
# (computed in the analyze_and_update.py pipeline).  Kept inline so this
# script is self-contained and reproducible without re-running the entire
# experimental pipeline.

# Values transcribed directly from the body tables in main.tex:
#   eacc / eforget / cacc / cforget — Table tab:main_results (DynDFCL uses
#   the "gradient" headline row).
#   emerge — mean of T1..T_{T-1} (EMNIST T1-T5 from tab:task_wise_accuracy),
#   which is the "collective intelligence emergence" axis defined in the
#   caption.
RADAR_DATA = {
    "FedAvg":   dict(eacc=50.35, eforget=57.49, cacc=10.21, cforget=12.10, emerge=19.34),
    "FedProx":  dict(eacc=53.84, eforget=54.12, cacc=10.79, cforget=12.98, emerge=20.50),
    "FedLwF":   dict(eacc=50.33, eforget=57.33, cacc=10.56, cforget=12.46, emerge=18.44),
    "SCAFFOLD": dict(eacc=42.48, eforget=60.74, cacc=8.26,  cforget=7.66,  emerge=8.14),
    "DCFCL":    dict(eacc=46.75, eforget=65.32, cacc=12.12, cforget=24.01, emerge=7.50),
    "DynDFCL":  dict(eacc=93.72, eforget=9.49,  cacc=24.98, cforget=41.28, emerge=86.86),
}

RADAR_AXES = [
    ("EMNIST\nAvg. Acc.",        "eacc",    False),
    ("EMNIST\nAnti-Forget.",     "eforget", True),  # smaller forget = better
    ("CIFAR-100\nAvg. Acc.",     "cacc",    False),
    ("CIFAR-100\nAnti-Forget.",  "cforget", True),
    ("Collective\nEmergence",    "emerge",  False),
]


def _radar_normalised() -> tuple[np.ndarray, list[str]]:
    algs = list(RADAR_DATA.keys())
    M = len(RADAR_AXES)
    A = len(algs)
    out = np.zeros((A, M))
    for j, (_, key, invert) in enumerate(RADAR_AXES):
        col = np.array([RADAR_DATA[a][key] for a in algs])
        if invert:
            # Convert "forgetting %" to "anti-forgetting %": 100 - forget.
            col = 100.0 - col
            col = np.clip(col, 0.0, 100.0)
        # Min-max scale within each axis so that "outer is better" for ALL
        col_max = col.max() if col.max() > 0 else 1.0
        out[:, j] = col / col_max
    return out, algs


def plot_radar(out_pdf: Path):
    # Local font overrides for Fig 7 only (do not change FONT_* globals,
    # which are shared with Fig 6's bubble matrix).
    fs_tick   = 14.0
    fs_axis   = 15.0
    fs_title  = 17.0
    fs_legend = 14.0

    norm_mat, algs = _radar_normalised()
    M = len(RADAR_AXES)
    angles = np.linspace(0.0, 2.0 * np.pi, M, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(6.4, 5.6),
                           subplot_kw=dict(projection="polar"))
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(90)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=fs_tick,
                       color="0.4")
    ax.set_xticks(angles)
    ax.set_xticklabels([lbl for lbl, _, _ in RADAR_AXES], fontsize=fs_axis)
    ax.tick_params(axis="x", pad=14)

    palette = {
        "FedAvg":   "#7f7f7f",
        "FedProx":  "#9467bd",
        "FedLwF":   "#8c564b",
        "SCAFFOLD": "#17becf",
        "DCFCL":    "#1f77b4",
        "DynDFCL": "#d62728",
    }
    handles = []
    for i, alg in enumerate(algs):
        vals = norm_mat[i].tolist() + [norm_mat[i][0]]
        is_focus = (alg == "DynDFCL")
        lw = 2.6 if is_focus else 1.3
        ls = "-" if is_focus else "--"
        a_fill = 0.22 if is_focus else 0.0
        line, = ax.plot(angles_closed, vals, ls, lw=lw,
                        color=palette[alg], label=alg, marker="o",
                        markersize=4 if is_focus else 3)
        if a_fill:
            ax.fill(angles_closed, vals, alpha=a_fill,
                    color=palette[alg])
        handles.append(line)

    # Title removed by request; legend pushed further down so it stops
    # overlapping the bottom-half axis labels ("CIFAR-100 Avg. Acc." and
    # "CIFAR-100 Anti-Forget."), which grew with the larger axis font.
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.28), ncol=3,
              fontsize=fs_legend, frameon=False,
              columnspacing=1.4, handletextpad=0.5)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.96, bottom=0.22)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[v] {out_pdf.name}")
    print(f"[v] {out_pdf.with_suffix('.png').name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------ fig5 ------
    panels = []
    diag_panels = []
    final_panels = []
    diag_data: dict[str, list[list[float]]] = {}
    final_data: dict[str, list[list[float]]] = {}
    for folder, label in PERCLIENT_ALGORITHMS:
        d = _load_perclient(folder)
        if d is None:
            print(f"[!] Missing per-client results for {folder}; skipping.")
            continue
        diag = _client_task_matrix(d, "per_client_diag_acc")
        final = _client_task_matrix(d, "per_client_final_acc")
        diag_panels.append(("After-task accuracy", label, diag))
        final_panels.append(("End-of-stream accuracy", label, final))
        diag_data[label] = diag.tolist()
        final_data[label] = final.tolist()
    # Only the after-task ("diag") row is rendered now; the end-of-stream
    # row was removed by request.  We still dump both into extra_figures.json
    # below so downstream tools can reconstruct it if needed.
    panels = diag_panels
    if panels:
        plot_bubble_matrix(panels, FIG_DIR / "client_task_bubble.pdf")

    # ------ fig6 ------
    plot_radar(FIG_DIR / "radar_metrics.pdf")

    # ------ dump data ------
    norm_mat, algs = _radar_normalised()
    out = {
        "client_task_bubble": {
            "after_task_diagonal": diag_data,
            "end_of_stream": final_data,
        },
        "radar_metrics": {
            "raw": RADAR_DATA,
            "axes": [
                {"label": lbl, "key": key, "invert_for_outer_is_better": inv}
                for (lbl, key, inv) in RADAR_AXES
            ],
            "normalised_outer_is_better": {
                alg: norm_mat[i].tolist() for i, alg in enumerate(algs)
            },
        },
    }
    out_json = DATA_DIR / "extra_figures.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[v] {out_json.relative_to(REPO)}")


if __name__ == "__main__":
    main()
