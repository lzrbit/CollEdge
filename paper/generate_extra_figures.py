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
      Five-axis radar chart comparing CollEdge (gradient mode, HP-tuned)
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
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "paper_experiments"
FIG_DIR = REPO / "paper" / "figures"
DATA_DIR = REPO / "paper" / "figure_data"

FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Unified font sizes (kept in sync with plot_figures.py)
FONT_BASE       = 9.0
FONT_TITLE      = 10.0
FONT_AXIS_LABEL = 9.0
FONT_TICK       = 8.0
FONT_LEGEND     = 8.0

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

PERCLIENT_ALGORITHMS = [
    ("FedAvg",    "FedAvg"),
    ("FedProx",   "FedProx"),
    ("DCFCL",     "DCFCL"),
    ("CollEdge",  "CollEdge"),
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

    We arrange them in a 2-row layout: first row "(a) after-task", second
    row "(b) end-of-stream", with one column per algorithm.
    """
    n_alg = len(PERCLIENT_ALGORITHMS)
    fig, axes = plt.subplots(
        nrows=2, ncols=n_alg, figsize=(2.85 * n_alg, 5.6),
        sharex=True, sharey=True,
    )
    if n_alg == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=100)

    # We expect panels grouped as: row0 = all algos diag, row1 = all algos final
    # panels is a flat list ordered: [(diag, alg1), (diag, alg2), ..., (final, alg1), ...]
    for idx, (row_label, alg_label, mat) in enumerate(panels):
        r, c = divmod(idx, n_alg)
        ax = axes[r, c]
        N, T = mat.shape
        xs, ys, vs = [], [], []
        for i in range(N):
            for t in range(T):
                xs.append(i)
                ys.append(t)
                vs.append(mat[i, t])
        sizes = [25 + (max(v, 0.0) / 100.0) ** 1.2 * 380 for v in vs]
        sc = ax.scatter(xs, ys, s=sizes, c=vs, cmap=cmap, norm=norm,
                        edgecolors="black", linewidths=0.4, alpha=0.92)
        # Cell labels for high-accuracy bubbles to reinforce the contrast
        for x, y, v in zip(xs, ys, vs):
            if v >= 70.0:
                ax.text(x, y, f"{v:.0f}", ha="center", va="center",
                        fontsize=6.2, color="white", fontweight="bold")
        ax.set_xticks(range(N))
        ax.set_xticklabels([f"C{i+1}" for i in range(N)], fontsize=FONT_TICK)
        ax.set_yticks(range(T))
        ax.set_yticklabels([f"T{t+1}" for t in range(T)], fontsize=FONT_TICK)
        ax.set_xlim(-0.7, N - 0.3)
        ax.set_ylim(-0.7, T - 0.3)
        ax.invert_yaxis()  # T1 on top
        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.55)
        ax.set_axisbelow(True)
        if r == 0:
            ax.set_title(alg_label, fontsize=FONT_TITLE, pad=4)
        if c == 0:
            ax.set_ylabel(row_label, fontsize=FONT_AXIS_LABEL)
        if r == 1:
            ax.set_xlabel("Client", fontsize=FONT_AXIS_LABEL)

    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, pad=0.025,
                        location="right", fraction=0.022)
    cbar.set_label("Per-client per-task accuracy (%)", fontsize=FONT_AXIS_LABEL)
    cbar.ax.tick_params(labelsize=FONT_TICK)

    fig.suptitle(
        "Client x Task accuracy on EMNIST-Letters: "
        "after-task (top) vs. end-of-stream (bottom)",
        fontsize=FONT_TITLE, y=0.995,
    )
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

RADAR_DATA = {
    "FedAvg":   dict(eacc=43.16, eforget=68.95, cacc=22.16, cforget=44.78, emerge=19.35),
    "FedProx":  dict(eacc=46.25, eforget=66.27, cacc=22.21, cforget=43.26, emerge=20.00),
    "FedLwF":   dict(eacc=42.83, eforget=70.52, cacc=22.50, cforget=43.58, emerge=18.00),
    "SCAFFOLD": dict(eacc=39.42, eforget=72.10, cacc=22.05, cforget=44.92, emerge=15.00),
    "DCFCL":    dict(eacc=41.38, eforget=79.99, cacc=12.08, cforget=66.27, emerge=7.49),
    "CollEdge": dict(eacc=93.51, eforget=9.49,  cacc=29.93, cforget=27.33, emerge=85.19),
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
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=FONT_TICK,
                       color="0.4")
    ax.set_xticks(angles)
    ax.set_xticklabels([lbl for lbl, _, _ in RADAR_AXES], fontsize=FONT_AXIS_LABEL)
    ax.tick_params(axis="x", pad=14)

    palette = {
        "FedAvg":   "#7f7f7f",
        "FedProx":  "#9467bd",
        "FedLwF":   "#8c564b",
        "SCAFFOLD": "#17becf",
        "DCFCL":    "#1f77b4",
        "CollEdge": "#d62728",
    }
    handles = []
    for i, alg in enumerate(algs):
        vals = norm_mat[i].tolist() + [norm_mat[i][0]]
        is_focus = (alg == "CollEdge")
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

    ax.set_title("Multi-metric Comparison (outer = better)",
                 fontsize=FONT_TITLE, pad=18)
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=3,
              fontsize=FONT_LEGEND, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.18)
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
        diag_panels.append(("(a) After-task accuracy", label, diag))
        final_panels.append(("(b) End-of-stream accuracy", label, final))
        diag_data[label] = diag.tolist()
        final_data[label] = final.tolist()
    panels = diag_panels + final_panels
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
