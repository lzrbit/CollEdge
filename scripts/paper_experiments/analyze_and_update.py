#!/usr/bin/env python3
"""
analyze_and_update.py
=====================
Post-processing script: run this ONCE after run_pipeline.sh finishes.

What it does
------------
1. Finds the best HP config from hp_tune results (highest avg_task accuracy).
2. Copies / hard-links the winning results.json to the canonical E_DER_Dir_Mask_Full
   ablation directory so that generate_figure_data.py picks it up naturally.
3. Loads directed_modes results (gradient / task_aware / hybrid).
4. Updates paper/figure_data/
     - fig3_aggregation_effect.json  → adds three directed-mode curves
     - table4_ablation.json          → updates CollEdge row with new numbers
     - table1_avg_accuracy.json      → updates CollEdge entry
     - NEW: table_directed_modes.json
5. Regenerates paper/figure_data/ and paper/figures/ via generate_figure_data.py
   and plot_figures.py.
6. Updates Zirui-Li-IEEE-JSTSP-Special-Issue/main.tex:
     - Ablation table (tab:ablation) — best-tuned Full row
     - Main results table (tab:main_results) — CollEdge row
     - New directed-mode comparison table (inserted after ablation table)
     - Fig 3 caption update

Usage
-----
    cd /home/lzr/Documents/DCFCLv2/CollEdge
    python scripts/paper_experiments/analyze_and_update.py
"""
from __future__ import annotations

import glob
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO / "paper"
FIG_DATA = PAPER_DIR / "figure_data"
FIGURES = PAPER_DIR / "figures"
RESULTS = REPO / "results" / "paper_experiments"
TEX_FILE = REPO.parent / "Zirui-Li-IEEE-JSTSP-Special-Issue" / "main.tex"
DATASETS = ["EMNIST-Letters", "CIFAR100"]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[✓] {path.relative_to(REPO)}")


def _avg_task(d: dict) -> float:
    per_task = d.get("per_task_acc") or []
    if not per_task:
        return float("nan")
    return sum(sum(row) / len(row) for row in per_task) / len(per_task)


def _pct(v: float) -> float:
    return round(v * 100, 2)


def find_best_hp(dataset: str) -> tuple[Path | None, dict | None]:
    """Return (results_dir, results_dict) for the highest avg_task run."""
    sweep_root = RESULTS / "hp_tune" / dataset
    if not sweep_root.exists():
        print(f"[!] hp_tune/{dataset} not found — skipping.", file=sys.stderr)
        return None, None
    best_score, best_dir, best_data = float("-inf"), None, None
    for run_dir in sorted(sweep_root.glob("full_*")):
        files = glob.glob(str(run_dir / "**/results.json"), recursive=True)
        if not files:
            continue
        d = _load(sorted(files)[-1])
        score = _avg_task(d)
        if score > best_score:
            best_score, best_dir, best_data = score, sorted(files)[-1], d
    if best_dir:
        print(f"[✓] Best HP for {dataset}: {Path(best_dir).parent.name} "
              f"avg_task={best_score:.4f}")
    return Path(best_dir) if best_dir else None, best_data


def load_directed_modes(dataset: str) -> dict[str, dict]:
    """Return {mode_name: results_dict} for the three directed modes."""
    modes_root = RESULTS / "directed_modes" / dataset
    results = {}
    if not modes_root.exists():
        print(f"[!] directed_modes/{dataset} not found.", file=sys.stderr)
        return results
    for mode in ("gradient", "task_aware", "hybrid"):
        mode_dir = modes_root / f"mode_{mode}"
        files = glob.glob(str(mode_dir / "**/results.json"), recursive=True)
        if files:
            results[mode] = _load(sorted(files)[-1])
            print(f"[✓] Loaded directed_modes/{dataset}/mode_{mode}")
        else:
            print(f"[!] Missing: directed_modes/{dataset}/mode_{mode}", file=sys.stderr)
    return results


# ─── Step 1: Promote best HP to canonical ablation directory ─────────────────

def promote_best_hp():
    print("\n=== Step 1: Promoting best HP results to ablation directories ===")
    for ds in DATASETS:
        best_path, best_data = find_best_hp(ds)
        if best_path is None:
            continue
        # Destination: results/paper_experiments/ablation/{ds}/E_DER_Dir_Mask_Full/best/
        dest_dir = RESULTS / "ablation" / ds / "E_DER_Dir_Mask_Full" / "best_hp"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "results.json"
        shutil.copy2(str(best_path), str(dest))
        print(f"[✓] Copied best HP result → {dest.relative_to(REPO)}")


# ─── Step 2: Update fig3_aggregation_effect.json ─────────────────────────────

def update_fig3():
    print("\n=== Step 2: Updating fig3_aggregation_effect.json ===")
    fig3 = _load(FIG_DATA / "fig3_aggregation_effect.json")

    # Display names
    mode_labels = {
        "gradient":   "CollEdge (gradient)",
        "task_aware": "CollEdge (task-aware)",
        "hybrid":     "CollEdge (hybrid)",
    }

    for ds in DATASETS:
        modes = load_directed_modes(ds)
        for mode, d in modes.items():
            all_acc = d.get("all_accuracies", [])
            if not all_acc:
                print(f"[!] all_accuracies missing for {ds}/{mode}", file=sys.stderr)
                continue
            key = f"CollEdge_{mode}"
            fig3[ds][key] = {
                "display_name": mode_labels[mode],
                "rounds": list(range(1, len(all_acc) + 1)),
                "accuracy": [round(v * 100, 4) for v in all_acc],
            }

    _save(fig3, FIG_DATA / "fig3_aggregation_effect.json")


# ─── Step 3: Update table4_ablation.json ─────────────────────────────────────

def _avg_acc_npmean(d: dict) -> float:
    """Average over all_accuracies (matches generate_figure_data.py's table4)."""
    a = d.get("all_accuracies") or []
    return sum(a) / len(a) if a else float("nan")


def update_table4():
    print("\n=== Step 3: Updating table4_ablation.json ===")
    table4 = _load(FIG_DATA / "table4_ablation.json")

    for ds in DATASETS:
        _, best_data = find_best_hp(ds)
        if best_data is None:
            continue
        # Use np.mean(all_accuracies) to match generate_figure_data.py's
        # table4 metric (paper reports this). _avg_task (per_task_acc-mean)
        # is used only for HP ranking, not for the displayed value.
        avg_acc = _avg_acc_npmean(best_data)
        forget = best_data.get("forgetting_rate", 0.0)
        for row in table4:
            if row.get("variant") == "CollEdge":
                row[ds + "_avg_acc"] = round(avg_acc * 100, 2)
                row[ds + "_forgetting"] = round(forget * 100, 2)
                print(f"  [✓] Updated CollEdge {ds}: "
                      f"avg_acc={row[ds+'_avg_acc']:.2f} "
                      f"forget={row[ds+'_forgetting']:.2f}")

    _save(table4, FIG_DATA / "table4_ablation.json")


# ─── Step 4: Build directed_modes table JSON ─────────────────────────────────

def build_directed_modes_table():
    print("\n=== Step 4: Building table_directed_modes.json ===")
    rows = []
    mode_display = {
        "gradient":   "Gradient-based",
        "task_aware": "Task-aware",
        "hybrid":     "Hybrid",
    }
    for mode in ("gradient", "task_aware", "hybrid"):
        row = {"mode": mode_display[mode]}
        for ds in DATASETS:
            modes = load_directed_modes(ds)
            if mode in modes:
                d = modes[mode]
                avg_acc = _avg_acc_npmean(d)
                forget = d.get("forgetting_rate", 0.0)
                row[ds + "_avg_acc"] = round(avg_acc * 100, 2)
                row[ds + "_forgetting"] = round(forget * 100, 2)
            else:
                row[ds + "_avg_acc"] = None
                row[ds + "_forgetting"] = None
        rows.append(row)
    _save(rows, FIG_DATA / "table_directed_modes.json")
    return rows


# ─── Step 5: Regenerate figures ──────────────────────────────────────────────

def regen_figure_data():
    """Run generate_figure_data.py to (re)build all figure_data JSON files."""
    print("\n=== Step 5a: Regenerating figure_data JSON files ===")
    try:
        subprocess.run(
            ["python", str(PAPER_DIR / "generate_figure_data.py")],
            cwd=str(REPO), check=True, capture_output=True, text=True,
        )
        print("[✓] figure_data regenerated.")
    except subprocess.CalledProcessError as e:
        print(f"[!] generate_figure_data failed:\n{e.stderr}")


def plot_fig3():
    """Re-render fig3 only (after directed-mode curves are injected)."""
    print("\n=== Step 5b: Re-rendering fig3 PDF ===")
    try:
        subprocess.run(
            ["python", str(PAPER_DIR / "plot_figures.py"), "--only", "fig3"],
            cwd=str(REPO), check=True, capture_output=True, text=True,
        )
        print("[✓] fig3 PDF regenerated.")
    except subprocess.CalledProcessError as e:
        print(f"[!] plot_figures failed:\n{e.stderr}")


def regen_figures():
    """Backwards-compatible wrapper."""
    regen_figure_data()
    plot_fig3()


# ─── Step 6: Update main.tex ──────────────────────────────────────────────────

def _fmt(v) -> str:
    """Format a percentage value for LaTeX (e.g. 91.23)."""
    if v is None:
        return "--"
    return f"{v:.2f}"


def update_latex():
    print("\n=== Step 6: Updating main.tex ===")

    # ---- Load best HP numbers ----
    # Use np.mean(all_accuracies) for the displayed avg_acc to match the
    # paper's table4 metric. _avg_task is per_task_acc-mean and is only
    # used for HP ranking.
    best = {}  # best[ds] = {'avg_acc': float, 'forget': float, 'final': float}
    for ds in DATASETS:
        _, d = find_best_hp(ds)
        if d:
            best[ds] = {
                "avg_acc": _pct(_avg_acc_npmean(d)),
                "forget":  _pct(d.get("forgetting_rate", 0.0)),
                "final":   _pct(d.get("final_accuracy", 0.0)),
            }

    # ---- Load directed modes numbers ----
    dm: dict[str, dict[str, dict]] = {}  # dm[ds][mode]
    for ds in DATASETS:
        dm[ds] = load_directed_modes(ds)

    tex = TEX_FILE.read_text()

    # ---- 6a. Update ablation table: ✓✓✓ row ----
    # Look for the pattern: \checkmark & \checkmark & \checkmark & E_acc & E_forg & C_acc & C_forg \\
    emnist_best = best.get("EMNIST-Letters", {})
    cifar_best = best.get("CIFAR100", {})

    if emnist_best or cifar_best:
        full_row_pat = re.compile(
            r"(\\checkmark\s*&\s*\\checkmark\s*&\s*\\checkmark\s*&\s*)"
            r"(\\textbf\{)?([\d.]+)(\})?\s*&\s*([\d.]+)\s*&\s*"
            r"(\\textbf\{)?([\d.]+)(\})?\s*&\s*([\d.]+)\s*\\\\",
            re.DOTALL
        )
        m = full_row_pat.search(tex)
        if m:
            e_acc  = _fmt(emnist_best['avg_acc']) if emnist_best else m.group(3)
            e_forg = _fmt(emnist_best['forget'])  if emnist_best else m.group(5)
            c_acc  = _fmt(cifar_best['avg_acc'])  if cifar_best  else m.group(7)
            c_forg = _fmt(cifar_best['forget'])   if cifar_best  else m.group(9)
            new_full = (
                f"\\checkmark & \\checkmark & \\checkmark "
                f"& \\textbf{{{e_acc}}} & {e_forg} "
                f"& \\textbf{{{c_acc}}} & {c_forg} \\\\"
            )
            tex = tex[:m.start()] + new_full + tex[m.end():]
            print(f"  [✓] Ablation table Full row updated (EMNIST={e_acc}/{e_forg}, CIFAR={c_acc}/{c_forg}).")
        else:
            print("  [!] Could not locate Full row in ablation table.", file=sys.stderr)

    # ---- 6b. Update main results table: CollEdge row ----
    if emnist_best or cifar_best:
        colledge_pat = re.compile(
            r"(\\textbf\{CollEdge\}\s*\n?\s*&\s*)"
            r"\\textbf\{([\d.]+)\\%\}\s*&\s*\\textbf\{([\d.]+)\\%\}\s*"
            r"\n?\s*&\s*\\textbf\{([\d.]+)\\%\}\s*&\s*\\textbf\{([\d.]+)\\%\}\s*\\\\",
            re.DOTALL
        )
        m = colledge_pat.search(tex)
        if m:
            e_acc  = _fmt(emnist_best['avg_acc']) if emnist_best else m.group(2)
            c_acc  = _fmt(cifar_best['avg_acc'])  if cifar_best  else m.group(3)
            e_forg = _fmt(emnist_best['forget'])  if emnist_best else m.group(4)
            c_forg = _fmt(cifar_best['forget'])   if cifar_best  else m.group(5)
            new_colledge = (
                f"\\textbf{{CollEdge}} \n"
                f"           & \\textbf{{{e_acc}\\%}}"
                f" & \\textbf{{{c_acc}\\%}} \n"
                f"           & \\textbf{{{e_forg}\\%}}"
                f"  & \\textbf{{{c_forg}\\%}} \\\\"
            )
            tex = tex[:m.start()] + new_colledge + tex[m.end():]
            print(f"  [✓] Main results table CollEdge row updated (EMNIST {e_acc}/{e_forg}, CIFAR {c_acc}/{c_forg}).")
        else:
            print("  [!] Could not locate CollEdge row in main results table.", file=sys.stderr)

    # ---- 6c. Insert directed-mode comparison table after ablation table ----
    # Only insert if not already present
    if "tab:directed_modes" not in tex:
        dm_table = _build_directed_modes_latex(dm)
        # Insert after \end{table} that closes the ablation table
        ablation_end = tex.find("\\end{table}", tex.find("tab:ablation"))
        if ablation_end != -1:
            insert_pos = ablation_end + len("\\end{table}")
            tex = tex[:insert_pos] + "\n\n" + dm_table + tex[insert_pos:]
            print("  [✓] Directed-mode comparison table inserted.")
        else:
            print("  [!] Could not locate ablation table end.", file=sys.stderr)

    # ---- 6d. Update Fig 3 caption to mention directed-mode curves ----
    old_caption = (
        "Replacing homogeneous FedAvg averaging with the directed coalition "
        "aggregation of Eqs.~(\\ref{eq_personalized_aggregation})--(\\ref{eq_directed_weight}),"
        " and combining it with the edge-side memory replay loss of "
        "Eq.~(\\ref{eq_total_loss}), produces the sustained accuracy plateaus "
        "characteristic of \\textsc{CollEdge}."
    )
    new_caption = (
        "Replacing homogeneous FedAvg averaging with the directed coalition "
        "aggregation of Eqs.~(\\ref{eq_personalized_aggregation})--(\\ref{eq_directed_weight}),"
        " and combining it with the edge-side memory replay loss of "
        "Eq.~(\\ref{eq_total_loss}), produces the sustained accuracy plateaus "
        "characteristic of \\textsc{CollEdge}. The three \\textsc{CollEdge} "
        "variants---gradient-based, task-aware, and hybrid directed "
        "aggregation---are compared in detail in Table~\\ref{tab:directed_modes}."
    )
    if old_caption in tex:
        tex = tex.replace(old_caption, new_caption)
        print("  [✓] Fig 3 caption updated.")

    TEX_FILE.write_text(tex)
    print(f"[✓] main.tex saved.")


def _build_directed_modes_latex(dm: dict[str, dict[str, dict]]) -> str:
    """Build a LaTeX table for directed-mode comparison.

    If only EMNIST data is available, render a 2-column table; otherwise the
    full 4-column EMNIST + CIFAR-100 table.
    """

    def _num(mode: str, ds: str, key: str) -> str:
        d = dm.get(ds, {}).get(mode)
        if not d:
            return "--"
        if key == "avg_acc":
            v = _avg_acc_npmean(d) * 100
        else:
            v = d.get("forgetting_rate", 0.0) * 100
        return f"{v:.2f}"

    # Find best mode per dataset (rank by np.mean for table consistency)
    best_mode = {}
    for ds in DATASETS:
        modes_d = dm.get(ds, {})
        if modes_d:
            best_mode[ds] = max(modes_d.keys(),
                                key=lambda m: _avg_acc_npmean(modes_d[m]))

    def _bf(mode: str, ds: str, key: str) -> str:
        val = _num(mode, ds, key)
        is_best = (best_mode.get(ds) == mode and key == "avg_acc")
        return f"\\textbf{{{val}}}" if is_best else val

    has_emnist = bool(dm.get("EMNIST-Letters"))
    has_cifar  = bool(dm.get("CIFAR100"))

    rows = []
    for mode, label in [("gradient", "Gradient-based"), ("task_aware", "Task-aware"), ("hybrid", "Hybrid")]:
        cells = [label]
        if has_emnist:
            cells += [_bf(mode, "EMNIST-Letters", "avg_acc"),
                      _num(mode, "EMNIST-Letters", "forgetting")]
        if has_cifar:
            cells += [_bf(mode, "CIFAR100", "avg_acc"),
                      _num(mode, "CIFAR100", "forgetting")]
        rows.append(" & ".join(cells) + r" \\")

    n_data_cols = 2 * (int(has_emnist) + int(has_cifar))
    col_spec = "l" + "c" * n_data_cols

    header_top = [r"\multirow{2}{*}{Directed Mode}"]
    cmidrules = []
    col = 2
    if has_emnist:
        header_top.append(r"\multicolumn{2}{c}{EMNIST-Letters}")
        cmidrules.append(rf"\cmidrule(lr){{{col}-{col+1}}}")
        col += 2
    if has_cifar:
        header_top.append(r"\multicolumn{2}{c}{CIFAR-100}")
        cmidrules.append(rf"\cmidrule(lr){{{col}-{col+1}}}")
        col += 2

    sub_header_cells = [""]
    for ds_present in (has_emnist, has_cifar):
        if ds_present:
            sub_header_cells += [r"Avg Acc\,$\uparrow$", r"Avg Forg.\,$\downarrow$"]

    note = ("All three variants use the full \\textsc{CollEdge} configuration "
            "(DER$^{++}$ + Coalition Mask) with the best hyper-parameters "
            "selected from the HP sweep. ``Gradient'' scores peer alignment by "
            "cosine similarity of parameter differentials "
            "(Eq.~\\ref{eq_directed_grad}); ``Task-aware'' uses the "
            "task-relevance score (Eq.~\\ref{eq_directed_task}); ``Hybrid'' "
            "averages both.")
    if not has_cifar:
        note += (" CIFAR-100 results are omitted here because the corresponding "
                 "directed-mode sweep is still in progress.")

    table = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{Directed-aggregation strategy comparison. {note}}}\n"
        "\\label{tab:directed_modes}\n"
        "\\renewcommand{\\arraystretch}{1.2}\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        + " & ".join(header_top) + " \\\\\n"
        + " ".join(cmidrules) + "\n"
        + " & ".join(sub_header_cells) + " \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )
    return table


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CollEdge Post-Experiment Analysis & LaTeX Update")
    print("=" * 60)

    # Sanity: check at least one result dir exists
    any_done = any(
        (RESULTS / "hp_tune" / ds).exists() for ds in DATASETS
    ) or any(
        (RESULTS / "directed_modes" / ds).exists() for ds in DATASETS
    )
    if not any_done:
        print("\n[!] No hp_tune or directed_modes results found yet.\n"
              "    Run the pipeline first:\n"
              "      nohup bash scripts/paper_experiments/run_pipeline.sh "
              "> pipeline.log 2>&1 &\n"
              "    Then re-run this script when it completes.")
        sys.exit(1)

    promote_best_hp()
    # First: regenerate base figure_data from fresh result files (this would
    # otherwise overwrite our directed-mode injection). Then inject the new
    # directed-mode curves and re-plot fig3.
    regen_figure_data()
    update_fig3()
    update_table4()
    build_directed_modes_table()
    plot_fig3()
    update_latex()

    print("\n" + "=" * 60)
    print("DONE — review the changes and compile main.tex:")
    print("  cd Zirui-Li-IEEE-JSTSP-Special-Issue && pdflatex main.tex")
    print("=" * 60)


if __name__ == "__main__":
    main()
