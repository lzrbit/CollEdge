"""
Generate all figure data for the paper from experimental results.
Outputs structured JSON files to paper/figure_data/.
"""
import json
import os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_BASE = os.path.join(BASE, "results", "paper_experiments")
OUTPUT_DIR = os.path.join(BASE, "paper", "figure_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Helper: load all results ────────────────────────────────────────────────

ABLATION_NAMES = {
    "A_DCFCL_baseline":    "DCFCL",
    "B_DER_only":          "DER only",
    "C_DER_Directed":      "DER+Dir",
    "D_DER_Mask":          "DER+Mask",
    "E_DER_Dir_Mask_Full": "CollEdge",
    "F_noDER_Directed":    "Dir only",
    "G_noDER_Mask":        "Mask only",
}

DATASETS = ["EMNIST-Letters", "CIFAR100"]

# Algorithms displayed in main comparison (order matters for plotting)
MAIN_ALGOS = [
    "Local", "FedAvg", "FedLwF", "FedProx",
    "SCAFFOLD", "PerAvg", "pFedMe", "DCFCL", "CollEdge"
]

ALGO_DISPLAY = {
    "Local":     "Local Only",
    "FedAvg":    "FedAvg",
    "FedLwF":    "FedLwF",
    "FedProx":   "FedProx",
    "SCAFFOLD":  "SCAFFOLD",
    "PerAvg":    "Per-FedAvg",
    "pFedMe":    "pFedMe",
    "DCFCL":     "DCFCL",
    "CollEdge": "CollEdge",
}

ABLATION_DISPLAY = {
    "DCFCL":     "DCFCL (Baseline)",
    "DER only":  "+DER++",
    "DER+Dir":   "+DER++ +Directed",
    "DER+Mask":  "+DER++ +Coalition",
    "CollEdge": "CollEdge",
    "Dir only":  "Directed only",
    "Mask only": "Coalition only",
}

def load_all():
    """Load all experiment results into a nested dict."""
    data = {ds: {} for ds in DATASETS}

    for ds in DATASETS:
        # Baselines
        bpath = os.path.join(RESULTS_BASE, "baselines", ds)
        if os.path.isdir(bpath):
            for exp_dir in sorted(os.listdir(bpath)):
                algo = exp_dir.split("_" + ds)[0]
                rf = os.path.join(bpath, exp_dir, "results.json")
                if os.path.exists(rf):
                    with open(rf) as f:
                        data[ds][algo] = json.load(f)

        # Ablation
        apath = os.path.join(RESULTS_BASE, "ablation", ds)
        if os.path.isdir(apath):
            for adir, aname in ABLATION_NAMES.items():
                dpath = os.path.join(apath, adir)
                if os.path.isdir(dpath):
                    for exp_dir in os.listdir(dpath):
                        rf = os.path.join(dpath, exp_dir, "results.json")
                        if os.path.exists(rf):
                            with open(rf) as f:
                                data[ds][aname] = json.load(f)
    return data

# ─── Fig 1: Communication Rounds vs Accuracy (all methods, all datasets) ─────

def gen_fig1_comm_rounds(data):
    """Line plots: accuracy vs. communication round for all methods."""
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for algo in MAIN_ALGOS:
            if algo in data[ds]:
                acc_list = data[ds][algo]["all_accuracies"]
                out[ds][algo] = {
                    "display_name": ALGO_DISPLAY.get(algo, algo),
                    "rounds": list(range(1, len(acc_list) + 1)),
                    "accuracy": [round(v * 100, 4) for v in acc_list],
                }
    with open(os.path.join(OUTPUT_DIR, "fig1_comm_rounds.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[✓] fig1_comm_rounds.json")

# ─── Table 1: Average Accuracy (over all rounds) ─────────────────────────────

def gen_table1_avg_accuracy(data):
    rows = []
    for algo in MAIN_ALGOS:
        row = {"algorithm": ALGO_DISPLAY.get(algo, algo)}
        for ds in DATASETS:
            if algo in data[ds]:
                acc_list = data[ds][algo]["all_accuracies"]
                avg = float(np.mean(acc_list)) * 100
                row[ds + "_avg_acc"] = round(avg, 2)
            else:
                row[ds + "_avg_acc"] = None
        rows.append(row)
    with open(os.path.join(OUTPUT_DIR, "table1_avg_accuracy.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("[✓] table1_avg_accuracy.json")

# ─── Table 2: Final Accuracy ──────────────────────────────────────────────────

def gen_table2_final_accuracy(data):
    rows = []
    for algo in MAIN_ALGOS:
        row = {"algorithm": ALGO_DISPLAY.get(algo, algo)}
        for ds in DATASETS:
            if algo in data[ds]:
                fa = data[ds][algo]["final_accuracy"] * 100
                row[ds + "_final_acc"] = round(fa, 2)
            else:
                row[ds + "_final_acc"] = None
        rows.append(row)
    with open(os.path.join(OUTPUT_DIR, "table2_final_accuracy.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("[✓] table2_final_accuracy.json")

# ─── Table 3: Catastrophic Forgetting ────────────────────────────────────────

def gen_table3_forgetting(data):
    rows = []
    for algo in MAIN_ALGOS:
        row = {"algorithm": ALGO_DISPLAY.get(algo, algo)}
        for ds in DATASETS:
            if algo in data[ds]:
                fr = data[ds][algo]["forgetting_rate"] * 100
                row[ds + "_forgetting"] = round(fr, 2)
            else:
                row[ds + "_forgetting"] = None
        rows.append(row)
    with open(os.path.join(OUTPUT_DIR, "table3_forgetting.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("[✓] table3_forgetting.json")

# ─── Table 4: Ablation Study ─────────────────────────────────────────────────

ABLATION_ORDER = ["DCFCL", "DER only", "DER+Dir", "DER+Mask", "CollEdge", "Dir only", "Mask only"]

def gen_table4_ablation(data):
    rows = []
    for aname in ABLATION_ORDER:
        row = {
            "variant": ABLATION_DISPLAY.get(aname, aname),
            "DER": "✓" if "DER" in aname or aname == "CollEdge" else "✗",
            "Directed": "✓" if "Dir" in aname or aname == "CollEdge" else "✗",
            "Coalition": "✓" if "Mask" in aname or aname == "CollEdge" else "✗",
        }
        # fix: DCFCL baseline has no modules
        if aname == "DCFCL":
            row["DER"] = "✗"; row["Directed"] = "✗"; row["Coalition"] = "✗"
        for ds in DATASETS:
            if aname in data[ds]:
                acc_list = data[ds][aname]["all_accuracies"]
                avg = float(np.mean(acc_list)) * 100
                fr = data[ds][aname]["forgetting_rate"] * 100
                row[ds + "_avg_acc"] = round(avg, 2)
                row[ds + "_forgetting"] = round(fr, 2)
            else:
                row[ds + "_avg_acc"] = None
                row[ds + "_forgetting"] = None
        rows.append(row)
    with open(os.path.join(OUTPUT_DIR, "table4_ablation.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("[✓] table4_ablation.json")

# ─── Fig 2: Catastrophic Forgetting Visualization (staircase per-task) ───────

def gen_fig2_forgetting_staircase(data):
    """
    Staircase plot data: after learning task t, test on tasks 0..t.
    per_task_acc[t] = list of accuracies on tasks 0..t after training task t.
    """
    out = {}
    TARGET_ALGOS = ["CollEdge", "DCFCL", "FedAvg", "Local"]
    for ds in DATASETS:
        out[ds] = {}
        for algo in TARGET_ALGOS:
            if algo not in data[ds]:
                continue
            per_task = data[ds][algo]["per_task_acc"]
            # per_task[t][j] = accuracy on task j after learning task t
            n_tasks = len(per_task)
            matrix = []
            for t in range(n_tasks):
                row = per_task[t]
                matrix.append([round(v * 100, 2) for v in row])
            out[ds][algo] = {
                "display_name": ALGO_DISPLAY.get(algo, algo),
                "n_tasks": n_tasks,
                "per_task_acc_matrix": matrix,
                # diagonal: accuracy on current task right after learning it
                "current_task_acc": [round(per_task[t][t] * 100, 2) for t in range(n_tasks)],
                # last row: final accuracy on all past tasks
                "final_per_task_acc": [round(v * 100, 2) for v in per_task[-1]],
            }
    with open(os.path.join(OUTPUT_DIR, "fig2_forgetting_staircase.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[✓] fig2_forgetting_staircase.json")

# ─── Fig 3: Effect of Aggregation on Client Accuracy ─────────────────────────

def gen_fig3_aggregation_effect(data):
    """
    Show how accuracy changes after each federation round vs. local-only.
    Use all_accuracies for CollEdge vs Local to illustrate aggregation benefit.
    """
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for algo in ["CollEdge", "DCFCL", "Local", "FedAvg"]:
            if algo in data[ds]:
                acc_list = data[ds][algo]["all_accuracies"]
                out[ds][algo] = {
                    "display_name": ALGO_DISPLAY.get(algo, algo),
                    "rounds": list(range(1, len(acc_list) + 1)),
                    "accuracy": [round(v * 100, 4) for v in acc_list],
                }
    with open(os.path.join(OUTPUT_DIR, "fig3_aggregation_effect.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[✓] fig3_aggregation_effect.json")

# ─── Fig 4: Emergence Comparison ─────────────────────────────────────────────

def gen_fig4_emergence(data):
    """
    Emergence: CollEdge clients learn classes never seen locally.
    Use per_task_acc to show task accuracy even when client has no local data for that task.
    We define emergence as: in the final round, accuracy > threshold on tasks the client
    never locally trained on.
    Here we use per_task_acc diagonal (local learning) vs off-diagonal (cross-task accuracy).
    """
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for algo in ["CollEdge", "DCFCL", "FedAvg"]:
            if algo not in data[ds]:
                continue
            per_task = data[ds][algo]["per_task_acc"]
            n = len(per_task)
            # Final row: accuracy on all tasks after full training
            final_row = [round(v * 100, 2) for v in per_task[-1]]
            # Avg of first n-1 tasks (past tasks) at final stage
            past_avg = round(float(np.mean(per_task[-1][:-1])) * 100, 2) if n > 1 else 0.0
            # Current (latest) task accuracy
            current_acc = round(per_task[-1][-1] * 100, 2)
            out[ds][algo] = {
                "display_name": ALGO_DISPLAY.get(algo, algo),
                "n_tasks": n,
                "final_per_task_acc": final_row,
                "past_tasks_avg_at_final": past_avg,
                "current_task_acc_at_final": current_acc,
                # Full lower-triangular per-task accuracy matrix (% in [0, 100]):
                # per_task_acc_matrix[t][j] = accuracy on task j (j<=t) after
                # finishing the local training of task t. Used by plot_fig4 to
                # visualize how the accuracy on each task "emerges" stage by
                # stage at clients that never locally trained on it.
                "per_task_acc_matrix": [
                    [round(v * 100, 2) for v in per_task[t]]
                    for t in range(n)
                ],
                # Full history of past-task avg: how well model retains old tasks over time
                "past_avg_over_time": [],
            }
            # Build past_avg_over_time: after learning task t, avg acc on tasks 0..t-1
            for t in range(n):
                if t == 0:
                    out[ds][algo]["past_avg_over_time"].append(None)
                else:
                    past = float(np.mean(per_task[t][:t])) * 100
                    out[ds][algo]["past_avg_over_time"].append(round(past, 2))
    with open(os.path.join(OUTPUT_DIR, "fig4_emergence.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[✓] fig4_emergence.json")

# ─── Table 5: Computational Complexity ───────────────────────────────────────

def gen_table5_complexity():
    """
    Complexity analysis table (theoretical, hand-crafted based on algorithms).
    """
    rows = [
        {
            "algorithm": "FedAvg",
            "client_complexity": "O(E·|B|·d)",
            "server_complexity": "O(n·d)",
            "communication_cost": "O(n·d)",
            "extra_memory": "O(d)",
            "notes": "Baseline. E=local epochs, |B|=batch size, d=model size, n=num clients"
        },
        {
            "algorithm": "FedProx",
            "client_complexity": "O(E·|B|·d)",
            "server_complexity": "O(n·d)",
            "communication_cost": "O(n·d)",
            "extra_memory": "O(d)",
            "notes": "Adds proximal term; same asymptotic complexity as FedAvg"
        },
        {
            "algorithm": "SCAFFOLD",
            "client_complexity": "O(E·|B|·d)",
            "server_complexity": "O(n·d)",
            "communication_cost": "O(2n·d)",
            "extra_memory": "O(d)",
            "notes": "Doubles communication for control variates"
        },
        {
            "algorithm": "Per-FedAvg",
            "client_complexity": "O(2E·|B|·d)",
            "server_complexity": "O(n·d)",
            "communication_cost": "O(n·d)",
            "extra_memory": "O(d)",
            "notes": "MAML-style meta-gradient doubling cost"
        },
        {
            "algorithm": "pFedMe",
            "client_complexity": "O(E·|B|·d + K·d)",
            "server_complexity": "O(n·d)",
            "communication_cost": "O(n·d)",
            "extra_memory": "O(2d)",
            "notes": "K inner steps for personalized model"
        },
        {
            "algorithm": "FedLwF",
            "client_complexity": "O(E·|B|·2d)",
            "server_complexity": "O(n·d)",
            "communication_cost": "O(n·d)",
            "extra_memory": "O(2d)",
            "notes": "Double forward pass for knowledge distillation"
        },
        {
            "algorithm": "DCFCL",
            "client_complexity": "O(E·|B|·2d + C·n²)",
            "server_complexity": "O(n²·d)",
            "communication_cost": "O(n·d)",
            "extra_memory": "O(2d + n²)",
            "notes": "Coalition game O(n²), KD+Proto-Aug double pass"
        },
        {
            "algorithm": "CollEdge",
            "client_complexity": "O(E·|B|·3d + B_size + C·n²)",
            "server_complexity": "O(n²·d)",
            "communication_cost": "O(n·d)",
            "extra_memory": "O(3d + B_size + n²)",
            "notes": "DER++ adds replay buffer B_size, directed collab O(n²). No extra comm."
        },
    ]
    with open(os.path.join(OUTPUT_DIR, "table5_complexity.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("[✓] table5_complexity.json")

# ─── Pseudocode data (structured description) ────────────────────────────────

def gen_pseudocode():
    pseudocode = {
        "algorithm_name": "CollEdge: Decentralized Federated Continual Learning with Directed Collaboration",
        "input": [
            "N clients, T tasks, R communication rounds per task",
            "Learning rates η, λ_kd, λ_proto, α (DER), β (ER)",
            "Buffer size B, directed temperature τ, self-weight w_self",
            "Coalition formation threshold δ"
        ],
        "output": [
            "Personalized models {θ_i} for each client i"
        ],
        "steps": [
            {
                "phase": "Initialization",
                "lines": [
                    "Initialize model θ_i for each client i",
                    "Initialize replay buffer Buf_i ← ∅ for each client i",
                    "Initialize gradient store G_i ← ∅"
                ]
            },
            {
                "phase": "Outer Loop: Tasks",
                "lines": [
                    "for task t = 1, 2, ..., T do:",
                    "  Each client i receives new task data D_i^t",
                    "  for round r = 1, 2, ..., R do:",
                    "    ── CLIENT LOCAL TRAINING ──",
                    "    for each client i in parallel do:",
                    "      Sample mini-batch (x, y) ~ D_i^t",
                    "      Compute L_CE = CrossEntropy(f_θi(x), y)",
                    "      Compute L_KD = KLDiv(f_θi(x), f_θ_old(x))   [knowledge distillation]",
                    "      Compute L_Proto = ProtoAug loss              [prototype augmentation]",
                    "      if Buf_i ≠ ∅ then:",
                    "        Sample (x_b, y_b, z_b) ~ Buf_i",
                    "        L_DER = MSE(f_θi(x_b), z_b)               [dark experience replay]",
                    "        L_ER  = CrossEntropy(f_θi(x_b), y_b)      [experience replay]",
                    "      else: L_DER = L_ER = 0",
                    "      L_total = L_CE + λ_kd·L_KD + λ_proto·L_Proto + α·L_DER + β·L_ER",
                    "      Update θ_i ← θ_i - η·∇L_total",
                    "      Store current logits in Buf_i via reservoir sampling",
                    "      Record gradient g_i ← ∇L_CE",
                    "    end parallel",
                    "    ── SERVER COALITION FORMATION ──",
                    "    Clients send {θ_i, g_i, class_set_i} to server",
                    "    Compute pairwise cosine similarity S[i][j] = cos(g_i, g_j)",
                    "    Form coalitions C via cooperative game with benefit φ_i(C):",
                    "      φ_i(C) = <g_C^agg, g_i> / ||g_C^agg||         [directed benefit]",
                    "    ── DIRECTED AGGREGATION ──",
                    "    Compute directed collaboration matrix D[i][j]:",
                    "      D[i][j] = 0.5·cos(∇_i, ∇_j)·sqrt(||∇_j||/||∇_i||)",
                    "              + 0.5·(0.4·relevance + 0.3·diversity + 0.3·experience)",
                    "    for each client i do:",
                    "      Filter: J = {j ∈ C_i : D[i][j] > δ}",
                    "      w_ij = softmax(D[i][j]/τ) · s_j  for j ∈ J",
                    "      θ_i^agg = w_self·θ_i + (1-w_self)·Σ_{j∈J} w_ij·θ_j",
                    "    Distribute personalized models {θ_i^agg} to clients",
                    "    θ_i ← θ_i^agg  for each client i",
                    "  end rounds",
                    "  Update old model θ_old ← θ_i  (for KD in next task)",
                "end tasks"
                ]
            }
        ]
    }
    with open(os.path.join(OUTPUT_DIR, "pseudocode.json"), "w") as f:
        json.dump(pseudocode, f, indent=2)
    print("[✓] pseudocode.json")

# ─── Summary stats for the paper ──────────────────────────────────────────────

def gen_summary_stats(data):
    summary = {}
    for ds in DATASETS:
        summary[ds] = {}
        for algo in list(MAIN_ALGOS) + ["DER only", "DER+Dir", "DER+Mask", "Dir only", "Mask only"]:
            if algo not in data[ds]:
                continue
            acc_list = data[ds][algo]["all_accuracies"]
            fr = data[ds][algo]["forgetting_rate"]
            fa = data[ds][algo]["final_accuracy"]
            summary[ds][algo] = {
                "display": ALGO_DISPLAY.get(algo, ABLATION_DISPLAY.get(algo, algo)),
                "avg_acc_pct": round(float(np.mean(acc_list)) * 100, 2),
                "final_acc_pct": round(fa * 100, 2),
                "forgetting_pct": round(fr * 100, 2),
            }
    with open(os.path.join(OUTPUT_DIR, "summary_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[✓] summary_stats.json")

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading experimental results...")
    data = load_all()
    print(f"Loaded datasets: {list(data.keys())}")
    for ds, algos in data.items():
        print(f"  {ds}: {list(algos.keys())}")
    print()

    gen_fig1_comm_rounds(data)
    gen_table1_avg_accuracy(data)
    gen_table2_final_accuracy(data)
    gen_table3_forgetting(data)
    gen_table4_ablation(data)
    gen_fig2_forgetting_staircase(data)
    gen_fig3_aggregation_effect(data)
    gen_fig4_emergence(data)
    gen_table5_complexity()
    gen_pseudocode()
    gen_summary_stats(data)

    print(f"\nAll figure data written to: {OUTPUT_DIR}/")
