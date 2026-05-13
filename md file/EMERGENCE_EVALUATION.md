# Emergence Evaluation Protocol

> Detailed specification of how the *collective intelligence emergence* phenomenon is measured in CollEdge.
> All references below point to the source of truth in this repository.

---

## 1. Definition

A client $i$ exhibits **emergence** when, after federated continual training, it can correctly classify samples whose **ground-truth class never appeared in its own local training stream**.

Formally, let

$$
\mathcal{Y}_i^{\text{local}} \;=\; \bigcup_{t=1}^{T}\,\mathcal{Y}_i^{(t)} \;\subseteq\; \mathcal{Y}
$$

be the set of classes ever seen by client $i$ across all $T$ tasks, and let $\mathcal{Y}=\bigcup_{i=1}^{N}\mathcal{Y}_i^{\text{local}}$ be the global label space. The set of *unseen* (a.k.a. *foreign*) classes for client $i$ is

$$
\mathcal{U}_i \;=\; \mathcal{Y} \setminus \mathcal{Y}_i^{\text{local}}.
$$

Any correct prediction by $f_{\boldsymbol{\theta}_i}$ on a sample $(x,y)$ with $y\in\mathcal{U}_i$ is, by construction, *not derivable from local data alone* — the relevant decision regions can only have been transferred through the federated aggregation in Eq. (10)–(11) of the paper.

---

## 2. Evaluation Pipeline

The evaluation runs **once, after the final communication round** of the experiment, for every client model. The algorithm is implemented in two layers:

| Layer  | File | Function | Role |
|--------|------|----------|------|
| Server | [core/server.py](core/server.py) | `evaluate_emergence()` (line 1263) | Orchestrates client loop, builds global views, aggregates metrics |
| Client | [FL_model/base_client.py](FL_model/base_client.py) | `evaluate_emergence()` (line 495) | Per-client inference and seen/unseen bookkeeping |

### 2.1 Step 1 — Build the global label inventory

For each task index $t\in\{1,\dots,T\}$ the server scans the **shared test loader** (identical across clients) and records the union of ground-truth labels appearing in that task,

$$
\mathcal{Y}^{(t)} \;=\; \bigcup_{(x,y)\in\mathcal{D}_{\text{test}}^{(t)}}\{y\}.
$$

The list `all_labels_per_task = [Y(1), …, Y(T)]` is forwarded to every client as the *global view* needed to decide which test samples belong to a client's unseen set.

### 2.2 Step 2 — Per-client inference

For client $i$, set `local_seen_classes = self.classes_so_far` (i.e., $\mathcal{Y}_i^{\text{local}}$). For each test sample $(x,y)$ across **all $T$ tasks**:

1. Forward pass: $\hat{y} = \arg\max_c f_{\boldsymbol{\theta}_i}(x)_c$, $\;p = \max_c \mathrm{softmax}\!\bigl(z_c\bigr)$.
2. Bucket assignment:
   - If $y\in\mathcal{Y}_i^{\text{local}}$  →  **seen** sample: increment `seen_total`, and `seen_correct` if $\hat{y}=y$.
   - Else ($y\in\mathcal{U}_i$)            →  **unseen** sample: increment `unseen_total`, and on a correct prediction also store `(x, y, ŷ, p, t, i, Y_i^local)` in `emergence_samples` for later qualitative inspection.
3. Per-class tally: `per_class_emergence[y] += (correct, total)` for every $y\in\mathcal{U}_i$.

### 2.3 Step 3 — Aggregate metrics

After all clients finish, the server aggregates the following quantities.

| Metric | Formula | Meaning |
|--------|---------|---------|
| Global emergence rate | $E = \dfrac{\sum_i \text{unseen\_correct}_i}{\sum_i \text{unseen\_total}_i}$ | The headline number. Reported as `global_emergence_rate`. |
| Seen accuracy | $A_{\text{seen}} = \dfrac{\sum_i \text{seen\_correct}_i}{\sum_i \text{seen\_total}_i}$ | Sanity baseline. Should be high; small gap to $E$ ⇒ strong emergence. |
| Per-client emergence | $E_i = \dfrac{\text{unseen\_correct}_i}{\text{unseen\_total}_i}$ | Distribution of emergence across the federation. |
| Per-class emergence | $E_c = \dfrac{\sum_i\text{correct}_{i,c}}{\sum_i\text{total}_{i,c}}$ | Identifies which classes are easy / hard to "transmit" through the network. |
| Knowledge transfer matrix | $T[i][j] = \bigl|\{c \in \mathcal{Y}_j^{\text{local}}\setminus\mathcal{Y}_i^{\text{local}}\;:\; \text{client } i \text{ correctly predicts } c\text{ at least once}\}\bigr|$ | Directed flow: class-level evidence that knowledge from $j$ reached $i$. |

### 2.4 Step 4 — Persistence

`save_emergence_data(results, output_dir)` writes the structured payload to `<run>/emergence_analysis/`:

```
<run>/
└── emergence_analysis/
    ├── emergence_metadata.json   # all scalar/array metrics above
    └── emergence_samples.pkl     # raw (x, y, ŷ, p, t, client_id) tuples
```

`emergence_samples.pkl` carries the original input tensor `x`, enabling later qualitative figures (e.g., showing a CIFAR-100 image of class "rocket" correctly classified by a client that only ever saw "fish", "rose", and "lamp").

---

## 3. Counterfactual Comparison: With vs. Without Federation

The **federation increment** quantifies how much of the emergence is causally attributable to collaboration rather than to overlapping data. We run **paired** experiments under identical seeds, splits, and architectures:

| Condition | `algorithm` | Expected $E$ |
|-----------|-------------|--------------|
| Local (no federation) | `Local` | ≈ 0 — model never receives parameters from any other client. |
| CollEdge (federation) | `CollEdge` | Empirically positive — measures collective knowledge transfer. |

The driver script is [scripts/run_emergence_evaluation.py](scripts/run_emergence_evaluation.py); it parses the `Global Emergence Rate:` field from each run's `training.log` to produce the comparison table.

---

## 4. Practical Considerations

1. **Statistical noise.** A class that appears only once in the test set will register $E_c\in\{0,1\}$. Class-level conclusions should be drawn only when `total ≥ 30`.
2. **Memorization vs. emergence.** Replay-based methods (e.g., DER++) do *not* affect the unseen-class metric, because the client's buffer can store only data that has appeared *locally*. Therefore $E_i>0$ is unambiguous evidence of *parameter-mediated* knowledge transfer, not of replay leakage.
3. **Coalition mask interaction.** When `use_coalition_mask` partitions the federation into disjoint groups (Sec. III-B of the paper), the maximum achievable $E_i$ for client $i$ is bounded by the union of classes in $i$'s connected component. The script automatically reports this bound.
4. **Initialization parity.** All clients start from the same initial weights $\boldsymbol{\theta}^{(0)}$ at $r=0$. Without federation (`algorithm=Local`) the only commonality across runs is this initial point, which guarantees $E\to 0$ in expectation as training progresses.

---

## 5. How to Reproduce

```bash
# Single run with emergence evaluation enabled
python main.py \
    --config configs/colledge_complete_emnist.yaml \
    --evaluate_emergence

# Counterfactual: paired CollEdge vs. Local
python scripts/run_emergence_evaluation.py \
    --dataset EMNIST-Letters \
    --output results/emergence/
```

The summary printed at the end of each `training.log` includes:

```text
Global Emergence Rate (unseen class accuracy): 0.XXXX
Seen Class Accuracy:                           0.XXXX
Total Emergence Samples:                       NNNN
Unseen: <correct>/<total>
Seen:   <correct>/<total>

Per-client Emergence:
  Client 0: unseen_acc=…  seen_acc=…  local_classes=[…]
  …

Knowledge Transfer Matrix (row=receiver, col=source):
  Client 0: [ 0,  3, 11,  …]
  …
```

These three blocks are sufficient to reconstruct every quantity defined in §2.

---

## 6. Reporting Convention in the Paper

| Element | Source field | Section in `main.tex` |
|---------|-------------|-----------------------|
| Headline emergence number  | `global_emergence_rate` × 100 | Sec. V-D-4 (Emergent Collective Intelligence) |
| Per-client distribution    | `per_client_summary` | Fig. 4 (per-task accuracy bars at final round) |
| Cross-client knowledge flow | `knowledge_transfer_matrix` | Optional appendix figure |
| Past-task retention curve  | derived from `per_task_acc` of the run | Fig. 5 (timeline) |

When reporting, always pair the federation-condition emergence rate with its *Local* counterpart so the reader can read off the federation increment $\Delta E = E_{\text{CollEdge}} - E_{\text{Local}}$ at a glance.
