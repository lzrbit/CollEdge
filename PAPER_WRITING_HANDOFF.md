# Paper Writing Handoff — DynDFCL JSTSP Submission

**Last updated**: 2026-05-22
**Status**: Draft mostly complete (15 pages compiled clean). **Next task: De-AI polish pass.**

This document hands off the paper-writing context to a new session. Read it once; it should be enough to resume work without re-deriving anything.

---

## 1. Repos & paths

Two GitHub repos coupled to this paper, both on `main` and pushed:

| Repo | Path | Role |
|---|---|---|
| **CollEdge** (code) | `/Users/lzr/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/3-GitHub/DCFCL/CollEdge` | model, training, results, figure scripts |
| **JSTSP** (paper) | `/Users/lzr/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/3-GitHub/DCFCL/Zirui-Li-IEEE-JSTSP-Special-Issue` | `main.tex`, `ref.bib`, `figures/`, `main.pdf` |

Workflow: figures regenerated in CollEdge/paper, then synced to JSTSP/figures via `cp`. Both repos are pushed; pull before editing.

SSH for GitHub uses `~/.ssh/id_ed25519_github` (configured in `~/.ssh/config` for `Host github.com`).

Conda env with matplotlib: `/opt/miniconda3/envs/fcl/bin/python`.

LaTeX: TeX Live 2026 basic + `threeparttable` installed in user texmf (`tlmgr --usermode install threeparttable`).

---

## 2. Method name & variants — CRITICAL

- **Method**: `DynDFCL` (renamed from `CollEdge` in this session; 105+ replacements across main.tex, paper/*.py, paper/figure_data/*.json).
- **Three variants** (Roman numerals):
  - **DynDFCL-I** = gradient-aligned mode (Eq. `eq_directed_grad`)
  - **DynDFCL-II** = task-aware mode (Eq. `eq_directed_task`)
  - **DynDFCL-III** = hybrid mode (Eq. `eq_directed_hyb`)
- Variant labels (`DynDFCL-I/II/III`) appear **only in Table 2 (tab:main_results) and Table 3 (tab:task_wise_accuracy)**.
- Everywhere else use the unqualified name `DynDFCL`. Body text states: "Unless explicitly stated otherwise, the unqualified name `DynDFCL` in the remainder of the paper refers to the best-performing variant of the corresponding benchmark."
- The headline configuration (when one must be picked) is **DynDFCL-I (gradient)**.

### Disk folder NOT renamed
- `CollEdge/results/paper_experiments/perclient/EMNIST-Letters/CollEdge/` still exists with raw experiment outputs.
- `generate_extra_figures.py` has `("CollEdge", "DynDFCL")` tuple (disk → display name).
- `generate_figure_data.py` has `DISK_REMAP = {"CollEdge": "DynDFCL"}`.
- Do **NOT** rename the disk folder — code already handles the mapping.

---

## 3. Protected content — DO NOT EDIT

The 3 enumerate items + the "We empirically demonstrate..." sentence immediately after `We propose a decentralized federated continual learning framework that` in the Introduction are **user-polished and locked**.

```latex
We propose a decentralized federated continual learning framework that
\begin{enumerate}
    \item jointly addresses task-incremental adaptation, client-specific personalization and decentralized cross-client collaboration at the network edge.
    \item mitigates catastrophic forgetting locally with an edge-side memory replay that maintains a per-client experience buffer to rehear stored labels and output logits.
    \item enables asymmetric and task-aware knowledge transfer among dynamically formed client coalitions with a directed cooperation mechanism for decentralized aggregation.
\end{enumerate}
We empirically demonstrate that the proposed framework promotes the emergence of collective intelligence, allowing clients to benefit from distributed experiences beyond their own local task streams.
```

---

## 4. Style & notation conventions (established this session)

### LaTeX style
- **No `\textsc{}`** wrappers on method names. `\textsc{CollEdge}` was replaced with plain `DynDFCL`.
- **No `\emph{}`** on dataset / algorithm / enumeration labels. The following were globally stripped:
  - `\emph{EMNIST-Letters}`, `\emph{CIFAR-100}`
  - `\emph{FedAvg}`, `\emph{FedProx}`, `\emph{FedLwF}`, `\emph{SCAFFOLD}`, `\emph{Local}`, `\emph{DCFCL}`, `\emph{Clustered-FL}`, `\emph{pFedMe}`
  - `\emph{First}`, `\emph{Second}`, `\emph{Third}`
- `\emph{}` is allowed on **terms being defined** ("dynamic directed mask", "decentralized", "collective intelligence") and **contrastive emphasis** ("asymmetric"). Remove if it's just generic emphasis.
- **Subscript `\mathrm{}` → `\text{}`**: all 138 `_{\mathrm{...}}` and `^{\mathrm{...}}` patterns converted. Only operator-level `\mathrm{Acc}`, `\mathrm{Forget}`, `\mathrm{KL}`, `\mathrm{Uniform}` kept.
- **Tables**: use `\begin{threeparttable}` with `\begin{tablenotes}\footnotesize\item ...\end{tablenotes}` for auxiliary content. Package `threeparttable` is loaded.
- **Bold best / underline second best** in result tables (strict numerical ranking). Tablenotes carry the convention statement plus a footnote on the pFedMe-degenerate-low-forget edge case.

### Notation (defined in `\newcommand` block and used consistently)
- `\Param` = $\boldsymbol{\theta}$ (parameters)
- `\Grad` = $\boldsymbol{g}$ (local update direction)
- `\Coal` = $\mathcal{C}$ (coalition)
- `\Part` = $\Pi$ (partition)
- `\Data` = $\mathcal{D}$
- `\Buffer` = $\mathcal{B}$
- `\Loss` = $\mathcal{L}$
- `\Logit` = $\boldsymbol{z}$
- $D^{(r)}$, $M^{(r)}$, $\mathcal{N}_i^{\text{tr},(r)}$ — round-superscripted directed matrix, dynamic directed mask, trusted neighbourhood
- $\Loss_i^{\text{plas}}$, $\Loss_i^{\text{stab}}$, $\Delta_i^{\text{coll}}$ — three per-client objectives

The notation table (`tab:notation`) lives at the start of Section 4 (Overview) — moved forward from the appendix during this session.

---

## 5. Paper structure (as of last compile, 15 pages)

| § | Title | Label | Notes |
|---|---|---|---|
| 1 | Introduction | `sec_introduction` | Just rewritten — 5 prose paragraphs + protected contribution + organization. Ends Section 2 at line 148. |
| 2 | Related Works | `sec_relatedwork` | 3 subsections × ~0.25 page each: Federated Learning, Continual Learning, Collective Intelligence. 4 user-specified collective AI papers cited. |
| 3 | Problem Formulation | `sec_problem` | Subsections: Edge-Side Continual Learning (`sec_edge_cl`) + Decentralized Federated Continual Learning (`sec_dfcl`). Has a deliberate bridge paragraph. |
| 4 | Overview of the Proposed Approach | — | Subsections: Learning Objectives (`sec_objectives`) + Three-Stage Realisation (`sec_three_stage`). Contains tab:notation. |
| 5 | Collaboration Strategy | `sec_coalition` | Subsections: Coalitional Affinity Game / Dynamic Stable Partition / Directed Collaboration Matrix [3 subsubsections: Gradient-aligned / Task-aware / Hybrid] (`sec_directed`) / **Dynamic Directed Collaboration Mask** (`sec_dynamic_mask`) / Personalized Coalition Aggregation (`sec_aggregation`). |
| 6 | Edge-Side Memory Replay | `sec_replay` | Subsections: Replay-based CL + Composite Replay Loss. Closes the Sec 4 objectives via `eq_stab_decomposition`. |
| 7 | Experiments | `sec_experiments` | Setup / Metrics (`subsec:metrics`) / Implementation details (`sec:detailed_settings` with `sec:backbone` subsubsection) / Quantitative Results / Mitigating Catastrophic Forgetting / Ablation (`subsubsec:ablation`) / Computational and Communication Cost / Emergent Collective Intelligence (`subsubsec:emergence`). |
| 8 | Conclusion | `sec_conclusion` | (Not yet written by us in this session.) |
| App. A | Proof of Theorem: Dynamic Cooperative Equilibrium | `app_proof` | |
| App. B | Equiprobability of Reservoir Sampling | `app_reservoir` | |
| App. C | Computational Complexity | `sec:complexity` | |

**Two Algorithms** (split during this session):
- **Algorithm 1** = "Edge-Side Continual Training at Client $i$, Round $r$" (mirrors Sec 6, `alg_edge_training`)
- **Algorithm 2** = "Decentralized Collaboration at Client $i$, Round $r$" (mirrors Sec 5, `alg_collaboration`); consumes $(\Grad_i, \widetilde{\Param}_i^{(r)})$ from Alg 1.
- Stage 1/2/3 labelling removed. Alg 1 ID = `alg_edge_training`, Alg 2 ID = `alg_collaboration`.

---

## 6. Figures (6 result figures + 1 placeholder)

| Fig | Caption anchor | Generator | Status |
|---|---|---|---|
| 1 | `fig:placeholder` | Manual (`figures/fig_main.png`, AI-generated draft) | Placeholder; needs final hero figure |
| 2 | `fig:forgetting_heatmap_emnist` | `plot_figures.py` fig2 non-wide mode | Single col, big fonts (axis=14, cells=11) |
| 3 | `fig:forgetting_heatmap_cifar100` | `plot_figures.py` fig2 wide mode | figure*, all fonts uniform = 14 |
| 4 | `fig:aggregation_effect` (accuracy vs round) | `plot_figures.py` fig3 | figure* @ 0.98\linewidth, fonts uniform = 8.5 |
| 5 | `fig:emergence` | `plot_figures.py` fig4 | Single col, 2×3 + per-row colourbars, fonts = 7-8 |
| 6 | `fig:bubble_matrix` (client_task_bubble) | `generate_extra_figures.py` | 1×4 after-task only, PowerNorm(gamma=3) non-linear colour + size, no in-bubble labels |
| 7 | `fig:radar_metrics` | `generate_extra_figures.py` | Single col @ 0.78\linewidth; data sourced **strictly from tab:main_results + tab:task_wise_accuracy** (DynDFCL is on outer ring on 4 of 5 axes; SCAFFOLD wins CIFAR anti-forget because its CIFAR acc is degenerate). Local font override 14-17. No title. |

To regenerate: from `CollEdge/paper/`:
```bash
/opt/miniconda3/envs/fcl/bin/python plot_figures.py           # fig2/3/4/5
/opt/miniconda3/envs/fcl/bin/python generate_extra_figures.py # fig6/7
```
Then sync 6 PDFs to JSTSP/figures (no PNGs needed in JSTSP repo).

---

## 7. Tables (6 result tables; 4 backbone/HP/baseline tables removed)

| Table | Anchor | Notes |
|---|---|---|
| 1 | `tab:fcl_protocol` = `tab:dyndfcl_hp` (dual label) | Merged protocol + DynDFCL hyper-parameters. Has Distillation + Replay blocks. Directed Coal. block removed by user request earlier. |
| 2 | `tab:main_results` | **Dataset-first column order**: Algorithm \| EMNIST(Acc, Forget) \| CIFAR(Acc, Forget). Bold best, underline 2nd. pFedMe pathology footnoted. DynDFCL-I/II/III rows below baselines. |
| 3 | `tab:task_wise_accuracy` | 6 EMNIST tasks. Bold/underline per column. DynDFCL-I/II/III. |
| 4 | `tab:task_wise_accuracy_10tasks` | 10 CIFAR tasks + Avg. Single DynDFCL row (no variant split). Bold/underline. threeparttable wrapper added. |
| 5 | `tab:ablation` | Module ablation, EMNIST only (CIFAR columns dropped earlier). Bold/underline. |
| 6 | `tab:complexity_summary` | Per-round compute + comm cost. threeparttable with 3 footnotes. |
| Notation | `tab:notation` | **In Section 4** (moved forward from appendix). Symbol → meaning, ~13 entries. |

**Removed in earlier sessions** (do NOT add back without checking):
- `tab:simplecnn`, `tab:resnet18cbam`, `tab:backbone_hp` (3 backbone tables, content consolidated into one prose subsubsection `sec:backbone`)
- `tab:baseline_hp` (replaced by in-text "follow original papers" statement)
- `tab:complexity_notation` (merged into `tab:notation` in Section 4)

---

## 8. Bibliography (ref.bib)

22 entries. Added in this session: `lian2017d`, `koloskova2020decentralized`, `kirkpatrick2017overcoming`, `rebuffi2017icarl`, `buzzega2020dark`, `soltoggio2025mosaic`, `li2026collective`, `prorok2025extending`. Pre-existing: FedAvg, FedProx, SCAFFOLD, pFedMe, Clustered-FL, LwF, FedLwF, DCFCL, EMNIST, CIFAR, CBAM, Adam, de2021continual, soltoggio2024collective.

**No fabricated citations** — every entry was either user-supplied or verified via WebSearch / arXiv API. If new refs are added, consider running `/citation-audit` before submission.

---

## 9. Recent work log (this session, chronological highlights)

1. Pulled both repos; figure caption simplification with `threeparttable`.
2. Section 3 rewrite — bridge from Edge-Side CL → DFCL; redundant symbols removed.
3. Section 4 split — Learning Objectives subsection (system-level vs three per-client objectives) + Three-Stage Realisation.
4. Section 5.C split into 3 `\subsubsection`s; hard-coded coefficients (0.4/0.3/0.3, 1/2) → symbols $w_\rho, w_\delta, w_\eta, \beta$ specified in Implementation Details.
5. Section 6 closure: identified $\Loss_i^{\text{plas}} \equiv \Loss_{\text{CE}}$ and $\Loss_i^{\text{stab}} = $ KD+PA+LR+ER sum (new `eq_stab_decomposition`).
6. **Dynamic Directed Collaboration Mask** new subsection (`sec_dynamic_mask`) — formal $M^{(r)}$ definition + three properties (directed / dynamic / continual-aware); body and Section 4 three-stage description updated to surface it.
7. 10-item global cleanup: \mathrm subscripts → \text (138 occ); reduce emph; metrics decoupled from table refs; Per-round trajectory + Emergent CI metric paragraphs deleted; notation moved to Sec 4; Table 2 column reorder (dataset-first); bold-best/underline-2nd across Tables 2/3/4/5; algorithm split into 2 (alg_edge_training + alg_collaboration).
8. Three variants renamed to DynDFCL-I/II/III (in Tables 2 and 3 only).
9. Related Works subsections written (3 × ~0.25 page); 8 new bib entries added.
10. Introduction rewritten — 5 prose paragraphs before the protected "We propose..." contribution block + clean organization paragraph using `\ref{}` to all sections.
11. Two pushes to remote: figure regeneration + paper revision (commits `7144010` on CollEdge, `77395c2` on JSTSP). Last pull: JSTSP `f51c148..5e11896` (user-side hand-edit).

---

## 10. Pending TODO for next session

### Primary task: **De-AI polish**
The new session should focus on removing AI-flavoured writing introduced by this session (and possibly any AI assist that happened previously).

**No dedicated `de-AI` skill exists in the available skill list.** Recommended approach:

#### A. Manual scan + targeted edits (best for control & speed)
Use `Bash + grep` to flag common AI tells, then `Edit` line-by-line. Patterns to scan:

```bash
M=Zirui-Li-IEEE-JSTSP-Special-Issue/main.tex
echo "=== AI cliché words ==="
grep -nE "\b(delve|leverage|landscape|pivotal|intricate|robust|comprehensive|paradigm|holistic|multifaceted|seamlessly|underscores|exemplif)" "$M"
echo "=== Transitional fluff ==="
grep -nE "\b(Moreover|Furthermore|Notably|Additionally|In particular|It is worth noting|It is important to)" "$M"
echo "=== Triadic listing ==="
grep -nE "Three (insights|gaps|families|trajectories|properties|reasons)" "$M"
echo "=== Empty modifiers ==="
grep -nE "\b(uniquely (challenging|positioned)|natural fit|natural extension|rich literature|novel framework|state-of-the-art results)" "$M"
echo "=== Em-dash density (3+ in a paragraph is suspicious) ==="
awk 'BEGIN{RS=""} {gsub(/[^-]/,"",$0); if(length()>5) print NR": "length()" em-dashes"}' "$M"
echo "=== Hedging ==="
grep -nE "\b(begin(s)? to address|opens a path|sets the stage|paves the way)" "$M"
```

Then for each suspected paragraph: read the surrounding context with `Read`, propose a 1-sentence-at-a-time rewrite, get user approval, `Edit`.

**Highest-AI-density sections** (where this session wrote/heavily-edited):
- Introduction (lines ~89-145, rewritten just now)
- Related Works (lines ~150-180, written this session)
- Section 4 Learning Objectives (lines ~207-247)
- Section 5.D Dynamic Directed Mask (lines ~360-385, new subsection)
- Quantitative Results opening (line ~600 area)

**Lower priority** (less likely to be AI-tainted):
- Proofs in appendix (mostly user-written)
- Equation-heavy sections (less prose)
- Specific number-heavy paragraphs (claim-driven, hard to AI-ify)

#### B. `/auto-paper-improvement-loop` with de-AI focus (heavier, more thorough)
2-round GPT-5.4 review → fix → recompile. Override prompt to focus only on writing-style cleanup, not content changes. Caution: GPT itself writes AI-flavoured text; check that round-2 doesn't reintroduce the same tells.

### Secondary tasks (lower priority)
- `/citation-audit` on the 8 new bib entries added this session
- `/kill-argument` before submission (theory/scope-heavy paper)
- Conclusion section (Section 8) is still empty
- `fig:placeholder` (Fig 1, hero figure) still uses AI-generated draft `fig_main.png` — needs a real architecture/workflow figure
- `sec:backbone` reference in complexity section was changed from "Appendix" to "Sec." — verify reads naturally
- The pre-existing `sec:backbone` ref was resolved; no dangling references remain as of last compile

---

## 11. Tooling / environment quick reference

```bash
# Compile (15 pages clean as of last commit)
cd ~/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/3-GitHub/DCFCL/Zirui-Li-IEEE-JSTSP-Special-Issue
latexmk -pdf -interaction=nonstopmode main.tex

# Regenerate figures (after editing plot scripts)
cd ../CollEdge/paper
/opt/miniconda3/envs/fcl/bin/python plot_figures.py
/opt/miniconda3/envs/fcl/bin/python generate_extra_figures.py
# Then cp updated PDFs to ../../Zirui-Li-IEEE-JSTSP-Special-Issue/figures/

# Push (SSH config already set; just `git push`)
git status -s
git add <specific files; never -A>
git commit -m "..." 
git push
```

---

## 12. Hard rules for the next session

1. **DO NOT modify** the 3 contribution items + "We empirically demonstrate..." sentence after `We propose a decentralized federated continual learning framework that`.
2. **DO NOT rename** the disk folder `CollEdge` (only the display name was renamed to DynDFCL).
3. **DO NOT undo** any of the style conventions listed in §4 (mathrm→text, no-textsc, no-emph-on-names, threeparttable for tables, bold/underline ranking, DynDFCL-I/II/III only in Tables 2-3).
4. **DO NOT add fabricated citations**. If new refs are added, verify each via WebSearch / arXiv API / Semantic Scholar before adding to `ref.bib`.
5. **DO NOT amend git commits** unless explicitly asked. Create new commits.
6. **DO NOT push** without explicit user instruction.

---

## 13. One-line summary for the next agent

> Paper is method-renamed (DynDFCL with -I/II/III variants), structure-cleaned, figure-regenerated, table-formatted, related-work-filled, and introduction-written. Compiles to 15 pages clean. Next task is a de-AI writing-style pass — no skill exists for this; recommend manual grep + targeted Edit, prioritising Introduction / Related Works / Sec 4.A / Sec 5.D / Quantitative Results opening.

---

*End of handoff document.*
