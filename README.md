# SCM-VLM with a Discrete Causal Graph Layer

> **This is a separate copy of the SCM-VLM project.** The original project is unchanged.
> It adds the **Discrete Causal Graph Sampling layer** (`--graph-mode discrete`) and,
> further down, **adaptive loss weighting + curriculum** and **neuro-symbolic knowledge**.
> These come from the paper's future work, which says:
> *"While the present architecture uses soft adjacency constrained attention, future research may
> work on exploring discrete graph sampling techniques and probabilistic causal discovery methods
> to improve structural interpretability and stability."*

## New: Discrete Causal Graph Sampling layer

**Problem with the soft graph.** Every object always receives exactly k parents (soft top-k
attention). A graph in which every node has a parent always contains a cycle, so the NOTEARS
acyclicity term can never reach zero: the "causal graph" is never actually a DAG, there are no
root causes, and edges are dense.

**The layer** (`models/scm_graph.py`, `SCMGraph(mode="discrete")`):

1. **Learned causal order.** Each object gets a question-conditioned order score
   `r_i = MLP([o_i ; q])`, sampled with Gumbel noise during training (a sampled causal order).
   An object can only choose parents that come earlier in the order, so **every graph is a DAG
   by construction** (acyclicity is guaranteed, not just penalised) and the first object is a root.
2. **Discrete parent sets with an exogenous option.** Each object chooses its top-k parents among
   the allowed objects **and a learnable "no parent" option** (root cause). Training: Gumbel top-k
   with a straight-through estimator (discrete forward pass, relaxed gradient to every edge and
   order score), temperature annealed from `--tau-start` (1.0) to `--tau-end` (0.2). Test time:
   deterministic.
3. **Causal and diagnostic messages.** Pearl-style belief propagation on the DAG: messages flow
   parent -> child (causal) and child -> parent (diagnostic) along the same edges, with separate
   transforms (`models/causal_attention.py`). A do() intervention that cuts the edges into k
   removes them in both directions.
4. **Parent-selection supervision.** The parent choice is supervised by the contact prior: an
   object should choose the touching objects that come before it, or "no parent" when there is
   none (`parent_selection_loss` in `train.py`). The sparsity penalty is not used (top-k parent
   sets are sparse by construction; with it, every object chose "no parent").

New metrics (`utils/evaluation.py`, `graph_structure`): acyclicity h(A), DAG rate, share of
root-cause objects, parents per object, structural stability (edge-set overlap under small input
noise), "no parent" weight. They are reported for every graph model, so the soft and discrete
graphs can be compared directly.

```bash
python train.py --samples 30000 --epochs 10 --tag soft                         # original soft graph
python train.py --samples 30000 --epochs 10 --graph-mode discrete --tag discrete
python relation_analysis.py --checkpoint checkpoints/scm_vlm_discrete.pt --samples 30000 --tag discrete
```

| Flag | Default | Meaning |
|---|---|---|
| `--graph-mode discrete` | | the new layer |
| `--tau-start`, `--tau-end` | 1.0, 0.2 | Gumbel temperature annealing (geometric, per epoch) |
| `--graph-top-k` | 3 | candidates chosen per object (discrete: including "no parent") |

Checks run on fake data (stubbed DETR/BERT, 3 seeds; **not evidence about real VQA**):

| | soft graph | discrete causal graph |
|---|---|---|
| acyclic graphs (DAG) | 3 % | **100 %** (guaranteed) |
| acyclicity h(A) | 0.53 | **0.00** |
| root-cause objects | 10 % | 56 % |
| parents per object | 1.39 | 0.46 |
| structural stability | 0.99 | 1.00 |
| factual accuracy | 92.8 ± 0.8 | 90.6 ± 3.9 |
| counterfactual accuracy | 89.2 ± 1.4 | 89.2 ± 3.7 |

Real accuracy has to be measured on VQA v2 (Colab notebook, section 12).

## Round 2: fixes after the first real VQA v2 run

The first Kaggle run (VQA v2, 30,000 questions, 10 epochs, one seed) found three problems.
This section describes the fixes.

**1. The discrete graph collapsed to an empty graph at test time.** On real data, every object chose
"no parent" (root_fraction 100 %, parents_per_node 0), so "100 % DAG" was trivially true.

* **Cause.** While training, Gumbel noise often lets an edge beat a slightly stronger "no parent"
  logit, so the training log still showed edges. At test time there is no noise and the
  temperature is low, so "no parent" wins every time and the graph is empty.
* **Fix.** A collapse guard (`--root-target 0.5`, default on for the discrete graph) puts a hinge
  penalty on the "no parent" probability of the **noise-free, test-time** graph. Below 50 % roots it
  costs nothing; above, it pushes edges back. The log now prints this test-time no-parent weight
  and warns when the graph empties.
* **Fake-data check.** Without parent supervision the old code reproduces the exact real-data
  symptom: 100 % roots, 0 parents per object. With the guard: 28 % roots, 0.91 parents per object.
* **No cost with supervision on.** Over 3 seeds: factual 87.2 ± 6.3 vs 88.9 ± 1.0, counterfactual
  87.8 ± 2.3 vs 81.4 ± 9.8. None of the paired differences is significant except one seed in favour
  of the guard.

**2. Symbolic fusion needs an honest metric.** Existence / counting what-ifs are answered by the
rules by construction. The evaluation now also reports:

* `overall_neural_only`: the same model with the rule answer removed;
* `symbolic_coverage`: the share of questions the rules could answer;
* `without_rule_answer`: accuracy on the questions and what-ifs the rules could not answer.

On fake data the neural part alone scores far lower (factual 65 vs 90). With fusion, the network
leans on the rules, so report both numbers.

**3. One seed, no significance tests.** `compare_runs.py`:

* groups seeds (`soft`, `soft_s1`, `soft_s2`) into mean ± std;
* tests every run against the baseline of the same seed with an exact McNemar test, on identical
  held-out questions and counterfactual examples, plus a bootstrap CI of the difference.

```bash
python compare_runs.py --baseline soft soft soft_s1 soft_neurosym soft_neurosym_s1 discrete discrete_s1
```

**New configuration.** `soft_neurosym` adds the knowledge prior + symbolic fusion to the soft graph,
which has working edges and the best replacement score.

`kaggle_train.ipynb` section 5 runs round 2:

1. seed 42: `soft`, `soft_neurosym`, `discrete`;
2. extra seeds;
3. the comparison.

## New: adaptive loss weighting, curriculum and neuro-symbolic knowledge

Two more items from the paper's Future Work. All of them are **off by default**: without the
new flags, training is the same as before, and old checkpoints still load.

### 1. Adaptive loss weighting and curriculum (`losses/balancing.py`)

> *"... adaptive weighting mechanisms for dynamically balancing these objectives ... curriculum
> based optimization strategies and staged regularization scheduling ..."*

The training loss is now a set of separate terms (task, counterfactual, relation supervision,
acyclicity, sparsity, entropy). `LossBalancer` combines them:
`total = sum_k w_k(t) * c_k(t) * lambda_k * L_k`.

* `--loss-weighting uncertainty`: learned homoscedastic-uncertainty weights for the three
  objectives (task, cf, edge), following Kendall, Gal & Cipolla (CVPR 2018). The term is
  `exp(-s_k) * L_k + s_k / 2`, so noisy objectives get down-weighted automatically.
* `--loss-weighting dwa`: Dynamic Weight Average (Liu, Johns & Davison, CVPR 2019). An objective
  whose loss has stopped falling gets more weight.
* The structural constraints (acyclicity, sparsity, entropy) keep fixed weights. If they were
  learned, the model could simply switch them off.
* `--curriculum staged` (staged regularisation):
  * a warm-up stage (`--warmup-frac`, default 20 % of the epochs) trains without the causal
    constraints and without the counterfactual loss, so the model learns semantics first;
  * then those terms ramp in linearly (`--ramp-frac`, default 30 %).
* `--curriculum full` adds a competence-based data curriculum (Platanios et al., NAACL 2019):
  * epoch t trains only on the easiest `c(t)` share of the questions, where
    `c(t) = sqrt(c0^2 + (1 - c0^2) t / T)` and `c0 = --competence-start` (default 0.3);
  * difficulty = question length + relational question + answer type (yes/no < other < number);
  * counterfactual examples are ordered too: irrelevant < existence < counting < replacement.
* The training log and `training_history_<tag>.csv` show the weights, the stage and the
  competence for every epoch.

### 2. Neuro-symbolic knowledge (`models/knowledge.py`, `models/symbolic.py`)

> *"... integrating external symbolic knowledge bases or neuro symbolic reasoning modules ..."*

* **Knowledge base `K[a, b, r]`** for every ordered pair of COCO classes (a, b):
  * relations: co-occurrence PMI, touches, overlaps, above, below, inside, near;
  * estimated from the COCO instance annotations, with Bayesian smoothing. **The validation
    images are excluded**, so nothing leaks into the evaluation;
  * an `external` channel holds triples from any TSV file (`subject<TAB>relation<TAB>object<TAB>confidence`),
    e.g. ConceptNet, Visual Genome relationships or hand-written rules;
  * every entry can be printed as a symbolic triple, such as `(person, above, bicycle) 0.98`.
* **Knowledge prior on the edges** (`--knowledge-prior builtin`): for a candidate edge j -> i,
  the graph adds `gate * MLP([K[c_i, c_j] ; K[c_j, c_i]])` to the edge logit.
  * The MLP and the gate are learned; K is a buffer, saved in the checkpoint.
  * Works with the soft and the discrete graph.
* **Symbolic answer fusion** (`--symbolic-fusion`):
  * a rule-based reasoner (`models/scene_answer.py`) answers counting, existence, colour and
    spatial questions from the detected scene;
  * its answer becomes evidence `e` over the answer vocabulary, fused as
    `logits' = logits + softplus(w^T q + b) * e`;
  * the gate depends on the question, so the model learns which question types to trust the
    rules for. Questions the rules cannot parse leave the neural answer unchanged;
  * under do() the symbolic answer is recomputed on the edited scene. Removing the only dog turns
    "Is there a dog?" into "no" in the symbolic path too. This holds in training, in the
    evaluation and in `whatif_qa.py`.

```bash
python build_knowledge_base.py --coco-instances coco/annotations/instances_train2014.json --exclude-val
python train.py --samples 30000 --epochs 10 --graph-mode discrete \
    --loss-weighting uncertainty --curriculum full --knowledge-prior builtin --symbolic-fusion --tag fw_all
```

| Flag | Default | Meaning |
|---|---|---|
| `--loss-weighting {fixed,uncertainty,dwa}` | fixed | adaptive objective weights |
| `--curriculum {none,staged,full}` | none | staged regularisation (+ data curriculum) |
| `--warmup-frac`, `--ramp-frac` | 0.2, 0.3 | stage lengths (share of epochs) |
| `--competence-start` | 0.3 | share of the easiest questions in epoch 1 |
| `--knowledge-prior {none,builtin}` | none | class-pair knowledge prior on the edges |
| `--knowledge-file` | | saved knowledge base (`*.pt`) instead of building it |
| `--external-knowledge` | | TSV of extra triples |
| `--symbolic-fusion` | off | gated rule-based answer fusion |

Checks on fake data (stubbed DETR/BERT, discrete graph, 6 epochs, 60 validation questions and
68 validation counterfactual examples). **These are not evidence about real VQA.**

| run (3 seeds, mean ± std) | factual acc. | counterfactual acc. | DAG |
|---|---|---|---|
| discrete (baseline) | 88.9 ± 1.0 | 81.4 ± 9.8 | 100 % |
| + uncertainty weighting + full curriculum | 75.0 ± 2.9 | 76.5 ± 10.3 | 100 % |
| + knowledge prior + symbolic fusion | 93.3 ± 2.9 | 99.5 ± 0.8 | 100 % |
| + all four | 93.9 ± 1.9 | 100.0 ± 0.0 | 100 % |

Single components, seed 42 only. The baseline with that seed scores 88.3 / 89.7.

| component | factual | counterfactual |
|---|---|---|
| uncertainty weighting | 88.3 | 86.8 |
| DWA | 83.3 | 88.2 |
| staged regularisation | 85.0 | 70.6 |
| full curriculum | 81.7 | 92.6 |
| knowledge prior | 83.3 | 82.4 |
| symbolic fusion | 85.0 | 100.0 |

How to read this:

* **Symbolic fusion causes almost all of the gain.** The fake questions are templates
  ("Is there a dog?", "How many cars ...") that the rules parse perfectly, so this is a best case.
  On real VQA the rules apply to fewer questions and the detections are noisy. The learned gate is
  meant to handle that, but the size of the effect has to be measured on VQA v2.
* **The knowledge prior has no measurable effect here.** The fake knowledge base comes from only
  108 annotated images and 6 classes. Its value should come from the full COCO statistics.
* **Adaptive weighting and the curriculum do not help in 6 epochs.** Staged regularisation switches
  the counterfactual loss off in epoch 1 and ramps it in over epochs 2–3, so a short run loses
  counterfactual training. The data curriculum also shows the model fewer questions early on.
  These methods are designed for longer training, so test them with the 10-epoch Colab runs
  (notebook, section 13).
* With 60 / 68 validation items, one question is 1.7 / 1.5 points, and the seed-to-seed spread of
  the counterfactual accuracy is about ±10 points. Single-seed differences are noise.

---

# SCM-VLM Research Prototype

Implementation of the core architecture proposed in:

**Intervention Aware Multimodal Reasoning via Structural Causal Graph Integration in Vision Language Transformers**

This version follows the requested research-prototype layout and includes:

- DETR-based object extraction
- BERT-based question encoding
- learnable SCM graph construction
- DAG regularization
- multi-head causal attention
- intervention operators with descendant recomputation
- counterfactual-ready training pipeline
- evaluation utilities
- graph visualization

## No GPU on Colab? Use Kaggle (free) and download the data directly

`kaggle_train.ipynb` runs the whole pipeline on a free Kaggle GPU: clone the repo, download the
data, run the tests, do the training runs and compare them. It needs no Kaggle dataset, because
`download_data.py` fetches everything from the official VQA / COCO sites:

```bash
python download_data.py --images 8000     # VQA v2 + COCO annotations + a fixed 8000-image subset (~1.6 GB)
python validate_dataset.py
```

The image subset is fixed (seed 0), so all runs use the same images and the same split.
Interrupted downloads can be restarted, and files already on disk are skipped. This also works
on a laptop.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train

DETR and BERT are frozen; their outputs are cached in `cache/` on the first epoch,
so later epochs are much faster.

```bash
python train.py                               # 20k questions, 10 epochs, top-3129 answers
python train.py --samples 3000 --epochs 20    # smaller run
python train.py --cf-mode none                # ablation: no counterfactual loss
```

Accuracies in `results/final_metrics.csv` and `results/training_history.csv` are
measured on a held-out validation split (separate images), not on training data.

**Relation graph.** Each object attends over its 3 most likely parents (soft edges, no hard
threshold, so the graph cannot collapse to empty). The edge scorer combines appearance
(DETR features + question) with a separate box-geometry branch (relative position, gaps,
overlap, size ratio), and edges are weakly supervised: objects whose boxes touch should be
linked. Each epoch logs the parent weight per object and, on validation images, the edge
probability for touching vs. distant objects. Ablation: `--no-edge-supervision`;
original graph: `--graph-mode legacy`.

> Checkpoints from before this change load with the old graph (hard 0.5 threshold), which
> ended with no edges at all. Retrain to get working relations.

> Checkpoints trained before the answer-head fix still load, but that head never
> saw the question, so it answers every question the same way (mostly "yes"/"no").
> Retrain to get question-dependent answers.

## Relation-aware model with counterfactual supervision (checkpoint format 3)

Earlier experiments showed the model did not use relations to answer: the no-graph ablation was
as accurate as the full model, scrambling the graph changed only 6 % of answers, and the old
counterfactual loss (KL to the factual prediction under random noise) trained the model to
*ignore* interventions. This version changes:

| Part | Change | Files |
|---|---|---|
| DETR features | Each detection's embedding now comes from **the query that produced it** (the old code indexed the decoder states by the position in the filtered list). Features are DETR's `last_hidden_state`, the class-head input in every transformers version (`decoder_hidden_states[-1]` is not: transformers 5.x records it before the final layernorm). Feature cache is versioned (`FEATURE_CACHE_VERSION = 2`), old cache files are rebuilt. | `models/vision_encoder.py`, `configs/config.py`, `train.py` |
| Counterfactual supervision | Examples with known answers generated from VQA v2 + COCO instances: **existence** (remove every node of the class -> "no"; remove another class -> "yes"), **counting** (1-5 matched instances; remove one -> count - 1), **replacement** (VQA answer is a COCO class matching one node; do(X_k = x') with x' steered to a class of the same super-category -> new class), **irrelevant** (remove/replace an object not mentioned and not touching a mentioned one -> answer unchanged; during training also skipped when the node is a parent/child of a mentioned object in the current graph). Loss: `L = L_task + lambda_cf * CE(Y_do, Y_cf) + graph regularisers`. | `datasets/counterfactual.py`, `train.py` |
| Relation-aware messages | `m_ij = W_r [h_j ; MLP(g_ij)]`, `g_ij` = pair box geometry; optional R learned relation types. | `models/causal_attention.py`, `models/scm_graph.py` |
| Node identity | Node input = `[DETR embedding ; class embedding ; box (x0, y0, x1, y1, w, h)]`; replace interventions change the class embedding too. | `models/scm_vlm.py`, `configs/coco_labels.py` |
| Answer head | Reads the final graph-updated states; `--no-residual-last` removes the residual bypass of the last message step. | `models/scm_vlm.py` |
| Interventions | `SCMVLM.intervene()`: *remove* deletes nodes and their edges (rest of the graph kept, rows renormalised); *replace* is a strict do() (incoming edges cut, rest kept, only descendants change). `mode="recompute"` rebuilds the graph instead. Boxes and class ids stay aligned. | `models/scm_vlm.py`, `models/do_intervention.py` |

```bash
python train.py --samples 30000 --epochs 10 --tag full                      # full model
python train.py --samples 30000 --epochs 10 --graph-mode none --tag nograph # no graph, same inputs + CF training
python train.py --samples 30000 --epochs 10 --cf-mode invariance_old --tag oldcf   # previous loss
python relation_analysis.py --checkpoint checkpoints/scm_vlm_full.pt --samples 30000 --tag full
```

New `train.py` options:

| Flag | Default | Meaning |
|---|---|---|
| `--cf-mode {none,invariance_old,supervised}` | `supervised` | counterfactual loss |
| `--lambda-cf` | 1.0 (0.1 for `invariance_old`) | weight of the counterfactual loss |
| `--cf-ratio` | 1.0 | counterfactual examples per VQA question per epoch |
| `--coco-instances` | `coco/annotations/instances_train2014.json` | COCO annotations |
| `--graph-mode {soft,legacy,none}` | `soft` | `none` = ablation without graph and messages |
| `--relation-types R` | 1 | learned relation types (one message transform each) |
| `--no-node-identity`, `--no-relation-messages` | off | ablations |
| `--no-residual-last` | off | remove the residual bypass of the last message step |
| `--no-irrelevant-graph-check` | off | keep 'irrelevant' examples even when related in the learned graph |
| `--seed` | 42 | initialisation / sampling seed (the train/val split is always the same) |
| `--tag NAME` | none | writes `checkpoints/scm_vlm_NAME.pt` and `results/*_NAME.*` |
| `--skip-final-eval` | off | skip the full evaluation after training |

Evaluation (end of `train.py`, or `relation_analysis.py`; held-out images, same split):
`results/eval_summary_<tag>.json` with factual accuracy (overall, per VQA answer type,
relational vs. non-relational questions), counterfactual accuracy per type (all models are
scored on the same examples), the graph knock-out (learned graph / messages off / scrambled
graph: accuracy, % answers changed, exact McNemar p-values), and relation probes with
bootstrap 95 % CIs (edge probability for touching vs. distant objects, % answers changed by
relevant vs. irrelevant interventions, answer-head attention on the answer object).
Per-question and per-example predictions: `results/eval_factual_<tag>.csv`,
`results/eval_counterfactual_<tag>.csv`.

Checkpoints store their architecture (`format_version` 3); older checkpoints still load with
the architecture they were trained with, and a note that they used the misaligned features.

## Run on any image

```bash
python main.py --image path/to/photo.jpg --question "What is the man holding?"
python evaluate.py --image path/to/photo.jpg --question "Is the light red?"
python scene_graph_qa.py --image path/to/photo.jpg --question "What is he riding?"
```

## What-if (counterfactual) questions with do()

```bash
python whatif_qa.py --image photo.jpg --whatif "What if the bicycle were a motorcycle? What is the man riding?"
python whatif_qa.py --image photo.jpg --whatif "What if there were no umbrella, what is the person holding?"
python whatif_qa.py --image photo.jpg --whatif "What if the traffic light turns green"
python whatif_qa.py --image photo.jpg --whatif "What if there was a dog, is there a dog?"
python whatif_qa.py --image photo.jpg --question "How many cars are there?" --do "remove:car"
python whatif_qa.py --image photo.jpg --question "What is he riding?" --do "replace:#3->motorcycle"
python whatif_qa.py --image photo.jpg --whatif "What if the cars were removed, is the road busy?" --use-llm
```

Two modes:

* `--mode recompute` (default) - edit the scene, then the model **re-calculates all relations**
  between the objects and answers the question with exactly the same pipeline as a normal
  question. The output lists new, lost and changed relations and the changed object's links
  before and after.
* `--mode do` - strict SCM intervention do(X_k = x'): the learned graph is kept, only the edges
  into the changed object are cut, and only its descendants are recomputed.

```bash
python whatif_qa.py --image photo.jpg --whatif "What if the bicycle were a motorcycle? What is the man riding?"
python whatif_qa.py --image photo.jpg --mode do --whatif "What if the bicycle were a motorcycle? What is the man riding?"
```

Supported interventions: **remove** an object, **replace** it with another COCO class,
change an **attribute** (e.g. colour), or **add** an object. Answers are given for the real and the intervened scene, from the
scene graph (counting, existence, colour, relations such as riding/holding/sitting on)
and from the SCM-VLM classifier. A before/after causal-graph figure is saved to
`results/whatif/`.

## Tests

```bash
python -m pytest tests
```

## Google Collab NoteBook
https://colab.research.google.com/drive/1gc8_qWXL_sxYWe8thv1v4v5BfdpCYWNU?usp=sharing
