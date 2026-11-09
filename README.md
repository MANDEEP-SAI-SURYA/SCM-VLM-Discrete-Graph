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
