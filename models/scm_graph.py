import torch
import torch.nn as nn


class EdgeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


GEOMETRY_DIM = 9


def pairwise_box_features(boxes: torch.Tensor) -> torch.Tensor:
    """Geometry of every ordered object pair, for boxes [B, N, 4] = (x0, y0, x1, y1) scaled
    to 0-1.  Feature [b, i, j] describes object j relative to object i:
    centre offset (dx, dy), horizontal / vertical gap (0 when the boxes overlap on that
    axis), IoU, share of i covered by j, share of j covered by i, log width and height
    ratio.  Returns [B, N, N, GEOMETRY_DIM]."""
    x0, y0, x1, y1 = boxes.unbind(-1)
    w = (x1 - x0).clamp(min=1e-4)
    h = (y1 - y0).clamp(min=1e-4)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dx = cx[:, None, :] - cx[:, :, None]
    dy = cy[:, None, :] - cy[:, :, None]
    gap_x = (torch.maximum(x0[:, :, None], x0[:, None, :]) - torch.minimum(x1[:, :, None], x1[:, None, :])).clamp(min=0)
    gap_y = (torch.maximum(y0[:, :, None], y0[:, None, :]) - torch.minimum(y1[:, :, None], y1[:, None, :])).clamp(min=0)
    inter_w = (torch.minimum(x1[:, :, None], x1[:, None, :]) - torch.maximum(x0[:, :, None], x0[:, None, :])).clamp(min=0)
    inter_h = (torch.minimum(y1[:, :, None], y1[:, None, :]) - torch.maximum(y0[:, :, None], y0[:, None, :])).clamp(min=0)
    inter = inter_w * inter_h
    area = w * h
    union = area[:, :, None] + area[:, None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    cover_i = inter / area[:, :, None]          # how much of i is covered by j
    cover_j = inter / area[:, None, :]          # how much of j is covered by i
    log_w = torch.log(w[:, None, :] / w[:, :, None])
    log_h = torch.log(h[:, None, :] / h[:, :, None])
    return torch.stack([dx, dy, gap_x, gap_y, iou, cover_i, cover_j, log_w, log_h], dim=-1)


CONTACT_GAP = 0.02       # boxes closer than 2% of the image size count as touching


def contact_targets(boxes: torch.Tensor, gap: float = CONTACT_GAP) -> torch.Tensor:
    """[N, N] 1 where two boxes (x0, y0, x1, y1 in 0-1) overlap or nearly touch (a physical-
    interaction prior, e.g. person-bicycle, hand-umbrella), 0 otherwise.  Diagonal is 0."""
    x0, y0, x1, y1 = boxes.unbind(-1)
    gap_x = torch.clamp(torch.maximum(x0[:, None], x0[None, :]) - torch.minimum(x1[:, None], x1[None, :]), min=0)
    gap_y = torch.clamp(torch.maximum(y0[:, None], y0[None, :]) - torch.minimum(y1[:, None], y1[None, :]), min=0)
    touching = ((gap_x <= gap) & (gap_y <= gap)).float()
    touching.fill_diagonal_(0.0)
    return touching


class SCMGraph(nn.Module):

    def __init__(self, object_dim, question_dim, hidden_dim=512, top_k=2, threshold=0.5, mode="legacy",
                 geometry=False, relation_types=1, temperature=1.0, knowledge=False):
        """mode:
        * "soft"   - every node attends over its top_k most likely parents; the weights are
                     the edge probabilities normalised to sum to 1 (no hard cut-off).  The
                     graph therefore cannot collapse to empty, and gradients always reach
                     the edge scorer.  Which parents, and how strongly, is learned (and
                     supervised by box contact in train.py).
        * "legacy" - original behaviour: top_k, hard threshold, rows normalised to sum 1.
                     Edges below the threshold get no gradient; checkpoints trained this
                     way ended with an empty graph.
        * "discrete" - Discrete Causal Graph Sampling (see _discrete_parents): every node
                     chooses its parents as a discrete set (Gumbel top-k during training,
                     deterministic top-k at test time) among the other objects AND a learnable
                     "no parent" (exogenous) option.  Unlike "soft", a node can be a root
                     cause, so the graph can be a true DAG.
        """
        super().__init__()
        if mode not in ("soft", "legacy", "discrete"):
            raise ValueError("mode must be 'soft', 'legacy' or 'discrete'")
        self.mode = mode
        # Gumbel-softmax temperature (annealed by train.py; also used at test time so the
        # edge weights keep the sharpness they were trained with)
        self.temperature = float(temperature)
        # sample=True: stochastic Gumbel sampling while training; test time is deterministic
        self.sample = True
        # discrete graph: chosen parents below this weight are dropped (exact zeros)
        self.min_edge_weight = 1e-3
        # geometry=True: the edge scorer also sees the pair's box geometry (pairwise_box_features)
        self.geometry = geometry

        self.project_question = nn.Linear(question_dim, object_dim)
        self.edge_mlp = EdgeMLP(object_dim * 5, hidden_dim)
        if geometry:
            # separate low-dimensional geometry branch: its logit is added to the appearance
            # logit, so simple spatial rules (touching, above, inside) generalise instead of
            # being memorised through the 1280-d appearance input
            self.geometry_mlp = nn.Sequential(
                nn.Linear(GEOMETRY_DIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
            )

        # relation_types > 1: the scorer also predicts a softmax over R learned relation types
        # per pair (used by relation-aware messages: one message transform per type)
        self.relation_types = relation_types if geometry else 1
        if self.relation_types > 1:
            self.type_pair = nn.Linear(object_dim, 32)
            self.type_mlp = nn.Sequential(
                nn.Linear(GEOMETRY_DIM + 32, 64), nn.ReLU(), nn.Linear(64, self.relation_types)
            )

        if mode == "discrete":
            # exogeneity scorer: logit of "this object has NO parent in the scene" (a root
            # cause), computed from the object and the question
            self.null_mlp = nn.Sequential(
                nn.Linear(object_dim * 2, 128), nn.ReLU(), nn.Linear(128, 1)
            )
            # causal-order scorer: an object can only be a parent of objects later in the order,
            # which makes every sampled graph a DAG
            self.order_mlp = nn.Sequential(
                nn.Linear(object_dim * 2, 128), nn.ReLU(), nn.Linear(128, 1)
            )

        # knowledge=True: symbolic class-pair prior from a knowledge base (models/knowledge.py)
        # added to the edge logits; needs the DETR class ids of the objects
        self.knowledge_prior = None
        if knowledge:
            from models.knowledge import KnowledgePrior

            self.knowledge_prior = KnowledgePrior()

        self.top_k = top_k
        self.threshold = threshold

    def forward(self, objects, question, boxes=None, class_ids=None):
        """boxes: [B, N, 4] or [N, 4] (x0, y0, x1, y1 scaled to 0-1); used when geometry=True.
        class_ids: [B, N] or [N] DETR class ids; used by the knowledge prior."""

        if objects.dim() == 2:
            objects = objects.unsqueeze(0)
        if question.dim() == 1:
            question = question.unsqueeze(0)

        B, N, D = objects.shape

        # -------------------------
        # question projection
        # -------------------------
        q = self.project_question(question)

        # -------------------------
        # pair construction
        # -------------------------
        source = objects.unsqueeze(2).expand(B, N, N, D)
        target = objects.unsqueeze(1).expand(B, N, N, D)
        q_pair = q[:, None, None, :].expand(B, N, N, D)

        edge_input = torch.cat([source, target, target - source, target * source, q_pair], dim=-1)

        scores = self.edge_mlp(edge_input).squeeze(-1)
        geometry = None
        relation_probs = None
        if self.geometry:
            if boxes is None:
                raise ValueError("This graph uses box geometry: pass the object boxes.")
            if boxes.dim() == 2:
                boxes = boxes.unsqueeze(0)
            geometry = pairwise_box_features(boxes.to(objects.dtype).to(objects.device))
            scores = scores + self.geometry_mlp(geometry).squeeze(-1)
            if self.relation_types > 1:
                pair = self.type_pair(target * source)
                relation_probs = torch.softmax(self.type_mlp(torch.cat([geometry, pair], dim=-1)), dim=-1)

        knowledge_logits = None
        if self.knowledge_prior is not None and class_ids is not None:
            if class_ids.dim() == 1:
                class_ids = class_ids.unsqueeze(0)
            knowledge_logits = self.knowledge_prior(class_ids.to(objects.device))
            scores = scores + knowledge_logits

        # -------------------------
        # remove self loops
        # -------------------------
        eye = torch.eye(N, device=objects.device, dtype=torch.bool)
        scores = scores.masked_fill(eye.unsqueeze(0), -1e9)

        probs = torch.sigmoid(scores)

        if self.mode == "discrete":
            sampled = self._discrete_parents(scores, objects, q)
            adjacency = sampled["adjacency"]
            return {
                "candidate_logits": sampled["candidate_logits"],  # [B, N, N + 1], last = "no parent"
                "allowed_parents": sampled["allowed"],            # [B, N, N] causal-order mask
                "causal_order": sampled["order"],                 # [B, N] order scores
                "adjacency": adjacency,
                "hard_adjacency": (adjacency.detach() > 0).float(),
                "probs": probs,
                "scores": scores,
                "null_weight": sampled["null_weight"],            # [B, N] weight of "no parent"
                "relation_probs": relation_probs,
                "geometry": geometry,
            }

        # -------------------------
        # SAFE TOP-K (IMPORTANT FIX)
        # -------------------------
        adjacency = torch.zeros_like(probs)

        # FIX: prevent k out of range
        k = min(self.top_k, max(N - 1, 1))

        if N > 1:
            topk_vals, topk_idx = torch.topk(probs, k=k, dim=-1)
            adjacency.scatter_(-1, topk_idx, topk_vals)

        # -------------------------
        # thresholding (legacy only; soft mode keeps the weighted top-k edges)
        # -------------------------
        if self.mode == "legacy":
            adjacency = adjacency * (adjacency > self.threshold).float()

        # remove self loops again
        adjacency = adjacency * (~eye.unsqueeze(0)).float()

        # -------------------------
        # normalization
        # -------------------------
        adjacency = self.normalise(adjacency)

        return {
            "adjacency": adjacency,
            "hard_adjacency": (adjacency > 0).float(),
            "probs": probs,
            "scores": scores,
            "relation_probs": relation_probs,     # [B, N, N, R] or None
            "geometry": geometry,                 # [B, N, N, GEOMETRY_DIM] or None
        }

    def _discrete_parents(self, scores, objects, q) -> dict:
        """Discrete Causal Graph Sampling with a learned causal order.

        1. Causal order.  Every object gets an order score r_i = order_mlp([o_i ; q]).  While
           training r is perturbed with Gumbel noise (a sampled permutation, i.e. a sampled
           causal order); at test time it is deterministic.  Object j may only be a parent of
           object i if it comes earlier in the order (r_j > r_i).  Every sampled graph is
           therefore a DAG by construction - acyclicity is guaranteed, not just penalised -
           and the first object in the order is a root.

        2. Parent sets.  Candidates of node i: the objects allowed by the order (edge logits
           s_ij) and an exogenous "no parent" option (logit from null_mlp).  With temperature tau:

               training:  z = s + log m + Gumbel noise      test:  z = s + log m
               soft       = softmax(z_relaxed / tau)        (m relaxed: sigmoid((r_j - r_i) / tau))
               hard       = top-k of z, weights renormalised over the chosen candidates
               weights    = hard + (soft - stopgrad(soft))  straight-through estimator

           The forward pass uses the discrete, order-respecting parent set; the gradient is that
           of the relaxed distribution, so edge scores AND order scores keep learning.  The
           "no parent" weight is dropped: rows of A sum to 1 - w_i0.

        Returns adjacency [B, N, N], no-parent weight [B, N], candidate logits [B, N, N + 1]
        (order-masked; used by the parent-selection supervision), the hard order mask
        allowed[b, i, j] = "j may be a parent of i" and the order scores [B, N]."""
        B, N, D = objects.shape
        context = torch.cat([objects, q[:, None, :].expand(B, N, D)], dim=-1)
        null_logit = self.null_mlp(context)                                   # [B, N, 1]
        order = self.order_mlp(context).squeeze(-1)                           # [B, N]
        sampling = self.training and self.sample
        if sampling:
            uniform = torch.rand_like(order).clamp(1e-9, 1.0 - 1e-9)
            order = order - torch.log(-torch.log(uniform))
        tau = max(self.temperature, 1e-3)
        # ties are broken by index so the order is always a strict total order
        tie_break = torch.arange(N, device=order.device, dtype=order.dtype) * 1e-6
        ranked = order - tie_break
        allowed = ranked[:, None, :] > ranked[:, :, None]                     # [B, i, j]: j before i
        hard_mask = torch.where(allowed, 0.0, -1e9).to(scores.dtype)
        soft_mask = torch.nn.functional.logsigmoid((order[:, None, :] - order[:, :, None]) / tau)
        order_term = hard_mask + (soft_mask - soft_mask.detach()) if self.training else hard_mask
        edge_logits = scores + order_term
        candidates = torch.cat([edge_logits, null_logit], dim=-1)             # [B, N, N + 1]
        k = min(self.top_k, N)                                                # N - 1 objects + "none"
        if sampling:
            uniform = torch.rand_like(candidates).clamp(1e-9, 1.0 - 1e-9)
            perturbed = candidates - torch.log(-torch.log(uniform))
        else:
            perturbed = candidates
        soft = torch.softmax(perturbed / tau, dim=-1)
        chosen = torch.zeros_like(soft).scatter_(-1, torch.topk(perturbed, k=k, dim=-1).indices, 1.0)
        chosen[..., :N] = chosen[..., :N] * allowed.to(chosen.dtype)        # never an order-violating edge
        hard = soft * chosen
        hard = hard / hard.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        # a chosen parent with negligible weight (possible at low temperature) is not an edge
        hard = hard * (hard > self.min_edge_weight).to(hard.dtype)
        weights = hard + (soft - soft.detach()) if self.training else hard
        return {"adjacency": weights[..., :N], "null_weight": weights[..., N], "candidate_logits": candidates,
                "allowed": allowed, "order": order}

    def normalise(self, adjacency):
        """Row normalisation used by the message passing (also used after interventions).
        Discrete graphs keep their weights (rows sum to 1 - no-parent weight); removing a
        parent removes its weight instead of re-scaling the remaining parents."""
        if self.mode == "discrete":
            return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return adjacency / (adjacency.sum(dim=-1, keepdim=True) + 1e-8)
