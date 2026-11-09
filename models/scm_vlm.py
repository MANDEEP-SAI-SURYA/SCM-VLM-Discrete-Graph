import torch
import torch.nn as nn

from configs.coco_labels import NO_OBJECT_ID, NUM_CLASS_IDS
from configs.config import (
    HIDDEN_DIM,
    NUM_HEADS,
    OBJECT_DIM,
    QUESTION_DIM,
)

from models.causal_attention import CausalAttentionBlock
from models.intervention import InterventionEngine, find_descendants
from models.scm_graph import SCMGraph, pairwise_box_features
from models.text_encoder import TextEncoder
from models.vision_encoder import VisionEncoder

CLASS_EMBED_DIM = 64
BOX_FEATURE_DIM = 6          # x0, y0, x1, y1, w, h  (scaled to 0-1)

# Architecture of checkpoints saved before format_version 3 (they have no "architecture").
LEGACY_ARCHITECTURE = {
    "node_identity": False,
    "relation_messages": False,
    "relation_types": 1,
    "residual_last_block": True,
}


class QuestionAwareAnswerHead(nn.Module):
    """Y = Decoder(h, q).

    1. question-guided attention pooling over the FINAL (graph-updated) node states
           alpha_i = softmax_i( <W_k h_i, W_q q> / sqrt(d) )
           v       = sum_i alpha_i W_v h_i
    2. fusion of the pooled scene evidence with the question
           z = [v, q', v * q']
    3. MLP classifier over the answer vocabulary.

    The original head (``legacy``) only saw mean(h), so the predicted answer
    did not depend on the question at all.
    """

    def __init__(self, hidden_dim: int, question_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_proj = nn.Sequential(nn.Linear(question_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )
        self.scale = hidden_dim ** -0.5

    def forward(self, hidden: torch.Tensor, question: torch.Tensor,
                node_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if question.dim() == 1:
            question = question.unsqueeze(0)
        hidden = self.norm(hidden)
        q = self.q_proj(question)                                         # [B, D]
        scores = torch.einsum("bnd,bd->bn", self.key(hidden), self.query(q)) * self.scale
        if node_mask is not None:
            scores = scores.masked_fill(~node_mask, -1e9)
        attention = torch.softmax(scores, dim=-1)                          # [B, N]
        pooled = torch.einsum("bn,bnd->bd", attention, self.value(hidden))
        fused = torch.cat([pooled, q, pooled * q], dim=-1)
        return self.classifier(fused), attention


class SCMVLM(nn.Module):

    def __init__(self, num_classes, num_layers: int = 2, freeze_backbones: bool = True,
                 question_aware: bool = True, graph_mode: str = "soft", graph_top_k: int = 3,
                 graph_geometry: bool = True, node_identity: bool = True,
                 relation_messages: bool = True, relation_types: int = 1,
                 residual_last_block: bool = True):
        """
        graph_mode           "soft" (learned top-k graph), "legacy" (old hard-threshold graph,
                             old checkpoints) or "none" (ablation: no graph, no messages;
                             nodes are processed independently, everything else identical)
        node_identity        node input = [DETR embedding ; class embedding ; box]
        relation_messages    messages depend on the (parent, child) pair geometry
        relation_types       R > 1: learned relation types, one message transform each
        residual_last_block  False removes the residual bypass of the last message step,
                             so the answer cannot ignore messages
        """
        super().__init__()

        if graph_mode not in ("soft", "legacy", "none"):
            raise ValueError("graph_mode must be 'soft', 'legacy' or 'none'")
        self.question_aware = question_aware
        self.graph_mode = graph_mode
        self.node_identity = node_identity
        # relation-aware messages need box geometry; the legacy graph has none
        self.relation_messages = relation_messages and graph_mode == "soft"   # needs the box-geometry graph
        self.relation_types = relation_types if (self.relation_messages and graph_mode == "soft") else 1
        self.residual_last_block = residual_last_block
        self.graph_top_k = graph_top_k
        self.graph_geometry = graph_geometry and graph_mode == "soft"

        # -------------------------
        # ENCODERS
        # -------------------------
        self.vision = VisionEncoder(freeze=freeze_backbones)
        self.text = TextEncoder(freeze=freeze_backbones)

        # -------------------------
        # NODE INPUT  (DETR embedding [; class embedding ; box])
        # -------------------------
        node_input_dim = OBJECT_DIM
        if node_identity:
            self.class_embedding = nn.Embedding(NUM_CLASS_IDS, CLASS_EMBED_DIM)
            node_input_dim += CLASS_EMBED_DIM + BOX_FEATURE_DIM
        self.project_objects = nn.Sequential(
            nn.Linear(node_input_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
        )

        # unused; kept so that older checkpoints load without "unexpected key" errors
        self.project_replacement = nn.Sequential(
            nn.Linear(OBJECT_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
        )

        # -------------------------
        # GRAPH
        # -------------------------
        if graph_mode == "soft":
            self.graph = SCMGraph(HIDDEN_DIM, QUESTION_DIM, top_k=graph_top_k, threshold=0.0, mode="soft",
                                  geometry=self.graph_geometry, relation_types=self.relation_types)
        elif graph_mode == "legacy":   # checkpoints trained before the collapse fix
            self.graph = SCMGraph(HIDDEN_DIM, QUESTION_DIM, mode="legacy")
        else:
            self.graph = None

        # -------------------------
        # CAUSAL LAYERS
        # -------------------------
        self.layers = nn.ModuleList([
            CausalAttentionBlock(
                HIDDEN_DIM, NUM_HEADS,
                relation_messages=self.relation_messages,
                relation_types=self.relation_types,
                residual_messages=residual_last_block or index < num_layers - 1,
            )
            for index in range(num_layers)
        ])

        # -------------------------
        # DECODER
        # -------------------------
        if question_aware:
            self.answer_head = QuestionAwareAnswerHead(HIDDEN_DIM, QUESTION_DIM, num_classes)
        else:
            # legacy head (checkpoints trained before the fix): mean(h) only
            self.decoder = nn.Sequential(
                nn.LayerNorm(HIDDEN_DIM),
                nn.Linear(HIDDEN_DIM, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes),
            )

        # -------------------------
        # INTERVENTION ENGINE
        # -------------------------
        self.intervention = InterventionEngine()

    # =====================================================
    # ARCHITECTURE (saved in checkpoints)
    # =====================================================
    def architecture(self) -> dict:
        return {
            "question_aware": self.question_aware,
            "graph_mode": self.graph_mode,
            "graph_top_k": self.graph_top_k,
            "graph_geometry": self.graph_geometry,
            "node_identity": self.node_identity,
            "relation_messages": self.relation_messages,
            "relation_types": self.relation_types,
            "residual_last_block": self.residual_last_block,
        }

    # =====================================================
    # NODE INPUT
    # =====================================================
    def encode_nodes(self, raw_objects, class_ids=None, boxes=None):
        """raw_objects [B, N, OBJECT_DIM], class_ids [B, N] (DETR ids), boxes [B, N, 4] in 0-1."""
        if raw_objects.dim() == 2:
            raw_objects = raw_objects.unsqueeze(0)
        if not self.node_identity:
            return self.project_objects(raw_objects)
        B, N, _ = raw_objects.shape
        if class_ids is None:
            class_ids = torch.full((B, N), NO_OBJECT_ID, dtype=torch.long, device=raw_objects.device)
        if class_ids.dim() == 1:
            class_ids = class_ids.unsqueeze(0)
        if boxes is None:
            raise ValueError("Node identity needs the object boxes.")
        if boxes.dim() == 2:
            boxes = boxes.unsqueeze(0)
        boxes = boxes.to(raw_objects.dtype).to(raw_objects.device)
        size = boxes[..., 2:] - boxes[..., :2]
        node_input = torch.cat([
            raw_objects,
            self.class_embedding(class_ids.to(raw_objects.device).long()),
            boxes,
            size,
        ], dim=-1)
        return self.project_objects(node_input)

    # =====================================================
    # GRAPH
    # =====================================================
    def build_graph(self, objects, question_embedding, boxes=None):
        if objects.dim() == 2:
            objects = objects.unsqueeze(0)
        if self.graph is None:
            B, N, _ = objects.shape
            zeros = objects.new_zeros(B, N, N)
            return {"adjacency": zeros, "probs": zeros, "relation_probs": None, "geometry": None}
        return self.graph(objects, question_embedding, boxes)

    def normalise(self, adjacency):
        if self.graph is None:
            return adjacency
        return self.graph.normalise(adjacency)

    # =====================================================
    # CAUSAL PROPAGATION
    # =====================================================
    def run_causal_layers(self, objects, adjacency, boxes=None, relation_probs=None, messages=True):
        """messages=False (and graph_mode 'none') switch message passing off."""
        if objects.dim() == 2:
            objects = objects.unsqueeze(0)
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)
        messages = messages and self.graph is not None
        geometry = None
        if messages and self.relation_messages:
            if boxes is None:
                raise ValueError("Relation-aware messages need the object boxes.")
            if boxes.dim() == 2:
                boxes = boxes.unsqueeze(0)
            geometry = pairwise_box_features(boxes.to(objects.dtype).to(objects.device))
        hidden = objects
        attention_maps = []

        for layer in self.layers:
            hidden, attention = layer(hidden, adjacency, geometry, relation_probs, messages=messages)
            attention_maps.append(attention)

        return hidden, attention_maps

    # =====================================================
    # DECODER  Y = Decoder(h, q)
    # =====================================================
    def decode(self, hidden, question_embedding=None, node_mask=None, return_attention=False):
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        if self.question_aware:
            if question_embedding is None:
                raise ValueError("The question-aware answer head needs the question embedding.")
            logits, attention = self.answer_head(hidden, question_embedding, node_mask)
            return (logits, attention) if return_attention else logits
        logits = self.decoder(hidden.mean(dim=1))
        if return_attention:
            n = hidden.shape[1]
            return logits, hidden.new_full(hidden.shape[:2], 1.0 / n)
        return logits

    # =====================================================
    # FORWARD PASS
    # =====================================================
    def forward(self, image_path, question):

        vision_output = self.vision(image_path)
        question_embedding = self.text(question)

        boxes = vision_output.get("boxes_norm")
        boxes = torch.stack(boxes) if boxes else None
        class_ids = vision_output.get("label_ids")
        class_ids = torch.stack(class_ids) if class_ids else None
        output = self.forward_features(vision_output["embeddings"], question_embedding, boxes, class_ids)
        output.update({
            "labels": vision_output.get("labels"),
            "boxes": vision_output.get("boxes"),
            "scores": vision_output.get("scores"),
        })
        return output

    def forward_features(self, raw_objects, question_embedding, boxes=None, class_ids=None,
                         messages=True, adjacency_override=None, relation_probs_override=None):
        """Same as forward() but from pre-computed DETR object embeddings [B, N, OBJECT_DIM],
        BERT question embeddings [B, QUESTION_DIM], boxes [B, N, 4] scaled to 0-1 and DETR
        class ids [B, N] (used by the cached training loop).

        messages=False and adjacency_override (e.g. a scrambled graph) are used by the
        graph knock-out analysis."""

        if raw_objects.dim() == 2:
            raw_objects = raw_objects.unsqueeze(0)
        if raw_objects.shape[1] == 0:
            raise ValueError("No objects detected in the image.")
        if boxes is not None and boxes.dim() == 2:
            boxes = boxes.unsqueeze(0)
        if class_ids is not None and class_ids.dim() == 1:
            class_ids = class_ids.unsqueeze(0)

        objects = self.encode_nodes(raw_objects, class_ids, boxes)

        graph_output = self.build_graph(objects, question_embedding, boxes)

        adjacency = graph_output["adjacency"] if adjacency_override is None else adjacency_override
        relation_probs = graph_output["relation_probs"]
        if relation_probs_override is not None:
            relation_probs = relation_probs_override

        hidden, attention_maps = self.run_causal_layers(
            objects, adjacency, boxes, relation_probs, messages=messages
        )

        logits, answer_attention = self.decode(hidden, question_embedding, return_attention=True)

        return {
            "logits": logits,
            "adjacency": adjacency,
            "edge_probs": graph_output["probs"],
            "relation_probs": relation_probs,
            "objects": objects,
            "raw_objects": raw_objects,
            "boxes_norm": boxes,
            "class_ids": class_ids,
            "question_embedding": question_embedding,
            "hidden": hidden,
            "answer_attention": answer_attention,
            "attention_maps": attention_maps,
        }

    # =====================================================
    # INTERVENTIONS (differentiable; used by training and evaluation)
    # =====================================================
    def intervene(self, out, kind, nodes, new_raw=None, new_class_ids=None, mode="do"):
        """Apply an intervention to a factual ``forward_features`` output (batch size 1).

        kind="remove":  delete the nodes and all their edges from the scene.  With
                        mode="do" the rest of the learned graph is kept (children of a
                        removed node lose that parent; their rows are renormalised).
        kind="replace": do(X_k = x'): node k gets the new DETR embedding ``new_raw`` [len(nodes), D]
                        and class ``new_class_ids``; its box is unchanged.  With mode="do" the
                        incoming edges of k are cut and the rest of the graph is kept, so only
                        k's descendants change.
        mode="recompute" rebuilds the graph on the edited scene instead (scene edit).

        Returns a dict like forward_features plus ``keep`` (factual index of every node in the
        edited scene, or None for replace) and ``descendants``.
        """
        if out["objects"].shape[0] != 1:
            raise ValueError("intervene() expects a single scene (batch size 1).")
        nodes = [int(n) for n in nodes]
        q = out["question_embedding"]
        objects = out["objects"]
        adjacency = out["adjacency"]
        relation_probs = out.get("relation_probs")
        boxes = out.get("boxes_norm")
        class_ids = out.get("class_ids")
        n = objects.shape[1]

        if kind == "remove":
            keep = [i for i in range(n) if i not in nodes]
            if not keep:
                raise ValueError("Removing every object leaves an empty scene.")
            descendants = sorted({d for k in nodes for d in find_descendants(adjacency[0], k)} - set(nodes))
            index = torch.tensor(keep, device=objects.device)
            new_objects = objects.index_select(1, index)
            new_adjacency = self.normalise(adjacency.index_select(1, index).index_select(2, index))
            if relation_probs is not None:
                relation_probs = relation_probs.index_select(1, index).index_select(2, index)
            if boxes is not None:
                boxes = boxes.index_select(1, index)
            if class_ids is not None:
                class_ids = class_ids.index_select(1, index)
        elif kind == "replace":
            keep = None
            if new_raw is None or new_class_ids is None:
                raise ValueError("replace needs new_raw and new_class_ids")
            new_raw = new_raw.to(objects.device).reshape(len(nodes), -1)
            new_class = torch.as_tensor(new_class_ids, device=objects.device).reshape(1, len(nodes))
            node_boxes = boxes[:, nodes] if boxes is not None else None
            replacement = self.encode_nodes(new_raw.unsqueeze(0), new_class, node_boxes)[0]
            mask = torch.zeros(n, 1, device=objects.device, dtype=objects.dtype)
            mask[nodes] = 1.0
            scattered = torch.zeros_like(objects[0])
            scattered[nodes] = replacement
            new_objects = (objects[0] * (1 - mask) + scattered * mask).unsqueeze(0)
            row_cut = torch.ones(n, 1, device=objects.device, dtype=adjacency.dtype)
            row_cut[nodes] = 0.0                                   # do(): cut every parent -> k
            new_adjacency = adjacency * row_cut.unsqueeze(0)
            if class_ids is not None:
                class_ids = class_ids.clone()
                class_ids[0, nodes] = new_class[0]
            descendants = sorted({d for k in nodes for d in find_descendants(new_adjacency[0], k)} - set(nodes))
        else:
            raise ValueError(f"unknown intervention kind: {kind}")

        if mode == "recompute":
            graph_output = self.build_graph(new_objects, q, boxes)
            new_adjacency = graph_output["adjacency"]
            relation_probs = graph_output["relation_probs"]
        elif mode != "do":
            raise ValueError("mode must be 'do' or 'recompute'")

        hidden, _ = self.run_causal_layers(new_objects, new_adjacency, boxes, relation_probs)
        logits, answer_attention = self.decode(hidden, q, return_attention=True)
        return {
            "logits": logits,
            "adjacency": new_adjacency,
            "relation_probs": relation_probs,
            "objects": new_objects,
            "boxes_norm": boxes,
            "class_ids": class_ids,
            "question_embedding": q,
            "hidden": hidden,
            "answer_attention": answer_attention,
            "keep": keep,
            "descendants": descendants,
        }

    # =====================================================
    # COUNTERFACTUAL  Y_do(X_k = x')   (node-space replacement; used by --cf-mode invariance_old)
    # =====================================================
    def counterfactual(
        self,
        objects,
        adjacency,
        node_indices,
        replacement,
        question_embedding,
        boxes=None,
        relation_probs=None,
    ):
        """do(X_k = x'): cut the parents of k (row k), set h_k = x', keep every
        other structural equation fixed and propagate through the same graph.
        The graph is NOT rebuilt, otherwise the cut would be undone."""

        cf_objects, cf_adj, descendants = self.intervention.intervene(
            objects,
            adjacency,
            node_indices,
            replacement
        )

        cf_hidden, _ = self.run_causal_layers(cf_objects, cf_adj, boxes, relation_probs)

        logits = self.decode(cf_hidden, question_embedding)

        return {
            "logits": logits,
            "adjacency": cf_adj,
            "hidden": cf_hidden,
            "descendants": descendants
        }

    # =====================================================
    # PROPAGATE ONLY
    # =====================================================
    def propagate(self, objects, adjacency, boxes=None, relation_probs=None):
        hidden, _ = self.run_causal_layers(objects, adjacency, boxes, relation_probs)
        return hidden

    # =====================================================
    # FROM OBJECTS ONLY
    # =====================================================
    def forward_from_objects(self, objects, question, boxes=None, class_ids=None):

        question_embedding = self.text(question)
        output = self.forward_features(objects, question_embedding, boxes, class_ids)
        return {
            "logits": output["logits"],
            "adjacency": output["adjacency"],
            "hidden": output["hidden"]
        }


def build_model_from_checkpoint(checkpoint: dict, **kwargs) -> SCMVLM:
    """Create an SCMVLM whose architecture matches a saved checkpoint.

    format_version >= 3 stores an "architecture" dict; older checkpoints are mapped to the
    architecture they were trained with (no node identity, parent-only messages)."""
    architecture = dict(checkpoint.get("architecture") or {})
    if not architecture:
        architecture = {
            **LEGACY_ARCHITECTURE,
            "question_aware": bool(checkpoint.get("question_aware", False)),
            "graph_mode": checkpoint.get("graph_mode", "legacy"),
            "graph_top_k": checkpoint.get("graph_top_k", 3),
            "graph_geometry": bool(checkpoint.get("graph_geometry", False)),
        }
    architecture.update(kwargs)
    return SCMVLM(num_classes=checkpoint["num_classes"], **architecture)
