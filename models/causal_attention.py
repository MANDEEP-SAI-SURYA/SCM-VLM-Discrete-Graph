import torch
import torch.nn as nn

from models.scm_graph import GEOMETRY_DIM

# Convention: A[i, j] > 0 means j -> i (j is a parent of i); node i aggregates
# messages from its parents:  h_i += out( sum_j A_ij m_ij ).


class MultiHeadCausalAttention(nn.Module):
    """SCM message passing.

    legacy (relation_messages=False):   m_i = W (sum_j A_ij h_j)          (parent features only)
    relation-aware (default for new models):
        g_ij = MLP(pairwise_box_features[i, j])           geometry of parent j relative to child i
        m_ij = W_r [h_j ; g_ij] + b_r                      one W_r per relation type r
        m_i  = sum_r sum_j A_ij * p_ij(r) * m_ij           p_ij(r): relation-type probabilities
    so a message depends on the pair (j, i), not only on the parent.  A row with no
    parents receives exactly zero message.

    diagnostic=True (used with the discrete causal graph) adds Pearl-style DIAGNOSTIC messages:
    besides the causal messages parent -> child above, every node also receives
        d_i = sum_r sum_j A_ji * p_ji(r) * (V_r [h_j ; g_ij] + c_r)        child j -> parent i
    over the same edges, with separate transforms V_r.  As in belief propagation on a causal
    DAG, evidence then flows both ways while the graph stays acyclic; a do() intervention that
    cuts the edges into k removes them for both directions.

    The name is kept for compatibility with older checkpoints; this is not QKV attention.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1,
                 relation_messages: bool = False, relation_types: int = 1, relation_dim: int = 64,
                 diagnostic: bool = False) -> None:
        super().__init__()
        self.relation_messages = relation_messages
        self.diagnostic = diagnostic and relation_messages
        self.relation_types = relation_types if relation_messages else 1
        if relation_messages:
            self.relation_mlp = nn.Sequential(
                nn.Linear(GEOMETRY_DIM, relation_dim), nn.ReLU(), nn.Linear(relation_dim, relation_dim)
            )
            # bias added separately, scaled by the row's parent mass
            self.type_message = nn.ModuleList([
                nn.Linear(dim + relation_dim, dim, bias=False) for _ in range(self.relation_types)
            ])
            self.type_bias = nn.Parameter(torch.zeros(self.relation_types, dim))
            if self.diagnostic:
                self.diagnostic_message = nn.ModuleList([
                    nn.Linear(dim + relation_dim, dim, bias=False) for _ in range(self.relation_types)
                ])
                self.diagnostic_bias = nn.Parameter(torch.zeros(self.relation_types, dim))
        else:
            self.message = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor,
                geometry: torch.Tensor | None = None,
                relation_probs: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden [B, N, D]; adjacency [B, N, N]; geometry [B, N, N, GEOMETRY_DIM] (pairwise box
        features, required for relation-aware messages); relation_probs [B, N, N, R] or None."""
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)

        if not self.relation_messages:
            messages = torch.matmul(adjacency, hidden)
            messages = self.message(messages)
            messages = self.dropout(messages)
            return self.out(messages), adjacency

        if geometry is None:
            raise ValueError("Relation-aware messages need the pairwise box geometry.")
        if geometry.dim() == 3:
            geometry = geometry.unsqueeze(0)
        relation = self.relation_mlp(geometry)          # [B, receiver, sender, G]
        messages = self._aggregate(hidden, adjacency, relation, relation_probs, self.type_message, self.type_bias)
        if self.diagnostic:
            # child -> parent: the same edges, reversed (A^T); geometry stays [receiver, sender]
            reverse_probs = None if relation_probs is None else relation_probs.transpose(1, 2)
            messages = messages + self._aggregate(hidden, adjacency.transpose(1, 2), relation, reverse_probs,
                                                  self.diagnostic_message, self.diagnostic_bias)
        messages = self.dropout(messages)
        return self.out(messages), adjacency

    def _aggregate(self, hidden, adjacency, relation, relation_probs, linears, biases):
        """sum_r sum_j w_ij(r) (W_r [h_j ; g_ij] + b_r) with w_ij(r) = A_ij * p_ij(r)."""
        messages = torch.zeros_like(hidden)
        for r, linear in enumerate(linears):
            weight = adjacency
            if relation_probs is not None and self.relation_types > 1:
                weight = adjacency * relation_probs[..., r]
            agg_h = torch.matmul(weight, hidden)                                 # sum_j w_ij h_j
            agg_g = torch.einsum("bij,bijg->big", weight, relation)              # sum_j w_ij g_ij
            mass = weight.sum(dim=-1, keepdim=True)                              # sum_j w_ij
            messages = messages + linear(torch.cat([agg_h, agg_g], dim=-1)) + mass * biases[r]
        return messages


class CausalAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1,
                 relation_messages: bool = False, relation_types: int = 1,
                 residual_messages: bool = True, diagnostic: bool = False) -> None:
        """residual_messages=False removes the residual bypass of the message step:
        h_i = m_i instead of h_i + m_i, so everything after this block depends on messages."""
        super().__init__()
        self.residual_messages = residual_messages
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attention = MultiHeadCausalAttention(dim, num_heads, dropout, relation_messages, relation_types,
                                                  diagnostic=diagnostic)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor,
                geometry: torch.Tensor | None = None, relation_probs: torch.Tensor | None = None,
                messages: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """messages=False switches message passing off (used by --graph-mode none and by the
        'messages off' knock-out); the per-node FFN still runs."""
        if messages:
            propagated, graph = self.attention(self.norm1(hidden), adjacency, geometry, relation_probs)
            if self.residual_messages:
                hidden = hidden + self.dropout(propagated)
            else:
                hidden = self.dropout(propagated)
        else:
            graph = adjacency
        hidden = hidden + self.dropout(self.ffn(self.norm2(hidden)))
        return hidden, graph
