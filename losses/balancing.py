"""Adaptive weighting of the compound loss and curriculum / staged regularisation
(paper, Future Work: "adaptive weighting mechanisms for dynamically balancing these objectives",
"curriculum based optimization strategies and staged regularization scheduling").

The training loss is a sum of terms

    task      VQA answer cross-entropy
    cf        counterfactual cross-entropy CE(Y_do, Y_cf)  (or the old KL term)
    edge      relation supervision (contact BCE, parent selection)
    acyclic   NOTEARS h(A)
    sparse    mean edge weight
    entropy   edge-distribution entropy

LossBalancer turns the raw terms into the total loss:

    total = sum_k  w_k(t) * c_k(t) * lambda_k * L_k

* lambda_k  fixed base weights (the previous constants)
* c_k(t)    curriculum multiplier (staged regularisation):
              warm-up     c = 0 for the causal constraints (acyclic, sparse, entropy) and cf,
                          so the model first learns semantic representations;
              ramp        c rises linearly from 0 to 1;
              afterwards  c = 1.
            task and edge (relation supervision) are always on.
* w_k(t)    adaptive weight of the "objective" terms (task, cf, edge):
              fixed       w = 1
              uncertainty homoscedastic uncertainty weighting (Kendall, Gal & Cipolla, CVPR 2018):
                          w_k = exp(-s_k) with a learned log-variance s_k and the regulariser
                          + 0.5 * s_k, so noisy / hard objectives are down-weighted automatically
              dwa         Dynamic Weight Average (Liu, Johns & Davison, CVPR 2019):
                          w_k(t) = K * softmax_k( r_k(t-1) / T ),  r_k = L_k(t-1) / L_k(t-2),
                          objectives whose loss stopped decreasing (plateau) get more weight
  The structural regularisers keep their fixed weights (they are constraints, not objectives:
  letting them be learned would let the model switch them off).

competence() / select_by_competence() implement a competence-based data curriculum
(Platanios et al., NAACL 2019): at epoch t only the easiest c(t) fraction of the training
questions is used, c(t) = min(1, sqrt(c0^2 + (1 - c0^2) * t / T_c)).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

OBJECTIVES = ("task", "cf", "edge")
CONSTRAINTS = ("acyclic", "sparse", "entropy")
STAGED = CONSTRAINTS + ("cf",)            # off during the warm-up stage


def stage_multiplier(epoch: int, epochs: int, warmup_frac: float, ramp_frac: float) -> float:
    """0 during warm-up, linear ramp to 1, then 1.  ``epoch`` is 0-based."""
    warmup = int(round(warmup_frac * epochs))
    ramp = max(int(round(ramp_frac * epochs)), 1)
    if epoch < warmup:
        return 0.0
    return min(1.0, (epoch - warmup + 1) / ramp)


def competence(epoch: int, epochs: int, start: float = 0.3, curriculum_frac: float = 0.5) -> float:
    """Share of the (difficulty-sorted) training data used at ``epoch`` (0-based)."""
    length = max(int(round(curriculum_frac * epochs)), 1)
    t = min(epoch / length, 1.0)
    return min(1.0, math.sqrt(start ** 2 + (1 - start ** 2) * t))


def select_by_competence(indices: list[int], difficulty: dict[int, float], level: float) -> list[int]:
    """The easiest ``level`` fraction of ``indices`` (ties keep the original order)."""
    if level >= 1.0:
        return list(indices)
    ranked = sorted(indices, key=lambda i: difficulty.get(i, 0.0))
    keep = max(1, int(math.ceil(level * len(ranked))))
    chosen = set(ranked[:keep])
    return [i for i in indices if i in chosen]


class LossBalancer(nn.Module):
    """Combines the loss terms (see module docstring).  Its parameters (uncertainty weighting)
    must be added to the optimiser."""

    def __init__(self, base: dict[str, float], method: str = "fixed", curriculum: bool = False,
                 warmup_frac: float = 0.2, ramp_frac: float = 0.3, dwa_temperature: float = 2.0) -> None:
        super().__init__()
        if method not in ("fixed", "uncertainty", "dwa"):
            raise ValueError("method must be 'fixed', 'uncertainty' or 'dwa'")
        self.base = dict(base)
        self.method = method
        self.curriculum = curriculum
        self.warmup_frac = warmup_frac
        self.ramp_frac = ramp_frac
        self.dwa_temperature = dwa_temperature
        self.log_vars = nn.ParameterDict({k: nn.Parameter(torch.zeros(())) for k in OBJECTIVES})
        self.stage = 1.0
        self.dwa_weights = {k: 1.0 for k in OBJECTIVES}
        self.history: list[dict[str, float]] = []           # epoch means of the objective terms
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    # ---------------------------------------------------------------- per-epoch state
    def set_epoch(self, epoch: int, epochs: int) -> None:
        self.stage = stage_multiplier(epoch, epochs, self.warmup_frac, self.ramp_frac) if self.curriculum else 1.0
        if self.method == "dwa" and len(self.history) >= 2:
            ratios = {k: self.history[-1].get(k, 1.0) / max(self.history[-2].get(k, 1.0), 1e-8) for k in OBJECTIVES}
            exps = {k: math.exp(r / self.dwa_temperature) for k, r in ratios.items()}
            total = sum(exps.values())
            self.dwa_weights = {k: len(OBJECTIVES) * v / total for k, v in exps.items()}
        self._sums, self._counts = {}, {}

    def end_epoch(self) -> None:
        means = {k: self._sums[k] / self._counts[k] for k in self._sums if self._counts[k]}
        self.history.append(means)

    def curriculum_multiplier(self, name: str) -> float:
        return self.stage if (self.curriculum and name in STAGED) else 1.0

    def uses(self, name: str) -> bool:
        """False when a term currently has zero weight (e.g. cf during warm-up)."""
        return self.base.get(name, 0.0) > 0 and self.curriculum_multiplier(name) > 0

    # ---------------------------------------------------------------- combine
    def forward(self, terms: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        total = None
        weights = {}
        for name, value in terms.items():
            if value is None:
                continue
            base = self.base.get(name, 0.0) * self.curriculum_multiplier(name)
            if base == 0:
                weights[name] = 0.0
                continue
            if name in OBJECTIVES:
                self._sums[name] = self._sums.get(name, 0.0) + float(value.detach())
                self._counts[name] = self._counts.get(name, 0) + 1
            if self.method == "uncertainty" and name in OBJECTIVES:
                s = self.log_vars[name]
                term = torch.exp(-s) * base * value + 0.5 * s
                weights[name] = float(torch.exp(-s.detach())) * base
            elif self.method == "dwa" and name in OBJECTIVES:
                term = self.dwa_weights[name] * base * value
                weights[name] = self.dwa_weights[name] * base
            else:
                term = base * value
                weights[name] = base
            total = term if total is None else total + term
        if total is None:
            total = next(v for v in terms.values() if v is not None) * 0.0
        return total, weights

    def describe(self) -> str:
        parts = [f"stage x{self.stage:.2f}"] if self.curriculum else []
        if self.method == "uncertainty":
            parts += [f"w_{k} {math.exp(-float(self.log_vars[k])):.2f}" for k in OBJECTIVES]
        elif self.method == "dwa":
            parts += [f"w_{k} {self.dwa_weights[k]:.2f}" for k in OBJECTIVES]
        return ", ".join(parts) or "fixed weights"
