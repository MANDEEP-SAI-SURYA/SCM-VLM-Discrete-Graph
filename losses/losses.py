import torch
import torch.nn.functional as F


# =========================================================
# DAG constraint (NOTEARS-style)
# =========================================================
def acyclicity_loss(A):
    if A.dim() == 2:
        A = A.unsqueeze(0)

    A = A.float()
    expm = torch.matrix_exp(A * A)
    trace = expm.diagonal(dim1=-2, dim2=-1).sum(-1)

    return (trace - A.shape[-1]).mean()


# =========================================================
# SPARSITY (simple L1 works BEST for binary graphs)
# =========================================================
def sparsity_loss(A):
    return A.mean()


# =========================================================
# CONSISTENCY ACROSS WORLDS
# =========================================================
def consistency_loss(y1, y2):
    if y2 is None:
        return y1.new_tensor(0.0)
    return F.mse_loss(y1, y2)


# =========================================================
# TOTAL LOSS
# =========================================================
def total_scm_loss(
    logits,
    labels,
    adjacency,
    counterfactual_logits=None,
    counterfactual_labels=None,
    stable_logits=None,
    lambda_cf=1.0,
    lambda_acyc=0.1,
    lambda_sparse=0.001,
    lambda_cons=0.1,
):

    task_loss = F.cross_entropy(logits, labels)

    if counterfactual_logits is not None and counterfactual_labels is not None:
        cf_loss = F.cross_entropy(
            counterfactual_logits,
            counterfactual_labels
        )
    else:
        cf_loss = logits.new_tensor(0.0)

    acyc_loss = acyclicity_loss(adjacency)
    sparse_loss = sparsity_loss(adjacency)
    cons_loss = consistency_loss(logits, stable_logits)

    total = (
        task_loss
        + lambda_cf * cf_loss
        + lambda_acyc * acyc_loss
        + lambda_sparse * sparse_loss
        + lambda_cons * cons_loss
    )

    return total, {
        "task": task_loss.detach(),
        "counterfactual": cf_loss.detach(),
        "acyclicity": acyc_loss.detach(),
        "sparsity": sparse_loss.detach(),
        "consistency": cons_loss.detach(),
    }
