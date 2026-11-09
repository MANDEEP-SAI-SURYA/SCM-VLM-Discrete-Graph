import torch


def build_intervention_matrix(model, image_path, question, objects, adjacency, device, boxes=None, class_ids=None):
    """
    Computes causal effect proxy for each node:
    effect[i] = |f(x) - f(do(x_i = 0))|

    ``objects`` are the raw DETR object embeddings [1, N, 256]; ``boxes`` [1, N, 4] (0-1) and
    ``class_ids`` [1, N] are needed by models with node identity / box geometry (all models
    trained with format_version >= 3).  ``adjacency`` and ``device`` are kept for compatibility.
    """

    model.eval()

    with torch.no_grad():

        # factual prediction (on the same objects, so both sides use identical inputs)
        factual = model.forward_from_objects(objects, question, boxes, class_ids)["logits"]

        num_nodes = objects.shape[1]
        effects = []

        for i in range(num_nodes):

            # create intervention (zero the node's DETR embedding)
            intervened_objects = objects.clone()
            intervened_objects[:, i, :] = 0.0

            # run model with intervention
            cf_output = model.forward_from_objects(
                intervened_objects,
                question,
                boxes,
                class_ids,
            )

            cf_logits = cf_output["logits"]

            # causal effect magnitude
            effect = torch.abs(factual - cf_logits).mean(dim=-1)

            effects.append(effect)

        return torch.stack(effects, dim=1)
