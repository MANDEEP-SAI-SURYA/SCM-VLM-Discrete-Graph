import torch
import torch.nn as nn

from PIL import Image

from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection
)

from configs.config import (
    DETR_MODEL,
    DEVICE,
    NUM_OBJECTS
)

DETECTION_THRESHOLD = 0.9


def select_detections(
    logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    query_states: torch.Tensor,
    image_size: tuple[int, int],
    threshold: float = DETECTION_THRESHOLD,
    max_objects: int = NUM_OBJECTS,
) -> dict:
    """Pick the confident DETR detections of ONE image and keep each detection's own
    query embedding.

    logits       [Q, C + 1]  class logits (last column = "no object")
    pred_boxes   [Q, 4]      (cx, cy, w, h) scaled to 0-1
    query_states [Q, D]      decoder output fed to DETR's class head
    image_size   (width, height)

    This mirrors ``DetrImageProcessor.post_process_object_detection`` (softmax, best
    non-background class, score > threshold), but keeps track of WHICH query produced
    each detection.  The earlier code indexed the decoder states with the position in
    the post-processed (already filtered) list, so detection r received the embedding
    of query r instead of its own query.
    """
    probs = logits.softmax(-1)[..., :-1]
    scores, label_ids = probs.max(-1)
    kept_query = torch.nonzero(scores > threshold, as_tuple=False).flatten()
    order = torch.argsort(scores[kept_query], descending=True)[:max_objects]
    query_index = kept_query[order]

    width, height = image_size
    cx, cy, w, h = pred_boxes[query_index].unbind(-1)
    boxes_norm = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    scale = torch.tensor([width, height, width, height], dtype=boxes_norm.dtype, device=boxes_norm.device)
    return {
        "query_index": query_index,
        "embeddings": query_states[query_index],
        "label_ids": label_ids[query_index],
        "scores": scores[query_index],
        "boxes_norm": boxes_norm,
        "boxes": boxes_norm * scale,
    }


class VisionEncoder(nn.Module):
    """
    DETR-based vision encoder.

    Returns:
    - object embeddings (the decoder output of the query that produced each detection,
      i.e. the input of DETR's class head)
    - object labels and DETR class ids
    - bounding boxes (pixels and scaled to 0-1)
    - confidence scores
    """

    def __init__(
        self,
        model_name: str = DETR_MODEL,
        freeze: bool = True
    ) -> None:

        super().__init__()

        self.freeze = freeze

        self.processor = DetrImageProcessor.from_pretrained(
            model_name
        )

        self.model = DetrForObjectDetection.from_pretrained(
            model_name
        )

        self.model.to(DEVICE)

        if freeze:

            self.model.eval()

            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):

        super().train(mode)

        if self.freeze:
            self.model.eval()

        return self

    def _load_images(
        self,
        image_paths: str | list[str]
    ) -> list[Image.Image]:

        if isinstance(image_paths, str):
            image_paths = [image_paths]

        return [
            Image.open(path).convert("RGB")
            for path in image_paths
        ]

    def forward(
        self,
        image_paths: str | list[str]
    ):

        images = self._load_images(image_paths)

        inputs = self.processor(
            images=images,
            return_tensors="pt"
        ).to(DEVICE)

        grad_context = (
            torch.no_grad()
            if self.freeze
            else torch.enable_grad()
        )

        with grad_context:

            outputs = self.model(**inputs)

        # Object embeddings = the decoder output that DETR's class head reads.
        # ``last_hidden_state`` is that tensor (after the decoder's final layernorm) in
        # every transformers version.  ``decoder_hidden_states[-1]`` is NOT: in
        # transformers 4.x it is taken after the layernorm, in 5.x before it.
        query_states = outputs.last_hidden_state

        labels, label_ids, boxes, boxes_norm, scores, embeddings, query_index = [], [], [], [], [], [], []
        for batch_index, image in enumerate(images):
            picked = select_detections(
                outputs.logits[batch_index],
                outputs.pred_boxes[batch_index],
                query_states[batch_index],
                image.size,
            )
            labels.append([self.model.config.id2label[int(i)] for i in picked["label_ids"]])
            label_ids.append(picked["label_ids"])
            boxes.append(picked["boxes"])
            boxes_norm.append(picked["boxes_norm"])
            scores.append(picked["scores"])
            embeddings.append(picked["embeddings"])
            query_index.append(picked["query_index"])

        if len({len(e) for e in embeddings}) == 1:
            stacked = torch.stack(embeddings)
        else:  # different numbers of objects per image: pad (callers use one image at a time)
            longest = max(len(e) for e in embeddings)
            stacked = torch.zeros(len(embeddings), longest, query_states.shape[-1], device=query_states.device)
            for i, e in enumerate(embeddings):
                stacked[i, :len(e)] = e

        return {
            "embeddings": stacked,
            "labels": labels,
            "label_ids": label_ids,
            "boxes": boxes,
            "boxes_norm": boxes_norm,
            "scores": scores,
            "query_index": query_index,
        }

    def encode(
        self,
        image_paths: str | list[str]
    ):

        return self.forward(image_paths)
