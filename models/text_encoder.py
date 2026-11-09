import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

from configs.config import BERT_MODEL, DEVICE


class TextEncoder(nn.Module):
    """BERT question encoder using mean-pooled token embeddings."""

    def __init__(self, model_name: str = BERT_MODEL, freeze: bool = True) -> None:
        super().__init__()
        self.freeze = freeze
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(DEVICE)

        if freeze:
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> "TextEncoder":
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, questions: str | list[str]) -> torch.Tensor:
        if isinstance(questions, str):
            questions = [questions]

        inputs = self.tokenizer(
            questions,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(DEVICE)

        grad_context = torch.no_grad() if self.freeze else torch.enable_grad()
        with grad_context:
            outputs = self.model(**inputs)

        mask = inputs["attention_mask"].unsqueeze(-1)
        summed = (outputs.last_hidden_state * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        return summed / lengths

    def encode(self, questions: str | list[str]) -> torch.Tensor:
        return self.forward(questions)
