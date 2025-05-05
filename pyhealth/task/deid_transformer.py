"""
Task module: Deidentification with Transformer models.

Extends PyHealth's Task base class to integrate HuggingFace token‑classification
heads into PyHealth's training flow.

Example:
>>> from pyhealth.task import DeidTransformer
>>> task = DeidTransformer(model_name="bert-base-uncased")
"""

from typing import Dict
from transformers import AutoModelForTokenClassification
from pyhealth.tasks import TokenClassificationTask


class DeidTransformer(TokenClassificationTask):
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask, labels=None) -> Dict:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {"loss": outputs.loss, "logits": outputs.logits}
