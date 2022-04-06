import torch
from typing import Tuple
from tokenizers import Tokenizer
from cleanrnns.models import ClassificationBase
from cleanrnns import tensors as T


class PipelineForClassification:
    """
    감성분석, 혐오분석 등
    """
    def __init__(self, model: ClassificationBase, tokenizer: Tokenizer, config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, text: str) -> Tuple[int, float]:
        # build the inputs
        x = T.inputs_for_classification(text2label=[(text, None)],
                                        max_length=self.config['max_length'],
                                        tokenizer=self.tokenizer)
        probs = self.model.predict(x)  # (N, L) -> (N, C)
        preds = torch.argmax(probs, dim=-1)  # (N, C) -> (N,)
        label = int(preds.squeeze())
        probability = float(probs.squeeze()[label])
        return label, probability


class PipelineForSeq2Seq:
    """
    🚧 to be added later 🚧
    Q & A, language modeling, 번역 등.
    """
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class PipelineForNER:
    """
    🚧 to be added later 🚧
    Q & A, language modeling, 번역 등.
    """
    def __call__(self, *args, **kwargs):
        raise NotImplementedError
