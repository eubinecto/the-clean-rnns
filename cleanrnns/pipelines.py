import torch
from typing import Tuple


class Pipeline:
    def __init__(self, model: str):
        self.model = model


class PipelineForClassification(Pipeline):
    """
    감성분석, 혐오분석 등
    """
    def __init__(self, model: str):
        super().__init__(model)


class PipelineForSeq2Seq(Pipeline):
    """
    🚧 to be added later 🚧
    Q & A, language modeling, 번역 등.
    """
    raise NotImplementedError


class PipelineForNER(Pipeline):
    """
    🚧 to be added later 🚧
    Q & A, language modeling, 번역 등.
    """
    raise NotImplementedError
