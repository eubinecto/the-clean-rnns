import torch
from typing import Tuple
from torch.utils.data import Dataset


class DatasetForClassification(Dataset):
    """
    감성분석, 혐오분석 등
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        """
        :param X: (N, L)
        :param y: (N,)
        """
        self.X = X
        self.y = y

    def __len__(self):
        assert self.X.shape[0] == self.y.shape[0]
        return self.y.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


class DatasetForConditionalGeneration(Dataset):
    """
    🚧 to be added later 🚧
    Q & A, language modeling, 번역 등.
    """
    pass


class DatasetForNER(Dataset):
    """
    🚧 to be added later 🚧
    Q & A, language modeling, 번역 등.
    """
    pass
