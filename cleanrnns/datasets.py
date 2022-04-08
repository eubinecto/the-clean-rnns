import torch  # noqa
from torch.utils.data import Dataset  # noqa
from typing import Tuple


class DatasetForClassification(Dataset):
    """
    감성분석, 혐오분석 등
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        """
        :param x: (N, L)
        :param y: (N,)
        """
        self.x = x
        self.y = y

    def __len__(self):
        assert self.x.shape[0] == self.y.shape[0]
        return self.y.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class DatasetForSeq2Seq(Dataset):
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
