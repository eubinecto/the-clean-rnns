import torch
from torch.nn import functional as F
from typing import Tuple, Union
import pytorch_lightning as pl
from cleanrnns.models import RNN, LSTM, BiLSTM, BiLSTMSearch


class Classification(pl.LightningModule):
    def __init__(self, encoder: Union[RNN, LSTM, BiLSTM, BiLSTMSearch], num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Linear(encoder.hidden_size, num_classes)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> dict:
        X, y = batch
        H_all = self.encoder(X)  # (N, L) -> (N, L, H)
        H_last = H_all[:, -1]  # (N, L, H) -> (N, H)
        logits = self.classifier(H_last)  # (N, H) -> (N, C)
        probs = torch.softmax(logits, dim=-1)  # (N, C) -> (N, C(normalised))
        loss = F.cross_entropy(probs, y).sum()  # (N, C), (N,) -> (N,) -> (,)
        return {
            "loss": loss
        }


class RNNForClassification(Classification):

    def __init__(self, vocab_size: int, hidden_size: int,
                 num_classes: int, lr: float):
        super().__init__(RNN(vocab_size, hidden_size), num_classes)
        self.save_hyperparameters()


class LSTMForClassification(Classification):

    def __init__(self, vocab_size: int, hidden_size: int,
                 num_classes: int, lr: float):
        super().__init__(LSTM(), num_classes)
        self.save_hyperparameters()
        raise NotImplementedError


class BiLSTMForClassification(Classification):

    def __init__(self, vocab_size: int, hidden_size: int,
                 num_classes: int, lr: float):
        super().__init__(BiLSTM(), num_classes)
        self.save_hyperparameters()
        raise NotImplementedError


class BiLSTMSearchForClassification(Classification):

    def __init__(self, vocab_size: int, hidden_size: int,
                 num_classes: int, lr: float):
        super().__init__(BiLSTMSearch(), num_classes)
        self.save_hyperparameters()
        raise NotImplementedError


class Seq2Seq(pl.LightningModule):
    """
    i.e. conditional generator.
    """
    pass


class NER(pl.LightningModule):
    pass
