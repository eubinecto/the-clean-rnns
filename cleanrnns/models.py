from typing import Union, Tuple
import pytorch_lightning as pl
import torch
from torch.nn import functional as F


class RNNCell(torch.nn.Module):
    def __init__(self, embeddings: torch.nn.Embedding, hidden_size: int):
        super().__init__()
        self.embeddings = embeddings
        self.W_hh = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.W_xh = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.register_buffer("dummy", torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        h_t = f_W(h_t-1(past), x_t(now))
        h_t = tanh(W_hh * h_t-1 + W_xh * x_t)
        :param x - (N, L)
        :return: memories - (N, L, H)
        """
        N, L = x.shape
        memories = list()
        past = self.dummy.unsqueeze(0).expand(N, -1)  # (H) -> (1, H) ->  (N, H)
        for time in range(L):
            now = self.embeddings(x[:, time])  # (N, L) -> (N, 1) -> (N, H)
            past = torch.tanh(self.W_hh(past) + self.W_xh(now))  # ... -> (N, H)
            memories.append(past)
        return torch.stack(memories, dim=1)  # ... -> (N, L, H)


class RNN(torch.nn.Module):
    """
    A simple multi-layer RNN.
    Complexity - (L * D)
    https://medium.com/ecovisioneth/building-deep-multi-layer-recurrent-neural-networks-with-star-cell-2f01acdb73a7
    """
    def __init__(self, vocab_size: int, hidden_size: int, depth: int):
        super().__init__()
        self.embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.cells = torch.nn.ModuleList([RNNCell(self.embeddings, hidden_size) for _ in range(depth)])

    def forward(self, x: torch.Tensor):
        for cell in self.cells:
            x = cell(x)
        return x


class LSTM(torch.nn.Module):
    """
    🚧 공사중 🚧
    """
    pass


class BiLSTM(torch.nn.Module):
    """
    🚧 공사중 🚧
    """
    pass


class BiLSTMSearch(torch.nn.Module):
    """
    🚧 공사중 🚧
    """
    pass


# --- lightning modules --- #
class Classification(pl.LightningModule):
    def __init__(self, encoder: Union[RNN, LSTM, BiLSTM, BiLSTMSearch], num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Linear(self.hparams['hidden_size'], num_classes)

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Instantiates and returns the optimizer to be used for this model
        e.g. torch.optim.Adam
        """
        # The authors used Adam, so we might as well use it as well.
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])


class RNNForClassification(Classification):

    def __init__(self, vocab_size: int, hidden_size: int,
                 num_classes: int, lr: float, depth: int):
        self.save_hyperparameters()
        super().__init__(RNN(vocab_size, hidden_size, depth), num_classes)


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