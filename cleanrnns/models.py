import torch
import pytorch_lightning as pl
from typing import Union, Tuple, List
from torch.nn import functional as F
from torchmetrics import functional as mF
from cleanrnns.rnns import RNN, LSTM, BiLSTM


# --- lightning modules --- #
class ClassificationBase(pl.LightningModule):
    def __init__(self, encoder: Union[RNN, LSTM, BiLSTM], num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Linear(self.hparams['hidden_size'], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        memories = self.encoder(x)  # (N, L) -> (N, L, H)
        last = memories[:, -1]  # (N, L, H) -> (N, H)
        return last

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        last = self.forward(x)  # (N, L) -> (N, H)
        logits = self.classifier(last)  # (N, H) -> (N, C)
        probs = torch.softmax(logits, dim=-1)  # (N, C) -> (N, C) (normalised over C)
        return probs

    def on_train_start(self):
        # deep models should be initialised with so-called "Xavier initialisation"
        # refer to: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args) -> dict:
        x, y = batch
        last = self.forward(x)  # (N, L) -> (N, H)
        logits = self.classifier(last)  # (N, H) -> (N, C)
        loss = F.cross_entropy(logits, y).sum()  # (N, C), (N,) -> (N,) -> (,)
        return {
            "logits": logits.detach(),
            "y": y.detach(),
            "loss": loss
        }

    def training_step_end(self, step_output: dict):
        self.log("Train/loss", step_output['loss'])

    def training_epoch_end(self, outputs: List[dict]):
        logits = torch.concat([out['logits'] for out in outputs], dim=0)  # noqa, num_batches * (N, C) -> (num_batches * N, C)
        y = torch.concat([out['y'] for out in outputs], dim=0)  # # num_batches * (N,) -> (num_batches * N,)
        f1_score = mF.f1_score(preds=logits, target=y)
        accuracy = mF.accuracy(preds=logits, target=y)
        self.log("Train/f1_score", f1_score)
        self.log("Train/accuracy", accuracy)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args) -> dict:
        return self.training_step(batch)

    def validation_step_end(self, step_output: dict):
        self.log("Validation/loss", step_output['loss'])

    def validation_epoch_end(self, outputs: List[dict]):
        logits = torch.concat([out['logits'] for out in outputs], dim=0)  # noqa, num_batches * (N, C) -> (num_batches * N, C)
        y = torch.concat([out['y'] for out in outputs], dim=0)  # num_batches * (N,) -> (num_batches * N,)
        f1_score = mF.f1_score(preds=logits, target=y)
        accuracy = mF.accuracy(preds=logits, target=y)
        self.log("Validation/f1_score", f1_score)
        self.log("Validation/accuracy", accuracy)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], * args) -> dict:
        x, y = batch
        memories = self.encoder(x)  # (N, L) -> (N, L, H)
        last = memories[:, -1]  # (N, L, H) -> (N, H)
        logits = self.classifier(last)  # (N, H) -> (N, C)
        return {
            "logits": logits.detach(),
            "y": y.detach(),
        }

    def test_epoch_end(self, outputs: List[dict]):
        logits = torch.concat([out['logits'] for out in outputs], dim=0)  # noqa, num_batches * (N, C) -> (num_batches * N, C)
        y = torch.concat([out['y'] for out in outputs], dim=0)  # num_batches * (N,) -> (num_batches * N,)
        f1_score = mF.f1_score(preds=logits, target=y)
        accuracy = mF.accuracy(preds=logits, target=y)
        self.log("Test/f1_score", f1_score)
        self.log("Test/accuracy", accuracy)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Instantiates and returns the optimizer to be used for this model
        e.g. torch.optim.Adam
        """
        # The authors used Adam, so we might as well use it as well.
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])


class RNNForClassification(ClassificationBase):

    def __init__(self, vocab_size: int, hidden_size: int,
                 num_classes: int, lr: float, depth: int):  # noqa
        self.save_hyperparameters()
        super().__init__(RNN(vocab_size, hidden_size, depth), num_classes)


class LSTMForClassification(ClassificationBase):

    def __init__(self, vocab_size: int, hidden_size: int,
                 num_classes: int, lr: float, depth: int):  # noqa
        self.save_hyperparameters()
        super().__init__(LSTM(vocab_size, hidden_size, depth), num_classes)


class BiLSTMForClassification(ClassificationBase):

    def __init__(self, vocab_size: int, hidden_size: int,
                 num_classes: int, lr: float, depth: int):
        self.save_hyperparameters()
        super().__init__(BiLSTM(vocab_size, hidden_size, depth), num_classes)


class Seq2SeqBase(pl.LightningModule):
    """
    i.e. conditional generator.
    """
    pass


class NERBase(pl.LightningModule):
    pass
