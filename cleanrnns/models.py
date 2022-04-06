import pytorch_lightning as pl
from typing import Union, Tuple, List
import torch
from torch.nn import functional as F
from torchmetrics import functional as mF
from cleanrnns.rnns import RNN, LSTM, BiLSTM, BiLSTMSearch


# --- lightning modules --- #
class ClassificationBase(pl.LightningModule):
    def __init__(self, encoder: Union[RNN, LSTM, BiLSTM, BiLSTMSearch], num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Linear(self.hparams['hidden_size'], num_classes)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args) -> dict:
        X, y = batch
        H_all = self.encoder(X)  # (N, L) -> (N, L, H)
        H_last = H_all[:, -1]  # (N, L, H) -> (N, H)
        logits = self.classifier(H_last)  # (N, H) -> (N, C)
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
                 num_classes: int, lr: float, depth: int):
        self.save_hyperparameters()
        super().__init__(LSTM(), num_classes)
        raise NotImplementedError


class BiLSTMForClassification(ClassificationBase):

    def __init__(self, vocab_size: int, hidden_size: int,
                 num_classes: int, lr: float, depth: int):
        self.save_hyperparameters()
        super().__init__(BiLSTM(), num_classes)
        raise NotImplementedError


class BiLSTMSearchForClassification(ClassificationBase):

    def __init__(self, vocab_size: int, hidden_size: int,
                 num_classes: int, lr: float, depth: int):
        self.save_hyperparameters()
        super().__init__(BiLSTMSearch(), num_classes)
        raise NotImplementedError


class Seq2SeqBase(pl.LightningModule):
    """
    i.e. conditional generator.
    """
    pass


class NERBase(pl.LightningModule):
    pass
