"""
여기는... Korpora의 데이터 모듈을 정의하는 것이 목표.
일단 시작은.. Naver Sentiment Movie Corpus로!
나중에 시간이 남는다면, 다른 말뭉치 지원도 생각해보자.
"""
from typing import Optional, Tuple
import pytorch_lightning as pl
import wandb
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, random_split
from wandb.sdk.wandb_run import Run
from builders import InputsForClassificationBuilder, LabelsForClassificationBuilder
from cleanrnns.fetchers import fetch_nsmc
from cleanrnns.datasets import DatasetForClassification


class NSMC(pl.LightningDataModule):
    """
    네이버 영화리뷰 긍/부정 데이터셋.
    """
    def __init__(self,
                 config: dict,
                 tokenizer: Tokenizer,
                 run: Run = None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.run = run
        # --- to be downloaded & built --- #
        self.nsmc: Optional[Tuple[wandb.Table, wandb.Table]] = None
        self.train: Optional[DatasetForClassification] = None
        self.val: Optional[DatasetForClassification] = None
        self.test: Optional[DatasetForClassification] = None

    def prepare_data(self):
        """
        prepare: download all data needed for this from wandb to local.
        """
        self.nsmc = fetch_nsmc(self.config['entity'], run=self.run)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = self.build_dataset(self.nsmc[0])
            self.val = self.build_dataset(self.nsmc[1])
        elif stage == "test" or stage is None:
            self.test = self.build_dataset(self.nsmc[2])
        else:
            raise NotImplementedError

    def build_dataset(self, table: wandb.Table) -> DatasetForClassification:
        X = InputsForClassificationBuilder(self.tokenizer, self.config['max_length'])(table.data)  # (N, L)
        y = LabelsForClassificationBuilder()(table.data)  # (N, L)
        return DatasetForClassification(X, y)  # noqa

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.config['batch_size'],
                          shuffle=self.config['shuffle'], num_workers=self.config['num_workers'])

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.config['batch_size'],
                          shuffle=False, num_workers=self.config['num_workers'])

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.config['batch_size'],
                          shuffle=False, num_workers=self.config['num_workers'])


class KHS(pl.LightningDataModule):
    """
    Korean Hate Speech dataset.
    한국어 혐오 데이터셋 - 다중분류 데이터셋.
    🚧 시간이 남으면 이 부분도 구현하기 🚧
    https://ko-nlp.github.io/Korpora/ko-docs/corpuslist/korean_hate_speech.html
    """
    pass
