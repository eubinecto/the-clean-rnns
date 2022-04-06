import torch
from abc import ABC
from typing import List, Tuple
from Korpora.korpora import LabeledSentenceKorpusData
from tokenizers import Tokenizer, Encoding


class TensorBuilder:

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class InputsBuilder(TensorBuilder, ABC):
    def __init__(self, tokenizer: Tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length


class InputsForClassificationBuilder(InputsBuilder):

    def __call__(self, text2label: List[Tuple[str, int]]) -> torch.Tensor:
        """
        :return: X (N, L)
        """
        texts = [text for text, _ in text2label]
        pad_token = self.tokenizer.pad_token  # noqa
        self.tokenizer.enable_padding(pad_token=pad_token,
                                      pad_id=self.tokenizer.token_to_id(pad_token),  # noqa
                                      length=self.max_length)
        # don't add special tokens, we will add them ourselves
        encodings: List[Encoding] = self.tokenizer.encode_batch(texts)
        X = torch.LongTensor([encoding.ids for encoding in encodings])
        return X


class LabelsForClassificationBuilder(TensorBuilder):

    def __call__(self, text2label: List[Tuple[str, int]]) -> torch.Tensor:
        """
        :return: y (N,)
        """
        y = torch.LongTensor([
            label
            for _, label in text2label
        ])
        return y


# --- to be implemented --- #
class InputsForSeq2SeqBuilder(InputsBuilder):

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class LabelsForSeq2SeqBuilder(TensorBuilder):
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class InputsForNERBuilder(InputsBuilder):

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class LabelsForNERBuilder(TensorBuilder):
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

