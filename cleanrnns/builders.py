import torch
from abc import ABC
from typing import List
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

    def __call__(self, split: LabeledSentenceKorpusData) -> torch.Tensor:
        """
        :param split:
        :return: X (N, L)
        """
        texts = [example.text for example in split]
        pad_token = self.tokenizer.pad_token  # noqa
        self.tokenizer.enable_padding(pad_token=pad_token,
                                      pad_id=self.tokenizer.token_to_id(pad_token),  # noqa
                                      length=self.max_length)
        # don't add special tokens, we will add them ourselves
        encodings: List[Encoding] = self.tokenizer.encode_batch(texts)
        X = torch.LongTensor([encoding.ids for encoding in encodings])
        return X


class LabelsForClassificationBuilder(TensorBuilder):

    def __call__(self, split: LabeledSentenceKorpusData) -> torch.Tensor:
        """
        :param split:
        :return: y (N,)
        """
        labels = [example.label for example in split]
        y = torch.IntTensor(labels)
        return y


# --- to be implemented --- #
class InputsForConditionalGenerationBuilder(InputsBuilder):

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class LabelsForConditionalGenerationBuilder(TensorBuilder):
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class InputsForNERBuilder(InputsBuilder):

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class LabelsForNERBuilder(TensorBuilder):
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

