import torch  # noqa
from typing import List, Tuple, Optional
from tokenizers import Tokenizer, Encoding


def inputs_for_classification(text2label: List[Tuple[str, Optional[int]]], max_length: int, tokenizer: Tokenizer) -> torch.Tensor:
    texts = [text for text, _ in text2label]
    pad_token = tokenizer.pad_token  # noqa
    tokenizer.enable_padding(pad_token=pad_token,
                             pad_id=tokenizer.token_to_id(pad_token),  # noqa
                             length=max_length)
    # don't add special tokens, we will add them ourselves
    encodings: List[Encoding] = tokenizer.encode_batch(texts)
    x = torch.LongTensor([encoding.ids for encoding in encodings])
    return x


def labels_for_classification(text2label: List[Tuple[str, Optional[int]]]) -> torch.Tensor:
    y = torch.LongTensor([
        label
        for _, label in text2label
    ])
    return y


def inputs_for_seq2seq():
    raise NotImplementedError


def labels_for_seq2seq():
    raise NotImplementedError


def inputs_for_ner():
    raise NotImplementedError


def labels_for_ner():
    raise NotImplementedError
