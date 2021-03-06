"""
A script for building a BPE tokenizer.
# for more details on using a different algorithm (e.g. WordPiece) refer to the tutorial below.
https://huggingface.co/docs/tokenizers/python/latest/pipeline.html
# for more details on normalizers, pre/post & tokenizers
https://huggingface.co/docs/tokenizers/python/latest/components.html
"""
import os
from itertools import chain
import wandb
from tokenizers import pre_tokenizers, normalizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace, Digits, Punctuation
from tokenizers.trainers import BpeTrainer
from cleanrnns.paths import ROOT_DIR
from cleanrnns.fetchers import fetch_config, fetch_nsmc
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def main():
    config = fetch_config()["tokenizer"]
    # --- prepare a tokenizer --- #
    special_tokens = [config['pad'], config['unk'], config['bos'], config['eos']]
    # tokenizer = Tokenizer(BPE(unk_token=config['unk']))
    tokenizer = Tokenizer(BPE(unk_token=config['unk']))
    trainer = BpeTrainer(vocab_size=config['vocab_size'],
                         special_tokens=special_tokens)
    # --- pre & post processing --- #
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(), Punctuation()])  # noqa
    tokenizer.normalizer = normalizers.Sequence([Lowercase()])  # noqa
    with wandb.init(project="the-clean-rnns", config=config) as run:
        # --- prepare the data --- #
        train, val, test = fetch_nsmc(run)
        iterator = chain((row[0] for row in train.data),
                         (row[0] for row in val.data),
                         (row[0] for row in test.data))
        # --- train the tokenizer --- #
        tokenizer.train_from_iterator(iterator, trainer=trainer)
        # --- save to local, and then to wandb --- #
        json_path = ROOT_DIR / "tokenizer.json"
        tokenizer.save(str(json_path), pretty=True)  # noqa
        artifact = wandb.Artifact(name="tokenizer", type="other", metadata=config, description=config['desc'])
        artifact.add_file(str(json_path))
        run.log_artifact(artifact, aliases=["latest", config['ver']])
        os.remove(str(json_path))  # make sure you delete it after you are done with uploading it


if __name__ == '__main__':
    main()
