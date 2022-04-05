"""
A script for building a BPE tokenizer.
# for more details on using a different algorithm (e.g. WordPiece) refer to the tutorial below.
https://huggingface.co/docs/tokenizers/python/latest/pipeline.html
# for more details on normalizers, pre/post & tokenizers
https://huggingface.co/docs/tokenizers/python/latest/components.html
"""
import os
import wandb
import argparse
from itertools import chain
from tokenizers import pre_tokenizers, normalizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace, Digits, Punctuation
from tokenizers.trainers import BpeTrainer
from cleanrnns.paths import ROOT_DIR
from cleanrnns.fetchers import fetch_config, fetch_nsmc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entity", type=str)
    args = parser.parse_args()
    config = fetch_config()["tokenizer"]
    config.update(vars(args))
    # --- prepare a tokenizer --- #
    special_tokens = [config['pad'], config['unk'], config['bos'], config['eos']]
    # tokenizer = Tokenizer(BPE(unk_token=config['unk']))
    tokenizer = Tokenizer(BPE(unk_token=config['unk']))
    trainer = BpeTrainer(vocab_size=config['vocab_size'],
                         special_tokens=special_tokens)
    # --- pre & post processing --- #
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(), Punctuation()])  # noqa
    tokenizer.normalizer = normalizers.Sequence([Lowercase()])  # noqa
    # --- prepare the data --- #
    nsmc = fetch_nsmc()
    # chaining two generators;  https://stackoverflow.com/a/3211047
    iterator = chain((example.text for example in nsmc.train),
                     (example.text for example in nsmc.test))
    # --- train the tokenizer --- #
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    # --- then save it --- #
    with wandb.init(entity=config['entity'], project="the-clean-rnns", config=config) as run:
        # save to local, and then to wandb
        json_path = ROOT_DIR / "tokenizer.json"
        tokenizer.save(str(json_path), pretty=True)  # noqa
        artifact = wandb.Artifact(name="tokenizer", type="other", metadata=config)
        artifact.add_file(str(json_path))
        run.log_artifact(artifact, aliases=["latest", config['ver']])
        os.remove(str(json_path))  # make sure you delete it after you are done with uploading it


if __name__ == '__main__':
    main()
