import os
import yaml
import wandb
from typing import Tuple
from tokenizers import Tokenizer
from wandb.sdk.wandb_run import Run
from cleanrnns.models import RNNForClassification
from cleanrnns.paths import CONFIG_YAML


# --- fetching from local --- #
def fetch_config() -> dict:
    with open(str(CONFIG_YAML), 'r', encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --- fetching from wandb --- #
def fetch_tokenizer(entity: str, run: Run = None) -> Tokenizer:
    ver = fetch_config()['tokenizer']['ver']
    if run:
        artifact = run.use_artifact(f"tokenizer:{ver}", type="other")
    else:
        artifact = wandb.Api().artifact(f"{entity}/the-clean-rnns/tokenizer:{ver}", type="other")
    artifact_path = artifact.download()
    json_path = os.path.join(artifact_path, "tokenizer.json")
    tokenizer = Tokenizer.from_file(json_path)
    # just manually register the special tokens
    tokenizer.pad_token = artifact.metadata['pad']
    tokenizer.unk_token = artifact.metadata['unk']
    tokenizer.bos_token = artifact.metadata['bos']
    tokenizer.eos_token = artifact.metadata['eos']
    return tokenizer


def fetch_nsmc(entity: str, run: Run = None) -> Tuple[wandb.Table, wandb.Table, wandb.Table]:
    ver = fetch_config()['nsmc']['ver']
    if run:
        artifact = run.use_artifact(f"nsmc:{ver}", type="dataset")
    else:
        artifact = wandb.Api().artifact(f"{entity}/the-clean-rnns/nsmc:{ver}", type="dataset")
    train = artifact.get("train")
    val = artifact.get("val")
    test = artifact.get("test")
    return train, val, test  # noqa


def fetch_rnn_for_classification(entity: str, run: Run = None) -> RNNForClassification:
    pass
