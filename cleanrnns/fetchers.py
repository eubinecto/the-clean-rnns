import os
import yaml  # noqa
import wandb
from typing import Tuple
from tokenizers import Tokenizer
from wandb.sdk.wandb_run import Run
from cleanrnns.models import ClassificationBase, RNNForClassification, LSTMForClassification, BiLSTMForClassification
from cleanrnns.paths import CONFIG_YAML
from cleanrnns.pipelines import PipelineForClassification


# --- fetching from local --- #
def fetch_config() -> dict:
    with open(str(CONFIG_YAML), 'r', encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --- fetching from wandb --- #
def fetch_tokenizer(run: Run = None) -> Tokenizer:
    ver = fetch_config()['tokenizer']['ver']
    if run:
        artifact = run.use_artifact(f"tokenizer:{ver}", type="other")
    else:
        artifact = wandb.Api().artifact(f"the-clean-rnns/tokenizer:{ver}", type="other")
    artifact_path = artifact.download()
    json_path = os.path.join(artifact_path, "tokenizer.json")
    tokenizer = Tokenizer.from_file(json_path)
    # just manually register the special tokens
    tokenizer.pad_token = artifact.metadata['pad']
    tokenizer.unk_token = artifact.metadata['unk']
    tokenizer.bos_token = artifact.metadata['bos']
    tokenizer.eos_token = artifact.metadata['eos']
    return tokenizer


def fetch_nsmc(run: Run = None) -> Tuple[wandb.Table, wandb.Table, wandb.Table]:
    ver = fetch_config()['nsmc']['ver']
    if run:
        artifact = run.use_artifact(f"nsmc:{ver}", type="dataset")
    else:
        artifact = wandb.Api().artifact(f"the-clean-rnns/nsmc:{ver}", type="dataset")
    train = artifact.get("train")
    val = artifact.get("val")
    test = artifact.get("test")
    return train, val, test  # noqa


def fetch_model_for_classification(name: str, run: Run = None) -> ClassificationBase:
    ver = fetch_config()[name]['ver']
    if run:
        artifact = run.use_artifact(f"{name}:{ver}", type="model")
    else:
        artifact = wandb.Api().artifact(f"the-clean-rnns/{name}:{ver}", type="model")
    artifact_path = artifact.download()
    ckpt_path = os.path.join(artifact_path, "model.ckpt")
    if name == "rnn_for_classification":
        model = RNNForClassification.load_from_checkpoint(ckpt_path)
    elif name == "lstm_for_classification":
        model = LSTMForClassification.load_from_checkpoint(ckpt_path)
    elif name == "bilstm_for_classification":
        model = BiLSTMForClassification.load_from_checkpoint(ckpt_path)
    else:
        raise ValueError
    return model


def fetch_model_for_seq2seq():
    raise NotImplementedError


def fetch_model_for_ner():
    raise NotImplementedError


def fetch_pipeline_for_classification(name: str, run: Run = None) -> PipelineForClassification:
    model = fetch_model_for_classification(name, run)
    model.eval()
    tokenizer = fetch_tokenizer(run)
    config = fetch_config()[name]
    pipeline = PipelineForClassification(model, tokenizer, config)
    return pipeline


def fetch_pipeline_for_seq2seq(name: str, run: Run = None):  # noqa
    raise NotImplementedError


def fetch_pipeline_for_ner(name: str, run: Run = None):  # noqa
    raise NotImplementedError
