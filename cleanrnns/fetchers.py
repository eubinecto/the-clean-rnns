import yaml
import wandb
from Korpora import NSMCKorpus, Korpora
from tokenizers import Tokenizer
from wandb.sdk.wandb_run import Run
from models import RNN
from paths import CONFIG_YAML, tokenizer_dir


# --- fetching from wandb --- #
def fetch_tokenizer(entity: str, ver: str, run: Run = None) -> Tokenizer:
    artifact = wandb.Api().artifact(f"{entity}/the-clean-rnns/tokenizer:{ver}", type="other")
    artifact_path = tokenizer_dir(ver)
    json_path = artifact_path / "tokenizer.json"
    artifact.download(root=str(artifact_path))
    tokenizer = Tokenizer.from_file(str(json_path))
    # just manually register the special tokens
    tokenizer.pad_token = artifact.metadata['pad']
    tokenizer.unk_token = artifact.metadata['unk']
    tokenizer.bos_token = artifact.metadata['bos']
    tokenizer.eos_token = artifact.metadata['eos']
    return tokenizer


def fetch_rnn(entity: str, ver: str, run: Run = None) -> RNN:
    pass


# --- fetching from local --- #
def fetch_config() -> dict:
    with open(str(CONFIG_YAML), 'r', encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def fetch_nsmc() -> NSMCKorpus:
    Korpora.fetch("nsmc")
    return NSMCKorpus()
