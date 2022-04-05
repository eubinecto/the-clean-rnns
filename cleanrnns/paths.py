from pathlib import Path
# --- directories --- #
ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


# --- directories for artifacts --- #
def tokenizer_dir(ver: str) -> Path:
    return ARTIFACTS_DIR / f"tokenizer-{ver}"


def rnn_dir(ver: str) -> Path:
    return ARTIFACTS_DIR / f"rnn-{ver}"


# --- local files --- #
CONFIG_YAML = ROOT_DIR / "config.yaml"


