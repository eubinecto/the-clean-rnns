import argparse
import wandb
from Korpora import Korpora, NSMCKorpus
from fetchers import fetch_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entity", type=str)
    args = parser.parse_args()
    config = fetch_config()["nsmc"]
    config.update(vars(args))
    Korpora.fetch("nsmc")
    nsmc = NSMCKorpus()
    train_data = [(example.text, example.label) for example in nsmc.train]
    test_data = [(example.text, example.label) for example in nsmc.test]
    train = wandb.Table(data=train_data, columns=["text", "label"])
    test = wandb.Table(data=test_data, columns=["text", "label"])
    with wandb.init(entity=config['entity'], project="the-clean-rnns", config=config) as run:
        artifact = wandb.Artifact(name="nsmc", type="dataset", metadata=config, description=config['desc'])
        artifact.add(train, name="train")
        artifact.add(test, name="test")
        run.log_artifact(artifact, aliases=["latest", config['ver']])


if __name__ == '__main__':
    main()
