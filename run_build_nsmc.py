import argparse
import pandas as pd
import wandb
from Korpora import Korpora, NSMCKorpus
from fetchers import fetch_config
from cleanrnns.preprocess import cleanse, stratified_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entity", type=str)
    args = parser.parse_args()
    config = fetch_config()["nsmc"]
    config.update(vars(args))
    Korpora.fetch("nsmc")
    nsmc = NSMCKorpus()
    train_df = pd.DataFrame([(example.text, example.label) for example in nsmc.train], columns=["text", "label"])
    test_df = pd.DataFrame([(example.text, example.label) for example in nsmc.test], columns=["text", "label"])
    # preprocessing
    test_df = test_df.pipe(cleanse)
    # we construct a validation set here
    train_df, val_df = train_df.pipe(cleanse)\
                               .pipe(stratified_split, ratio=config['val_ratio'], seed=config['seed'])
    train = wandb.Table(data=train_df)
    val = wandb.Table(data=val_df)
    test = wandb.Table(data=test_df)
    with wandb.init(entity=config['entity'], project="the-clean-rnns", config=config) as run:
        artifact = wandb.Artifact(name="nsmc", type="dataset", metadata=config, description=config['desc'])
        artifact.add(train, name="train")
        artifact.add(val, name="val")
        artifact.add(test, name="test")
        run.log_artifact(artifact, aliases=["latest", config['ver']])


if __name__ == '__main__':
    main()
