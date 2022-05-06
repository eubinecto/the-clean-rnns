import pandas as pd
import wandb
from Korpora import Korpora, NSMCKorpus
from cleanrnns.fetchers import fetch_config
from cleanrnns.preprocess import cleanse, stratified_split


def main():
    config = fetch_config()["nsmc"]
    Korpora.fetch("nsmc")
    nsmc = NSMCKorpus()
    train_df = pd.DataFrame([(example.text, example.label) for example in nsmc.train], columns=["text", "label"])
    val_df, train_df = train_df.pipe(cleanse) \
                               .pipe(stratified_split, ratio=config['val_ratio'], seed=config['seed'])
    test_df = pd.DataFrame([(example.text, example.label) for example in nsmc.test], columns=["text", "label"])
    test_df = test_df.pipe(cleanse)
    train = wandb.Table(data=train_df)
    val = wandb.Table(data=val_df)
    test = wandb.Table(data=test_df)
    with wandb.init(project="the-clean-rnns", config=config) as run:
        artifact = wandb.Artifact(name="nsmc", type="dataset", metadata=config, description=config['desc'])
        artifact.add(train, name="train")
        artifact.add(val, name="val")
        artifact.add(test, name="test")
        run.log_artifact(artifact, aliases=["latest", config['ver']])


if __name__ == '__main__':
    main()
