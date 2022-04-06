import os
import wandb
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from cleanrnns.datamodules import NSMC
from cleanrnns.fetchers import fetch_config, fetch_tokenizer, fetch_model_for_classification
from cleanrnns.paths import ROOT_DIR
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entity", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    args = parser.parse_args()
    # prepare the datamodule
    config = fetch_config()[args.model]
    config.update(vars(args))

    with wandb.init(entity=config['entity'], project="the-clean-rnns", config=config) as run:
        # --- prepare a pre-trained tokenizer & a module to train --- #
        tokenizer = fetch_tokenizer(config['entity'], run)
        model = fetch_model_for_classification(config['entity'], config['model'])
        datamodule = NSMC(config, tokenizer, run)
        logger = WandbLogger(log_model=False)
        trainer = pl.Trainer(fast_dev_run=config['fast_dev_run'],
                             gpus=torch.cuda.device_count(),
                             default_root_dir=str(ROOT_DIR),
                             enable_checkpointing=False,
                             logger=logger)
        # start testing
        trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    main()
