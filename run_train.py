import os
import wandb
import argparse
import torch
import pytorch_lightning as pl
from termcolor import colored
from pytorch_lightning.loggers import WandbLogger
from cleanrnns.models import RNNForClassification
from cleanrnns.datamodules import NSMC
from cleanrnns.fetchers import fetch_config, fetch_tokenizer
from cleanrnns.paths import ROOT_DIR
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entity", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    parser.add_argument("--upload", dest='upload', action='store_true', default=False)
    args = parser.parse_args()
    # prepare the datamodule
    config = fetch_config()[args.model]
    config.update(vars(args))
    if not config['upload']:
        print(colored("WARNING: YOU CHOSE NOT TO UPLOAD. NOTHING BUT LOGS WILL BE SAVED TO WANDB", color="red"))
    with wandb.init(entity=config['entity'], project="the-clean-rnns", config=config) as run:
        # --- prepare a pre-trained tokenizer & a module to train --- #
        if config['model'] == "rnn_for_classification":
            tokenizer = fetch_tokenizer(config['entity'], run)
            model = RNNForClassification(tokenizer.get_vocab_size(), config['hidden_size'],
                                         config['num_classes'], config['lr'], config['depth'])
            datamodule = NSMC(config, tokenizer, run)
        elif config['model'] == "lstm_for_classification":
            raise NotImplementedError
        elif config['model'] == "bilstm_for_classification":
            raise NotImplementedError
        elif config['model'] == "bilstmsearch_for_classification":
            raise NotImplementedError
        else:
            raise ValueError
        logger = WandbLogger(log_model=False)
        trainer = pl.Trainer(max_epochs=config['max_epochs'],
                             fast_dev_run=config['fast_dev_run'],
                             log_every_n_steps=config['log_every_n_steps'],
                             gpus=torch.cuda.device_count(),
                             default_root_dir=str(ROOT_DIR),
                             enable_checkpointing=False,
                             logger=logger)
        # start training
        trainer.fit(model=model, datamodule=datamodule)
        # upload the model to wandb only if the training is properly done  #
        if not config['fast_dev_run'] and trainer.current_epoch == config['max_epochs'] - 1:
            ckpt_path = ROOT_DIR / "model.ckpt"
            trainer.save_checkpoint(str(ckpt_path))
            artifact = wandb.Artifact(name=config['model'], type="model", metadata=config)
            artifact.add_file(str(ckpt_path))
            run.log_artifact(artifact, aliases=["latest", config['ver']])
            os.remove(str(ckpt_path))


if __name__ == '__main__':
    main()
