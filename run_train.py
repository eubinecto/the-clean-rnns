import os
import wandb
import argparse
import torch  # noqa
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from cleanrnns.modules import RNNForClassification, LSTMForClassification, BiLSTMForClassification
from cleanrnns.datamodules import NSMC
from cleanrnns.fetchers import fetch_config, fetch_tokenizer
from cleanrnns.paths import ROOT_DIR
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("module", type=str)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--limit_train_batches", type=float, default=None)
    parser.add_argument("--limit_val_batches", type=float, default=None)
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    args = parser.parse_args()
    # prepare the datamodule
    config = fetch_config()[args.module]
    config.update(vars(args))
    with wandb.init(project="the-clean-rnns", config=config) as run:
        # --- prepare the tokenizer & dataset to use for training --- #
        tokenizer = fetch_tokenizer(run)
        datamodule = NSMC(config, tokenizer, run)
        # --- prepare the module to train --- #
        if config['module'] == "rnn_for_classification":
            module = RNNForClassification(tokenizer.get_vocab_size(), config['hidden_size'],
                                          config['num_classes'], config['lr'], config['depth'])
        elif config['module'] == "lstm_for_classification":
            module = LSTMForClassification(tokenizer.get_vocab_size(), config['hidden_size'],
                                           config['num_classes'], config['lr'], config['depth'])
        elif config['module'] == "bilstm_for_classification":
            module = BiLSTMForClassification(tokenizer.get_vocab_size(), config['hidden_size'],
                                             config['num_classes'], config['lr'], config['depth'])
        else:
            raise ValueError
        # --- prepare a trainer instance --- #
        logger = WandbLogger(log_module=False)
        trainer = pl.Trainer(max_epochs=config['max_epochs'],
                             fast_dev_run=config['fast_dev_run'],
                             log_every_n_steps=config['log_every_n_steps'],
                             limit_train_batches=config['limit_train_batches'],
                             limit_val_batches=config['limit_val_batches'],
                             gpus=torch.cuda.device_count(),
                             default_root_dir=str(ROOT_DIR),
                             enable_checkpointing=False,
                             logger=logger)
        # --- start training --- #
        trainer.fit(model=module, datamodule=datamodule)
        # --- upload the module to wandb only if the training is properly done --- #
        if not config['fast_dev_run'] and not trainer.interrupted:
            ckpt_path = ROOT_DIR / "module.ckpt"
            trainer.save_checkpoint(str(ckpt_path))
            artifact = wandb.Artifact(name=config['module'], type="module", metadata=config)
            artifact.add_file(str(ckpt_path))
            run.log_artifact(artifact, aliases=["latest", config['ver']])
            os.remove(str(ckpt_path))


if __name__ == '__main__':
    main()
