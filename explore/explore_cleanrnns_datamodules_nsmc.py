import os
from cleanrnns.datamodules import NSMC
from fetchers import fetch_config, fetch_tokenizer

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def main():
    config = fetch_config()
    config['rnn']['num_workers'] = os.cpu_count()
    tokenizer = fetch_tokenizer("eubinecto", config['tokenizer']['ver'])
    datamodule = NSMC(config['rnn'], tokenizer)
    datamodule.prepare_data()
    datamodule.setup()  # this will tak some time
    print("--- A batch from the training set ---")
    for batch in datamodule.train_dataloader():
        X, y = batch
        print(X)
        print(X.shape)
        print(y)
        print(y.shape)
        break

    print("--- A batch from the validation set ---")
    for batch in datamodule.val_dataloader():
        X, y = batch
        print(X)
        print(X.shape)
        print(y)
        print(y.shape)
        break


if __name__ == '__main__':
    main()
