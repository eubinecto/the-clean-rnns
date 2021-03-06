import os
import shutil
from cleanrnns.datamodules import NSMC
from fetchers import fetch_config, fetch_tokenizer
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def main():
    config = fetch_config()['rnn_for_classification']
    config['num_workers'] = os.cpu_count()
    config['entity'] = "eubinecto"
    tokenizer = fetch_tokenizer()
    datamodule = NSMC(config, tokenizer)
    datamodule.prepare_data()
    datamodule.setup()  # this will tak some time
    print("--- A batch from the training set ---")
    for batch in datamodule.train_dataloader():
        x, y = batch
        print(x)  # (N, L)
        print(x.shape)
        print(y)  # (N,)
        print(y.shape)
        break

    print("--- A batch from the validation set ---")
    for batch in datamodule.val_dataloader():
        x, y = batch
        print(x)  # (N, L)
        print(x.shape)
        print(y)  # (N,)
        print(y.shape)
        break

    shutil.rmtree("artifacts")  # clear the cache after testing


if __name__ == '__main__':
    main()
