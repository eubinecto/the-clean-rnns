import shutil
from fetchers import fetch_model_for_classification


def main():
    model = fetch_model_for_classification("rnn_for_classification")
    print(model.hparams)  # just a sanity check
    shutil.rmtree("artifacts")  # clear the cache after testing


if __name__ == '__main__':
    main()
