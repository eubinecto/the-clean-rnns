import shutil
from fetchers import fetch_pipeline_for_classification


def main():
    pipeline = fetch_pipeline_for_classification("rnn_for_classification")
    print(pipeline("너무 좋다"))  # just a sanity check
    print(pipeline("재미없다"))  # just a sanity check
    pipeline = fetch_pipeline_for_classification("lstm_for_classification")
    print(pipeline("너무 좋다"))  # just a sanity check
    print(pipeline("재미없다"))  # just a sanity check
    # 향후 테스트를 진행할 계획
    pipeline = fetch_pipeline_for_classification("bilstm_for_classification")
    print(pipeline("너무 좋다"))  # just a sanity check
    print(pipeline("재미없다"))  # just a sanity check
    shutil.rmtree("artifacts")  # clear the cache after testing


if __name__ == '__main__':
    main()
