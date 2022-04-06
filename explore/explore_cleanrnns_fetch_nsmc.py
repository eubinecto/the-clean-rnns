import shutil
from fetchers import fetch_nsmc


def main():
    train, _, _ = fetch_nsmc(entity="eubinecto")
    for row in train.data:
        print(row[0], row[1])
    shutil.rmtree("artifacts")  # clear the cache after testing


if __name__ == '__main__':
    main()
