import shutil
from fetchers import fetch_tokenizer


def main():
    tokenizer = fetch_tokenizer()
    # retrieve the registered special tokens
    print(tokenizer.pad_token)  # noqa
    print(tokenizer.token_to_id(tokenizer.pad_token))  # noqa
    print(tokenizer.unk_token)  # noqa
    print(tokenizer.token_to_id(tokenizer.unk_token))  # noqa
    print(tokenizer.bos_token)  # noqa
    print(tokenizer.token_to_id(tokenizer.bos_token))  # noqa
    print(tokenizer.eos_token)  # noqa
    print(tokenizer.token_to_id(tokenizer.eos_token))  # noqa
    print(tokenizer.get_vocab_size())  # vocab size?
    shutil.rmtree("artifacts")  # clear the cache after testing


if __name__ == '__main__':
    main()
