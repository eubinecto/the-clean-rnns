from fetchers import fetch_tokenizer, fetch_config


def main():
    config = fetch_config()
    tokenizer = fetch_tokenizer(entity="eubinecto", ver=config['tokenizer']['ver'])
    # retrieve the registered special tokens
    print(tokenizer.pad_token)  # noqa
    print(tokenizer.token_to_id(tokenizer.pad_token))  # noqa
    print(tokenizer.unk_token)  # noqa
    print(tokenizer.token_to_id(tokenizer.unk_token))  # noqa
    print(tokenizer.bos_token)  # noqa
    print(tokenizer.token_to_id(tokenizer.bos_token))  # noqa
    print(tokenizer.eos_token)  # noqa
    print(tokenizer.token_to_id(tokenizer.eos_token))  # noqa


if __name__ == '__main__':
    main()
