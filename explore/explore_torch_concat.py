
import torch


def main():
    outputs = 3 * [torch.rand(size=(10, 3))]
    logits = torch.concat(outputs, dim=0)  # noqa, num_batches * (N, C) -> (num_batches * N, C)
    print(logits.shape)


if __name__ == '__main__':
    main()