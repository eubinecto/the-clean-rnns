"""
https://pytorch.org/docs/stable/generated/torch.fliplr.html
"""
import torch  # noqa


def main():
    x = torch.Tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])
    print(torch.fliplr(x))


if __name__ == '__main__':
    main()