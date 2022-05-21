import torch


def main():
    device = torch.device("mps")
    print(device)
    print(type(device))


if __name__ == '__main__':
    main()