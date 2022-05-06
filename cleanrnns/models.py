"""
A simple implementation of the RNN family - RNN, LSTM, BiLSTM, BiLSTMSearch.
"""
import torch  # noqa
from typing import Tuple


class RNNFamily(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, cells: torch.nn.Sequential):
        super().__init__()
        self.embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.cells = cells

    def forward(self, x: torch.Tensor):
        """
        :param x: (N, L)
        :return: memories (N, L, H)
        """
        x = self.embeddings(x)  # (N, L) -> (N, L, H)
        x = self.cells(x)
        return x


class RNNCell(torch.nn.Module):
    """
    RNN Cell with
    weights = 2 * H^2
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.weights = torch.nn.Linear(hidden_size * 2, hidden_size)  # (H * 2, H)
        self.register_buffer("dummy", torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        h_t = f_W(h_t-1(short), x_t(now))
        h_t = tanh(W_hh * h_t-1 + W_xh * x_t)
=        :param x - (N, L, H)
        :return: memories - (N, L, H)
        """
        N, L, H = x.shape
        memories = list()
        short = self.dummy.unsqueeze(0).expand(N, -1)  # (H) -> (1, H) ->  (N, H)
        for time in range(L):
            now = x[:, time]  # (N, L, H) -> (N, H)
            short_and_now = torch.concat([short, now], dim=-1)  # (N, H), (N, H) -> (N, H * 2)
            # why? A * W_A + B * W_B = A_cat_B * W_A_cat_W_B
            short = torch.tanh(self.weights(short_and_now))  # (N, H * 2) * (H * 2, H)  -> (N, H)
            memories.append(short)
        return torch.stack(memories, dim=1)  # ... -> (N, L, H)


class RNN(RNNFamily):
    """
    A vanilla  multi-layer RNN.
    H * H * 2 + V * H = 2*H^2 + V*H = H(2H + V)
    https://medium.com/ecovisioneth/building-deep-multi-layer-recurrent-neural-networks-with-star-cell-2f01acdb73a7
    """

    def __init__(self, vocab_size: int, hidden_size: int, depth: int):
        super().__init__(vocab_size, hidden_size,
                         cells=torch.nn.Sequential(*[RNNCell(hidden_size)for _ in range(depth)]))


class LSTMCell(torch.nn.Module):
    """
    weights = 2 * H * H * 4 = 8 * H^2. (4 * RNNCell)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.weights = torch.nn.Linear(in_features=hidden_size * 2, out_features=hidden_size * 4)
        self.register_buffer("dummy", torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x - (N, L, H)
        :return: memories (N, L, H)
        """
        N, L, H = x.shape
        memories = list()
        long = self.dummy.unsqueeze(0).expand(N, -1)  # (H) -> (1, H) ->  (N, H)
        short = self.dummy.unsqueeze(0).expand(N, -1)  # (H) -> (1, H) ->  (N, H)
        for time in range(L):
            now = x[:, time]  # (N, L, H) -> (N, H)
            short_and_now = torch.concat([short, now], dim=-1)  # (N, H), (N, H) -> (N, H * 2)
            gates = self.weights(short_and_now)  # (N, H * 2) *  (H * 2, H * 4) -> (N, H * 4)
            f = torch.sigmoid(gates[:, :H])  # (N, H * 4) -> (N, H) -> (N, H)
            i = torch.sigmoid(gates[:, H:H*2])  # (N, H * 4) * (N, H) -> (N, H)
            o = torch.sigmoid(gates[:, H*2:H*3])  # (N, H * 4) * (N, H) -> (N, H)
            h = torch.tanh(gates[L, H*3:H*4])  # (N, H * 4) * (N, H) -> (N, H)
            # forget parts of long-term memory, while adding parts of short-term memory to long-term memory
            long = torch.mul(f, long) + torch.mul(i, h)  # (N, H) + (N, H) -> (N, H)
            # generate short-term memory from parts of long-term memory
            short = torch.mul(o, torch.tanh(long))  # (N, H) + (N, H) -> (N, H)
            memories.append(short)
        return torch.stack(memories, dim=1)  # ... -> (N, L, H)


class LSTM(RNNFamily):
    """
    A simple, multi-layer LSTM.
    weights = H(8H + V)
    """

    def __init__(self, vocab_size: int, hidden_size: int, depth: int):
        super().__init__(vocab_size, hidden_size,
                         cells=torch.nn.Sequential(*[LSTMCell(hidden_size) for _ in range(depth)]))


class BiLSTMCell(torch.nn.Module):
    """
    weights = 2 * 8 * H^2 = 16 * H^2
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.lr = LSTMCell(hidden_size)
        self.rl = LSTMCell(hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        memories = self.lr(x)  # (N, L) -> (N, L, H)
        memories = self.rl(torch.fliplr(memories))  # (N, L) -> (N, L, H)
        return memories


class BiLSTM(RNNFamily):
    """
    A simple, multi-layer LSTM.
    weights = H(16H + V)
    """

    def __init__(self, vocab_size: int, hidden_size: int, depth: int):
        super().__init__(vocab_size, hidden_size,
                         cells=torch.nn.Sequential(*[BiLSTMCell(hidden_size) for _ in range(depth)]))
