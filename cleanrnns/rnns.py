"""
A simple implementation of the RNN family - RNN, LSTM, BiLSTM, BiLSTMSearch.
"""
import torch


class RNNCell(torch.nn.Module):
    def __init__(self, embeddings: torch.nn.Embedding, hidden_size: int):
        super().__init__()
        self.embeddings = embeddings
        self.W_hh = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.W_xh = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.register_buffer("dummy", torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        h_t = f_W(h_t-1(past), x_t(now))
        h_t = tanh(W_hh * h_t-1 + W_xh * x_t)
        :param x - (N, L)
        :return: memories - (N, L, H)
        """
        N, L = x.shape
        memories = list()
        past = self.dummy.unsqueeze(0).expand(N, -1)  # (H) -> (1, H) ->  (N, H)
        for time in range(L):
            now = self.embeddings(x[:, time])  # (N, L) -> (N, 1) -> (N, H)
            past = torch.tanh(self.W_hh(past) + self.W_xh(now))  # ... -> (N, H)
            memories.append(past)
        return torch.stack(memories, dim=1)  # ... -> (N, L, H)


class RNN(torch.nn.Module):
    """
    A vanilla  multi-layer RNN.
    Complexity - O(L * D)
    https://medium.com/ecovisioneth/building-deep-multi-layer-recurrent-neural-networks-with-star-cell-2f01acdb73a7
    """
    def __init__(self, vocab_size: int, hidden_size: int, depth: int):
        super().__init__()
        self.embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.cells = torch.nn.ModuleList([RNNCell(self.embeddings, hidden_size) for _ in range(depth)])

    def forward(self, x: torch.Tensor):
        for cell in self.cells:
            x = cell(x)
        return x


class LSTM(torch.nn.Module):
    """
    🚧 공사중 🚧
    """
    pass


class BiLSTM(torch.nn.Module):
    """
    🚧 공사중 🚧
    """
    pass


class BiLSTMSearch(torch.nn.Module):
    """
    🚧 공사중 🚧
    """
    pass
