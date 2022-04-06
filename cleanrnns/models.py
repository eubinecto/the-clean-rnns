import torch


class RNN(torch.nn.Module):
    """
    A simple RNN.
    """
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.W_hh = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)  # (H, H)
        self.W_xh = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)  # (H, H)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        h_t = f_W(h_t-1, x_t)
        h_t = tanh(W_hh * h_t-1 + W_xh * x_t)
        :param X: (N, L)
        :return: (N, L, H)
        """
        N, L = X.shape
        H_all = list()
        H_t = torch.zeros(size=(N, self.hidden_size))  # (N, H)
        for t in range(L):
            E_t = self.embeddings(X[:, t])  # (N, L) -> (N, 1) -> (N, H)
            H_t = torch.tanh(self.W_hh(H_t) + self.W_xh(E_t))  # ... -> (N, H)
            H_all.append(H_t)
        # many to one (sentiment analysis)
        return torch.stack(H_all, dim=1)  # a list of N, H -> (N, L, H)


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
