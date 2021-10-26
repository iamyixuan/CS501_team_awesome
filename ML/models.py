import torch
import torch.nn as nn

"""
Models:
    RNN: standard RNN
        args:
            input_size: input sequence length
            hidden_size: number of hidden states
            output_size: output lenght
    RNN with attention:
    Transformer:
"""

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.act = nn.Softmax()

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[:, -1, ...]
        x = self.linear(x)
        x = self.act(x)
        return x


if __name__ == "__main__":
    net = RNN(1, 3, 2, 2, 1)
    x = torch.ones((10, 12, 1))
    out = net(x)
    print(out)