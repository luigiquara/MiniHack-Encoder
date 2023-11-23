import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_size, cfg):
        super(Encoder, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_layers = nn.ModuleList()
        for i in range(cfg.num_layers - 1):
            self.linear_layers.append(nn.Linear(input_size, input_size//2))
            input_size = input_size // 2
        self.linear_layers.append(nn.Linear(input_size, cfg.hidden_size))

        if cfg.activation_fn == 'relu': self.act_fn = nn.ReLU()
        if cfg.activation_fn == 'tanh': self.act_fn = nn.Tanh()

        self.num_layers = cfg.num_layers

    def forward(self, x):
        input = self.flatten(x)
        for i in range(self.num_layers):
            out = self.linear_layers[i](input)
            out = self.act_fn(out)
            input = out

        return out


class Decoder(nn.Module):
    def __init__(self, input_size, cfg):
        super(Decoder, self).__init__()

        self.linear_layers = nn.ModuleList()
        for _ in range(cfg.num_layers - 1):
            self.linear_layers.insert(0, nn.Linear(input_size // 2, input_size))
            input_size = input_size // 2
        self.linear_layers.insert(0, nn.Linear(cfg.hidden_size, input_size))

        if cfg.activation_fn == 'relu': self.act_fn = nn.ReLU()
        if cfg.activation_fn == 'tanh': self.act_fn = nn.Tanh()

        self.num_layers = cfg.num_layers

    def forward(self, x):
        input = x
        for i in range(self.num_layers):
            out = self.linear_layers[i](input)
            out = self.act_fn(out)
            input = out
        return out


class MLPAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 52, 7, 29)
