from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU()
        )

    def forward(self, input):
        return self.encoder(input)

class Decoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, input_size),
            nn.ReLU()
        )

    def forward(self, input):
        return self.decoder(input)

class MLPAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        return self.decoder(self.encoder(input)).view(-1, 52, 7, 29)
