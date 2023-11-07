from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(Encoder, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(100,100),
            nn.ReLU()
        )

    def forward(self, input):
        out = self.conv_net(input)
        # out = self.bottleneck(out)

        return out

class Decoder(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(Decoder, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(100,100),
            nn.ReLU()
        )

        self.conv_net = nn.Sequential(
            nn.ConvTranspose2d(64, input_channels, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # out = self.linear(x)
        # out = out.view()
        out = self.conv_net(x)

        return out

class ConvAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(ConvAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded