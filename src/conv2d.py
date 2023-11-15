from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(Encoder, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, padding=1),
            nn.ReLU(),
            #nn.Conv2d(64, 128, 3, padding=1),
            #nn.ReLU(),
            #nn.Conv2d(128, 256, 3, padding=1),
            #nn.ReLU()
        )
        self.linear_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*7*29, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )

    def forward(self, input):
        conv_out = self.conv_net(input)
        out = self.linear_net(conv_out)

        return out

class Decoder(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(Decoder, self).__init__()

        self.linear_net = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 128*29*7),
            nn.ReLU()
        )

        self.conv_net = nn.Sequential(
            nn.ConvTranspose2d(128, input_channels, 3, padding=1),
            nn.ReLU(),
            #nn.ConvTranspose2d(128, 64, 3, padding=1),
            #nn.ReLU(),
            #nn.ConvTranspose2d(64, input_channels, 3, padding=1),
            #nn.ReLU()
        )

    def forward(self, x):
        linear_out = self.linear_net(x)
        linear_out = linear_out.view(-1, 256, 7, 29)
        out = self.conv_net(linear_out)

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
