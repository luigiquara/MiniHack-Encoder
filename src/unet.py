import numpy as np
import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, act_fn):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.act_fn = act_fn

    def forward(self, x):
        out1 = self.act_fn(self.bn1(self.conv1(x)))
        out2 = self.act_fn(self.bn2(self.conv2(out1)))

        return out2

class EncoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels, act_fn):
        super().__init__()
        
        self.conv = ConvBlock(input_channels, output_channels, act_fn)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        pooled = self.max_pool(x)

        return x, pooled

class DecoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels, act_fn):
        super().__init__()

        self.conv_tran = nn.ConvTranspose2d(input_channels, input_channels, kernel_size=3, stride=2, padding=0)
        self.conv = ConvBlock(input_channels+input_channels, output_channels, act_fn)

    def forward(self, x, skip):
        x = self.conv_tran(x)
        x = torch.cat((x, skip), dim=1)
        out = self.conv(x)

        return out

class BottleNeck(nn.Module):
    def __init__(self, input_size, hidden_size, act_fn):
        super().__init__()

        self.flatten = nn.Flatten()
        self.shrink = nn.Linear(input_size, hidden_size)
        self.enlarge = nn.Linear(hidden_size, input_size)
        self.act_fn = act_fn

    def forward(self, x):
        shape = x.shape
        shrinked = self.act_fn(self.shrink(self.flatten(x)))
        enlarged = self.act_fn(self.enlarge(shrinked))
        return shrinked, enlarged.view(shape)



class UNet(nn.Module):
    def __init__(self, input_channels, ll_input_size, cfg):
        super().__init__()

        if cfg.act_fn == 'relu': act_fn = nn.ReLU()
        if cfg.act_fn == 'tanh': act_fn = nn.Tanh()

        #self.encoder = EncoderBlock(input_channels, cfg.output_channels, act_fn)
        #self.bottleneck = BottleNeck(ll_input_size, cfg.hidden_size, act_fn)
        self.decoder = DecoderBlock(cfg.output_channels, input_channels, act_fn)
        self.encoder = EncoderBlock(input_channels, cfg.output_channels, act_fn)

    def forward(self, x):
        skip, encoded = self.encoder(x)
        #hidden_state, enlarged = self.bottleneck(encoded)
        #decoded = self.decoder(enlarged, skip)
        decoded = self.decoder(encoded, skip)

        #return hidden_state, decoded
        return decoded
        
