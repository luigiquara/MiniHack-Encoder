from argparse import ArgumentParser
import numpy as np

from dataset import MiniHackDataset
from conv2d import ConvAE, Encoder, Decoder 

import torch
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
import seaborn as sns

def main(data_path, batch_size, model_path):
    # Load the validation set
    data_handler = MiniHackDataset(data_path)
    validation_loader = DataLoader(data_handler.validation_set, 1, shuffle=True, drop_last=True, collate_fn=data_handler.collate_fn)

    # Load the model
    model = ConvAE(Encoder(data_handler.num_classes, 32), Decoder(data_handler.num_classes, 32))
    model.load_state_dict(torch.load(model_path))

    # Forward pass through the encoder
    encoded = []
    model.eval()
    with torch.no_grad():
        for mb in validation_loader:
            out = model.encoder(mb)
            encoded.extend(out.numpy())

    encoded = np.array(encoded)

    x = TSNE().fit_transform(encoded)
    sns.scatterplot(encoded)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='The path to the dataset'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help="The size of the batch (don't you say?)"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='The path to the model to load'
    )

    flags = parser.parse_args()
    main(**vars(flags))
