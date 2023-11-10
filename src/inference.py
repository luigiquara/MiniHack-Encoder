import argparse

import torch

from dataset import MiniHackDataset
from conv2d import ConvAE, Encoder, Decoder
from training import Trainer
from run import test_reconstruction

def main(data_path, model_path):
    # Load the dataset
    data_handler = MiniHackDataset(data_path, 'cpu')

    # Load the model
    model = ConvAE(Encoder(data_handler.num_classes, 10), Decoder(data_handler.num_classes, 10))
    model.load_state_dict(torch.load(model_path))

    # Create the trainer
    # Just to perform the forward pass
    trainer = Trainer(model, device='cpu')

    # Test the reconstruction capabilities
    try:
        input('Press Enter to start testing reconstruction\nCtrl+C to terminate')
        while True:
            test_reconstruction(data_handler, data_handler.training_set, trainer)
            try:
                input('Press Enter to get another sample')
            except KeyboardInterrupt:
                break
    except KeyboardInterrupt:
        print('Terminating')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str
    )
    parser.add_argument(
        '--model_path',
        type=str
    )

    flags = parser.parse_args()
    main(**vars(flags))