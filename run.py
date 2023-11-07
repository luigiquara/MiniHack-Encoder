import argparse

from torch import optim
from torch import nn
from torch.utils.data import random_split, DataLoader

from dataset import MiniHackDataset
from training import Trainer

import wandb

def main(path, batch_size, model_weights, lr, epochs, device, log):
    if log:
        wandb.init(
            project='encoding-minihack',
            config = {
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'architecture': 'conv2d'
            })

    # Load the dataset
    data_handler = MiniHackDataset(path)
    training_set, validation_set, test_set = random_split(data_handler, [0.7, 0.15, 0.15])

    training_loader = DataLoader(training_set, batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_set, batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True, drop_last=True)

    if log: wandb.config.training_set_size = len(training_set)
    print(f'The entire dataset has {len(data_handler)} frames')
    print(f'Training set: {len(training_set)} frames')
    print(f'Validation set: {len(validation_set)} frames')
    print(f'Test set: {len(test_set)} frames\n')


    # Define the model, with optimizer and loss function
    model = ConvAE()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    
    # Training process
    trainer = Trainer(model, loss_fn, optimizer, device, log)
    results = trainer.train(training_loader, validation_loader, epochs)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    reader_args = parser.add_argument_group('read the dataset')
    reader_args.add_argument(
        '--path',
        type = str,
        default = 'datasets/dataset',
        help = 'The filepath to the dataset'
    )
    reader_args.add_argument(
        '--batch_size',
        type = int,
        default = 32,
        help = 'The size of each batch'
    )

    trainer_args = parser.add_argument_group('training process')
    trainer_args.add_argument(
        '--model_weights',
        type = str,
        help='Path where to save the weights of the best model'
    )
    trainer_args.add_argument(
        '--lr',
        type = float,
        default = 1e-3,
        help = 'The learning rate'
    )
    trainer_args.add_argument(
        '--epochs',
        type = int,
        default = 5,
        help = 'The number of epochs'
    )
    trainer_args.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='The device to use to run the experiments'
    )
    trainer_args.add_argument(
        '--log',
        action='store_true',
        help='Log the results using wandb'
    )
    trainer_args.add_argument(
        '--no-log',
        dest='log',
        action='store_false'
    )

    flags = parser.parse_args()
    main(**vars(flags))
