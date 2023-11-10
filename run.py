import argparse
from random import randrange

from torch import optim
from torch import nn
from torch.utils.data import random_split, DataLoader

from dataset import MiniHackDataset
from training import Trainer
from models.conv2d import ConvAE, Encoder, Decoder

import wandb

def main(path, batch_size, use_loss_weights, model_weights, lr, epochs, device, log):
    if log:
        wandb.init(
            project='encoding-minihack',
            config = {
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'class_weights': use_loss_weights,
                'architecture': 'conv2d'
            })

    # Load the dataset
    data_handler = MiniHackDataset(path, device)

    training_loader = DataLoader(data_handler.training_set, batch_size, shuffle=True, drop_last=True, collate_fn=data_handler.collate_fn)
    validation_loader = DataLoader(data_handler.validation_set, batch_size, shuffle=True, drop_last=True, collate_fn=data_handler.collate_fn)
    test_loader = DataLoader(data_handler.test_set, batch_size, shuffle=True, drop_last=True, collate_fn=data_handler.collate_fn)

    if log: wandb.config.training_set_size = len(training_set)
    print(f'The entire dataset has {len(data_handler)} frames')
    print(f'Training set: {len(data_handler.training_set)} frames')
    print(f'Validation set: {len(data_handler.validation_set)} frames')
    print(f'Test set: {len(data_handler.test_set)} frames\n')


    # Define the model, with optimizer and loss function
    model = ConvAE(Encoder(data_handler.num_classes, 10), Decoder(data_handler.num_classes, 10))
    if use_loss_weights: loss_fn = nn.CrossEntropyLoss(weight=data_handler.weights)
    else: loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    
    # Training process
    trainer = Trainer(model, loss_fn, optimizer, device, log)
    results = trainer.train(training_loader, validation_loader, data_handler.num_classes,epochs)

    print(results)

    # Test reconstruction
    try:
        input('Press Enter to start testing reconstruction\nCtrl+C to terminate')
        while True:
            test_reconstruction(data_handler, data_handler.training_set, trainer)
            try:
                input('Press Enter to get another sample')
            except KeyboardInterrupt:
                if log: wandb.finish()
                break
    except KeyboardInterrupt:
        print('Terminating')
        if log: wandb.finish()

def test_reconstruction(data_handler, dataset, trainer):
    idx = randrange(len(dataset))
    sample = dataset[idx]
    data_handler.render(sample, trainer.forward_pass(data_handler.collate_fn(sample)))


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
        '--use_loss_weights',
        action='store_true',
        help='Use class weights in the loss function'
    )
    trainer_args.add_argument(
        '--no-use_loss_weights',
        action='store_false',
        dest='use_loss_weights'
    )
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
