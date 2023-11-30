import os
import time
import hydra
from omegaconf import OmegaConf, DictConfig
from random import randrange

from torch import optim
from torch import nn
from torch.utils.data import DataLoader

from dataset import MiniHackDataset
from training import Trainer
from mlp import MLPAE, Encoder, Decoder

import wandb
from torchinfo import summary

from sklearn.model_selection import train_test_split as tts

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    if cfg.log:
        wandb.init(
            project='encoding-minihack',
            config = {
                'epochs': cfg.train.epochs,
                'batch_size': cfg.train.batch_size,
                'lr': cfg.train.lr,
                'class_weights': cfg.train.use_loss_weights,
                'one_hot_encoding': cfg.train.one_hot,
                'model_type': cfg.architecture.name,
                'encoder_num_layers': cfg.architecture.encoder_num_layers,
                'decoder_num_layers': cfg.architecture.decoder_num_layers,
                'activation_fn': cfg.architecture.act_fn,
                'optimizer': cfg.optimizer.name,
                'weight_decay_lambda': cfg.optimizer.weight_decay,
                'momentum': cfg.optimizer.momentum,
                'nesterov': cfg.optimizer.nesterov,
                'label_smoothing': cfg.train.label_smoothing
            })

    if cfg.log:
        wandb.config.cfg = OmegaConf.to_yaml(cfg)

    # Load the dataset
    data_handler = MiniHackDataset(cfg.path.data_path, cfg.train.one_hot, cfg.train.device, sample=True)

    training_loader = DataLoader(data_handler.training_set, cfg.train.batch_size, shuffle=True, drop_last=True, collate_fn=data_handler.collate_fn)
    validation_loader = DataLoader(data_handler.validation_set, cfg.train.batch_size, shuffle=True, drop_last=True, collate_fn=data_handler.collate_fn)
    test_loader = DataLoader(data_handler.test_set, cfg.train.batch_size, shuffle=True, drop_last=True, collate_fn=data_handler.collate_fn)

    if cfg.log: wandb.config.training_set_size = len(data_handler.training_set)
    print(f'The entire dataset has {len(data_handler)} frames')
    print(f'Training set: {len(data_handler.training_set)} frames')
    print(f'Validation set: {len(data_handler.validation_set)} frames')
    print(f'Test set: {len(data_handler.test_set)} frames\n')


    if cfg.architecture.name == 'mlp':
        from mlp import MLPAE, Encoder, Decoder
        
        model = MLPAE(Encoder(data_handler.one_hot_input_size, cfg.architecture), Decoder(data_handler.one_hot_input_size, cfg.architecture), data_handler.num_classes)
    elif cfg.architecture.name == 'conv2d':
        pass
    elif cfg.architecture.name == 'unet':
        from unet import UNet

        model = UNet(data_handler.num_classes, ll_input_size=3*14*cfg.architecture.output_channels, cfg=cfg.architecture)
    summary(model, input_data=next(iter(training_loader)), col_names=['input_size', 'output_size'])
    

    '''
    # Define the model, with optimizer and loss function
    if cfg.architecture == 'mlp':
        if cfg.train.one_hot:
            print('Using one-hot encoding for the input')
            model = MLPAE(Encoder(data_handler.one_hot_input_size, cfg.architecture), Decoder(data_handler.one_hot_input_size, cfg.architecture))
        else:
            print('Taking raw input')
            model = MLPAE(Encoder(data_handler.input_size, cfg.architecture), Decoder(data_handler.one_hot_input_size, cfg.architecture))
        #summary(model, (cfg.train.batch_size, 52*7*29))
    '''

    if cfg.train.use_loss_weights: loss_fn = nn.CrossEntropyLoss(weight=data_handler.weights, label_smoothing=cfg.train.label_smoothing)
    else: loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)


    if cfg.optimizer.name == 'adam':
        if cfg.log: wandb.config.amsgrad = cfg.optimizer.amsgrad
        optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.optimizer.weight_decay, amsgrad=cfg.optimizer.amsgrad)
    if cfg.optimizer.name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.train.lr,
                              momentum=cfg.optimizer.momentum, nesterov=cfg.optimizer.nesterov, weight_decay=cfg.optimizer.weight_decay)
    if cfg.log: wandb.config.model = summary(model, (cfg.train.batch_size, 51, 7, 29))

    
    # Training process
    if cfg.log: save_path = os.getcwd() + '/' + wandb.run.name
    else: save_path = os.getcwd() + '/model'
    trainer = Trainer(model, loss_fn, optimizer, save_path, cfg.train.device, cfg.log)
    results = trainer.train(training_loader, validation_loader, data_handler.num_classes, cfg.train.epochs)
    print(results)


    # Test reconstruction
    try:
        input('Press Enter to start testing reconstruction\nCtrl+C to terminate')
        while True:
            test_reconstruction(data_handler, data_handler.training_set, trainer)
            try:
                input('Press Enter to get another sample')
            except KeyboardInterrupt:
                if cfg.log: wandb.finish()
                print()
                break
    except KeyboardInterrupt:
        print('Terminating')
        if cfg.log: wandb.finish()

def test_reconstruction(data_handler, dataset, trainer):
    idx = randrange(len(dataset))
    sample = dataset[idx]
    data_handler.render(sample, trainer.forward_pass(data_handler.collate_fn(sample)))


if __name__ == '__main__': main()