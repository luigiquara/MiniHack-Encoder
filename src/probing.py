from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot

from dataset import MiniHackDataset
from conv2d import ConvAE, Encoder, Decoder
#from mlp import MLPAE, Encoder, Decoder

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, num_classes)
        ) 

    def forward(self, x):
        return self.linear(x)

# see if the probe is able to separate the frames by the env
def separate_envs(path='../minihack_datasets/dataset/dataset', batch_size=4096, autoencoder_path='../models/major-pyramid-8', epochs=10, log=False):
    env_name_to_id = {
        'river': 0,
        'wod': 1,
        'quest': 1
    }

    if log:
        wandb.init(
            project='encoding-minihack',
            config = {
                'group': 'probe-separate_envs',
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
            })

    one_hot_encoding = True
    def collate_fn(data):
        def process(frame):
            f = [data_handler.mapper[(ch,co)] for ch, co in zip(frame['chars'].flatten(), frame['colors'].flatten())]
            f = torch.tensor(f)
            if one_hot_encoding:
                f = one_hot(f.long(), num_classes=data_handler.num_classes).float()
                f = f.view(data_handler.num_classes, frame['chars'].shape[0], frame['chars'].shape[1])
            else:
                f = f.view(frame['chars'].shape[0], frame['chars'].shape[1])

            return f

        if not isinstance(data, list): data = [data] 
        batch = torch.stack((list(map(process, data))))
        env_name = [f['env'] for f in data]
        env_ids = [env_name_to_id[e] for e in env_name]
        env_ids = torch.tensor(env_ids)

        return batch, env_ids

    data_handler = MiniHackDataset(path)
    training_loader = DataLoader(data_handler.training_set, batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    validation_loader = DataLoader(data_handler.validation_set, batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    test_loader = DataLoader(data_handler.test_set, batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)

    # Load the model
    autoencoder = ConvAE(Encoder(data_handler.num_classes, 32), Decoder(data_handler.num_classes, 32))
    #autoencoder = MLPAE(Encoder(data_handler.input_size), Decoder(data_handler.one_hot_input_size))
    autoencoder.load_state_dict(torch.load(autoencoder_path))

    # Define the probe
    probe = LinearProbe(32, 2) # I have three possible envs

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(probe.parameters(), lr=1e-3)
    print('trying sgd')

    global losses
    losses = []
    for e in tqdm(range(epochs)):
        for mb, targets in tqdm(training_loader):
            optimizer.zero_grad()

            embedding = autoencoder.encoder(mb)
            logits = probe(embedding)
            #targets = [env_name_to_id[x] for x in targets]

            loss = loss_fn(logits, targets)
            losses.append(loss.item())

            optimizer.step()


if __name__ == '__main__': separate_envs()