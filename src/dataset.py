import time
import pickle
from collections import defaultdict
import numpy as np

import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

from sklearn.utils.class_weight import compute_class_weight

from nle.nethack import tty_render

class MiniHackDataset(Dataset):
    def __init__(self, path, device='cpu'):

        print(f'Loading dataset from {path}')
        with open(path+'.pkl', 'rb') as f: self.frames = pickle.load(f)
        with open(path+'.training.pkl', 'rb') as f: self.training_set = pickle.load(f)
        with open(path+'.validation.pkl', 'rb') as f: self.validation_set = pickle.load(f)
        with open(path+'.test.pkl', 'rb') as f: self.test_set = pickle.load(f)

        # create unique id for each {char + color} combination
        try:
            with open(path+'.mapper', 'rb') as f:
                self.mapper = pickle.load(f) # {char + color} ---> id
            print('Loaded mapper')
        # create mapper, if it doesn't exist
        except:
            print('Creating new mapper')
            start = time.time()
            self.mapper = self.create_mapper()
            print(f'Mapper created in {time.time()-start}')

            # save mapper to file
            with open(path+'.mapper', 'wb') as f: pickle.dump(self.mapper, f)

        self.inverse_mapper = {v:k for k, v in self.mapper.items()} # id ---> {char + color}

        # same instance variables
        self.num_classes = len(self.mapper)
        self.mapped_training_set = self.map_to_id(self.training_set)
        self.weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.mapped_training_set), y=self.mapped_training_set)
        self.weights = torch.tensor(self.weights).to(device)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

    def create_mapper(self):
        converter = defaultdict(lambda: len(converter))
        _ = [converter[(ch,co)] for frame in self.frames for ch,co in zip(frame['chars'].flatten(), frame['colors'].flatten())]

        return dict(converter)

    def map_to_id(self, data):
        return [self.mapper[(ch,co)] for frame in data for ch,co in zip(frame['chars'].flatten(), frame['colors'].flatten())]

    def collate_fn(self, data):
        def process(frame):
            f = [self.mapper[(ch,co)] for ch, co in zip(frame['chars'].flatten(), frame['colors'].flatten())]
            f = torch.tensor(f)
            f = one_hot(f.long(), num_classes=self.num_classes).float().view(self.num_classes, frame['chars'].shape[0], frame['chars'].shape[1])

            return f

        if not isinstance(data, list): data = [data] 
        batch = torch.stack((list(map(process, data))))
        return batch

    # print both the original frame
    # and the reconstruction from the model
    def render(self, original_frame, logits):
        original_render = tty_render(original_frame['chars'].astype(int), original_frame['colors'].astype(int))
        print('Original frame')
        print(original_render)

        # one-hot encoding ---> id
        logits = logits.view(original_frame['chars'].shape[0], original_frame['chars'].shape[1], -1)
        _, output = torch.max(logits, dim=2)
        output = output.numpy(force=True).astype('uint8').squeeze()

        # id ---> {char + color}
        out_chars, out_colors = zip(*[self.inverse_mapper[e] for e in output.flatten()])
        out_chars = np.array(out_chars).reshape(original_frame['chars'].shape).astype(int)
        out_colors = np.array(out_colors).reshape(original_frame['chars'].shape).astype(int)

        # print reconstruction
        rec_render = tty_render(out_chars, out_colors)
        print('Reconstruction')
        print(rec_render)


