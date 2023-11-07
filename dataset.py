import time
import pickle
from collections import defaultdict

from torch.utils.data import Dataset

class MiniHackDataset(Dataset):
    def __init__(self, path):

        print(f'Loading dataset from {path}')
        with open(path+'.pkl', 'rb') as f: self.frames = pickle.load(f)

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

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

    def create_mapper():
        converter = defaultdict(lambda: len(converter))
        _ = [converter[(ch,co)] for frame in frames for ch,co in zip(frame['chars'].flatten(), frame['colors'].flatten())]

        return dict(converter)
