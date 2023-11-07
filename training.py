import time
from tqdm import tqdm
import numpy as np
from collections import defaultdict

import torch
from torch import nn
from torch.nn.functional import normalize

from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassConfusionMatrix
from sklearn.metrics import balanced_accuracy_score

import wandb

class Trainer:

    def __init__(self, model, loss_fn, optimizer, device, log=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device('cuda') if device=='cuda' and torch.cuda.is_available() else torch.device('cpu')
        self.log = log

        print('Initialized trainer')
        print(f'Using {self.loss_fn}')
        print(f'Using {self.optimizer}')
        print(f'Using {self.device}')

    def train(self, training_set, validation_set=None, epochs=5):
        self.model.to(self.device)

        tr_results = defaultdict(list)
        vl_results = defaultdict(list)
        
        num_classes = training_set.dataset.dataset.num_classes
        tr_mca = MulticlassAccuracy(num_classes=num_classes, average='macro').to(self.device)
        tr_auroc = MulticlassAUROC(num_classes=num_classes, average='macro').to(self.device)
        tr_cm = MulticlassConfusionMatrix(num_classes=num_classes).to(self.device)
        vl_mca = MulticlassAccuracy(num_classes=num_classes, average='macro').to(self.device)
        vl_auroc = MulticlassAUROC(num_classes=num_classes, average='macro').to(self.device)
        vl_cm = MulticlassConfusionMatrix(num_classes=num_classes).to(self.device)
        
        for e in range(epochs):
            print(f'Epoch {e+1}/{epochs}')

            self.model.train()
            tr_results = self.training_step(training_set, [tr_mca])#, tr_auroc])#, tr_cm])
            tr_results['epoch'] = e
            if self.log: wandb.log(tr_results)

            if validation_set:
                print('Validating...')
                self.model.eval()
                vl_results = self.eval_step(validation_set, [vl_mca, vl_auroc])#, vl_cm])
                vl_results['epoch'] = e
                if self.log: wandb.log(vl_results)

        return tr_results, vl_results

    def training_step(self, training_set, tr_metrics):
        losses = [] 
        res = {}

        for mb in tqdm(training_set):
            self.optimizer.zero_grad()

            mb = mb.to(self.device)
            logits = self.model(mb)

            if len(logits.shape) == 3:
                logits = logits.unsqueeze(0) #no one-hot encoding
                mb = mb.long()

            #mb = mb.squeeze().long()
            loss = self.loss_fn(logits, mb)
            loss.backward()
            self.optimizer.step()

            #logging
            if len(mb.shape) == 3: mb_labels = mb # no one-hot encoding
            else: mb_labels = torch.argmax(mb, dim=1) #from one-hot encoding to labels
            for m in tr_metrics:
                m.update(logits, mb_labels)
            losses.append(loss.item())

        for m in tr_metrics:
            name = 'training_' + str(m).split('(')[0].split('Multiclass')[1].lower()
            res[name] = m.compute()
            m.reset()

        res['training_loss'] = sum(losses)

        return res

    def eval_step(self, set, vl_metrics):
        total_loss = 0
        res = {}

        for mb in tqdm(set):
            mb = mb.to(self.device)
            logits = self.model(mb)

            if len(logits.shape) == 3:
                logits = logits.unsqueeze(0) #no one-hot encoding
                mb = mb.long()

            #mb = mb.squeeze().long()
            loss = self.loss_fn(logits, mb)
            total_loss += loss.item()

            #logging
            if len(mb.shape) == 3: mb_labels = mb #no one-hot encoding
            else: mb_labels = torch.argmax(mb, dim=1) #from one-hot encoding to labels
            for m in vl_metrics:
                m.update(logits, mb_labels)

        for m in vl_metrics:
            name = 'validation_' + str(m).split('(')[0].split('Multiclass')[1].lower()
            res[name] = m.compute()
            m.reset() 

        res['validation_loss'] = total_loss

        return res

    def test(self, test_set):
        self.model.eval()

        test_results = defaultdict(list)
        test_mca = MulticlassAccuracy(num_classes=num_classes)

        test_loss, test_acc = self.eval_step(test_set, test_mca)
        test_results['test_loss'].append(test_loss)
        test_results['test_accuracy'].append(test_acc)

        return test_results

    def forward_pass(self, frame):
        self.model.eval()

        if len(frame.shape) == 2:
            frame = frame.view(1, 24, 80).float().to(self.device)
        logits = self.model(frame.to(self.device))
        return logits

    def load_model(self, model_weights):
        self.model.load_state_dict(torch.load(model_weights))
        self.model.to(self.device)

    ''' 
    def test(self, frame):
        self.model.eval()

        print('Original frame')
        print(tty_render(frame[0], frame[1]))

        in_chars = torch.as_tensor(frame[0], device=self.device).view(-1, 1, 24, 80)
        one_hot_in_chars = nn.functional.one_hot(in_chars.long(), num_classes=NUMBER_OF_CHARS).float()
        #one_hot_in_chars = one_hot_in_chars.view(-1, NUMBER_OF_CHARS, 24, 80)
        one_hot_out_chars = self.model(one_hot_in_chars)
        _, out_chars = torch.max(one_hot_out_chars, dim=4)
        out_chars = out_chars.view(24,80)
        out_chars = out_chars.numpy(force=True).astype('uint8')

        print('Reconstructed frame')
        print(tty_render(out_chars, frame[1]))

    def test(self, frame):
        print(tty_render(frame[0], frame[1]))

        frame = torch.tensor(frame).view(-1, 2, 24, 80).to(self.device)
        output_frame = self.model(frame).view(2, 24, 80)
        output_chars = output_frame[0].numpy(force=True).astype('uint8')
        output_colors = output_frame[1].numpy(force=True).astype('uint8')

        print(tty_render(output_chars, output_colors))
    '''