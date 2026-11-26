# -*- coding: utf-8 -*-
"""
EarlyStopping helper used when training the surrogate models: stops training when the
validation loss has not improved for `patience` epochs and saves the best model.
(The file name must stay as is: the released .pt files reference this module.)

@author: Gibaek Kim
"""
import torch
import numpy as np

if torch.backends.mps.is_available():
    device = "mps"  # Metal Performance Shaders (Mac GPU)
elif torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
else:
    device = "cpu"

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, path='./model/checkpoint.pt'):
        self.patience = patience
        
        self.verbose = verbose
        self.path = path
        self.counter = 0
        
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):

        if self.best_score is None:
            self.best_score = val_loss
        
        elif val_loss > self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        torch.save(model, self.path)
        self.val_loss_min = val_loss