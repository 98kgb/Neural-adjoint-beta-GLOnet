# -*- coding: utf-8 -*-
"""
Surrogate model CP: active-region model (SI Section 2).

Input : 8 active-region layer thicknesses (injector fixed), min-max normalized.
Output: 5 normalized performances (G43, linewidth, wavelength, E_inj, E_ext).
Network: 8-500-500-500-500-250-5, Leaky ReLU.
Data  : Data/active_only/ (37,881 structures). Batch 32, initial lr 1e-3, early stopping (patience 50).
Its first layer is transferred to the PBTL model.

Output: 4QW_active/model/CP.pt, model/info/CP_info.csv

@author: Gibaek Kim
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grand_parent_dir = os.path.dirname(parent_dir)

sys.path.append(current_dir)
sys.path.append(parent_dir)
sys.path.append(grand_parent_dir)

import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset
from Deep_Learning_class import EarlyStopping

if torch.backends.mps.is_available():
    device = "mps"  # Metal Performance Shaders (Mac GPU)
elif torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
else:
    device = "cpu"


# DNN class
class CP(nn.Module):
    def __init__(self):
        super(CP, self).__init__()        
        
        self.com1 = nn.Linear(8, 500)
        self.com2 = nn.Linear(500, 500)
        self.com3 = nn.Linear(500, 500)
        self.com4 = nn.Linear(500, 500)
        self.com5 = nn.Linear(500, 250)
        self.com6 = nn.Linear(250, 5)
                
    def forward(self, x):
        
        x = nn.functional.leaky_relu(self.com1(x))
        x = nn.functional.leaky_relu(self.com2(x))
        x = nn.functional.leaky_relu(self.com3(x))
        x = nn.functional.leaky_relu(self.com4(x))
        x = nn.functional.leaky_relu(self.com5(x))
        x = nn.functional.leaky_relu(self.com6(x))
        
        return x

    def train_model(self, model, optimizer, train_dataloader, val_dataloader, early_stopping,  criterion, l2_lambda=5e-5, Early = False, epoch = 2500):
        
        start = time.time()
        
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.criterion = criterion
        self.epoch = epoch
        self.Early = Early
        
        self.train_loss = torch.zeros(epoch).to(device)
        self.val_loss = torch.zeros(epoch).to(device)
        
        print("========       Iterations started       ========")
        for ii in range(self.epoch):
            
            epoch_start_time = time.time()
            
            num_batch = 0
            
            for X_train, Y_train in train_dataloader:
                
                # ititialize weight
                optimizer.zero_grad()
                
                num_batch += 1
                Y_hat = model(X_train)
    
                loss = self.criterion(Y_hat, Y_train)
                self.train_loss[ii] += loss
                
                # Backward and optimize
                loss.backward()
                
                # print(loss.grad)
                optimizer.step()
                
                        
            #final train loss calculate
            self.train_loss[ii] = self.train_loss[ii]/num_batch
            
            # Validation
            with torch.no_grad():
                num_batch = 0
                for X_val, Y_val in val_dataloader:
                    num_batch += 1
                    hat_val = model(X_val)
                    
                    loss = self.criterion(hat_val, Y_val)
                    self.val_loss[ii] += loss
                    
                self.val_loss[ii] = self.val_loss[ii]/num_batch
                
            # Early stopping
            if self.Early:
                early_stopping(self.val_loss[ii], model)
                if early_stopping.early_stop:
                    print("Early stopping at epoch {}".format(ii + 1))
                    self.epoch_stop = epoch
                    break
            if ii == epoch-1:
                self.epoch_stop = ii+1
                
            # Print loss
            print("Epoch [{}/{}]  Time: {:.1f}s  Loss_train: {:.6f} Loss_val: {:.6f}".format(ii + 1, epoch,
                                                                                       time.time()-epoch_start_time,
                                                                                       self.train_loss[ii],
                                                                                       self.val_loss[ii]))
            
        # self.train_loss = self.train_loss.to('cpu').detach().numpy()
        # self.val_loss = self.val_loss.to('cpu').detach().numpy()
        
        print("==========     Gain Trained model saved     ==========\n")
        print("Training time: {:.6f} sec".format(time.time()-start))

#%% Data prepare

# Backward compatibility: models saved with torch.save(model) store the class as
# "__main__.DNN_CP" (its former name). Registering that name lets torch.load() read them.
import __main__
if not hasattr(__main__, 'DNN_CP'):
    __main__.DNN_CP = CP

if __name__ == '__main__':
    x_file = grand_parent_dir + "/Data/active_only/structure_4QW_7_7um_valid.csv"
    y_file = grand_parent_dir + "/Data/active_only/values_4QW_7_7um_valid.csv"
    
    X = np.genfromtxt(x_file, delimiter = ',')
    Y = np.genfromtxt(y_file, delimiter = ',')
    
    # min, max for input values
    X_max = X.max(axis = 0)
    X_min = X.min(axis = 0)
    
    # min, max for output values
    Y_max = Y.max(axis = 0)
    Y_min = Y.min(axis = 0)
    
    # Max - min normalization
    X = (X - X_min)/(X_max - X_min)
    Y = (Y - Y_min)/(Y_max - Y_min)
        
    # Test/Train split
    X_train, X_test = train_test_split(X,test_size=0.2,random_state=2)
    X_test, X_val = train_test_split(X_test, test_size=0.5, random_state=2)
    
    Y_train, Y_test = train_test_split(Y,test_size=0.2,random_state=2)
    Y_test, Y_val = train_test_split(Y_test, test_size=0.5, random_state=2)
    
    # Conver to tensor and dataset
    X_train, X_test, X_val = torch.from_numpy(X_train).float().to(device), torch.from_numpy(X_test).float().to(device), torch.from_numpy(X_val).float().to(device)
    Y_train, Y_test, Y_val = torch.from_numpy(Y_train).float().to(device), torch.from_numpy(Y_test).float().to(device), torch.from_numpy(Y_val).float().to(device)
    
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)
    val_dataset = TensorDataset(X_val, Y_val)
    
    # Define Batch size and DataLoader
    batch_size = 32
    
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size)
    #%% model define
    
    model = CP().to(device)
    model_name = 'CP'
    optimizer = optim.AdamW(model.parameters(), lr = 1e-3)
    criterion = torch.nn.MSELoss(reduction = 'mean')
    early_stopping = EarlyStopping(patience=50, verbose = True, path = f'{current_dir}/model/{model_name}.pt')
    epoch = 200
    
    # train model
    model.train_model(model, optimizer, train_dataloader, val_dataloader, early_stopping,  criterion,l2_lambda= 0, Early = True, epoch = epoch)
    
    
    
    #%% fitting
    train_loss = model.train_loss.to('cpu').detach().numpy()
    val_loss = model.val_loss.to('cpu').detach().numpy()
    
    epoch =model.epoch_stop
    train_info = np.zeros([epoch, 2])
    train_info[:,0], train_info[:,1]= train_loss[:epoch], val_loss[:epoch]
    train_info= pd.DataFrame(train_info, columns=['Train loss', 'Val loss'])
    train_info.to_csv(current_dir + f"/model/info/{model_name}_info.csv")
    
    x = np.linspace(1,epoch,epoch)
    
    plt.figure(0)
    plt.title("Multi-task model updated")
    plt.plot(x, train_loss[:epoch], c = 'red', label = 'Train loss')
    plt.plot(x, val_loss[:epoch], c = 'blue', label = 'Validation loss')
    plt.yscale('log',base = 10)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    
    #%% visualize the train result
    # evaluate model
    # model = torch.load(current_dir+ f'/model/{model_name}.pt', map_location = device, weights_only= False)
    
    y_pred = model(X_test)
    y_pred = y_pred.to('cpu').detach().numpy()

    y_actual = Y_test.to('cpu').detach().numpy() * (Y_max-Y_min) + Y_min
    y_pred = y_pred * (Y_max-Y_min) + Y_min
    
    fig, ax = plt.subplots(1,5, figsize = (25,5))
    
    label = ['$G_{43}$', '$\Gamma$', '$\lambda$', "$E_{inj}$", "$E_{ext}$"]
    
    for ii in range(5):
        ax[ii].set_xlabel(f"Actual {label[ii]}")
        ax[ii].set_ylabel(f"Predict {label[ii]}")
    
        ax[ii].set_xlim([min(y_actual[:,ii]),max(y_actual[:,ii])])
        ax[ii].set_ylim([min(y_actual[:,ii]),max(y_actual[:,ii])])
    
        ax[ii].scatter(y_actual[:,ii], y_pred[:,ii], c = 'g', s = 0.5)
    
    # Accuracy calculation
    PCC = np.zeros(5)
    R2 = np.zeros(5)

    print("="*30, "\n")
    label = ['G43', 'Linewidth', 'wavelength', "E inj", "E ext"]
    for ii in range(5):
        PCC[ii] = np.corrcoef(y_actual[:,ii], y_pred[:,ii])[0,1]
        R2[ii] = r2_score(y_actual[:,ii], y_pred[:,ii])
        
        print(f"R2 of {label[ii]}: {R2[ii]:.3f}")
        print(f"PCC of {label[ii]}: {PCC[ii]:.3f}\n")
        
    print("="*30)
