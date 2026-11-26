# -*- coding: utf-8 -*-
"""
Surrogate model TP: total-structure model trained directly (no transfer learning); used as the
baseline in SI Section 2.

Input : 24 layer thicknesses.  Output: 5 performances.
Network: 24-500-500-500-250-250-5, Leaky ReLU.
Data  : Data/total/ + Data/active_only/. Batch 32, initial lr 1e-3 with step decay (x0.1 every
40 epochs), early stopping (patience 50).

Output: 4QW_total/model/TP.pt, model/info/TP_info.csv

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
import torch.optim.lr_scheduler as lr_scheduler

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

# Backward compatibility: models saved with torch.save(model) store the class as
# "__main__.DNN_TP" (its former name). Registering that name lets torch.load() read them.
import __main__
if not hasattr(__main__, 'DNN_TP'):
    __main__.DNN_TP = TP

if __name__ == '__main__':
    print(f"Using {device} device")
#%% DNN class
class TP(nn.Module):
    def __init__(self):
        super(TP, self).__init__()        
        
        self.fc1 = nn.Linear(24, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 250)
        self.fc5 = nn.Linear(250, 250)
        self.fc6 = nn.Linear(250, 5)
        
                
    def forward(self, x):
        
        x = nn.functional.leaky_relu(self.fc1(x))
        x = nn.functional.leaky_relu(self.fc2(x))
        x = nn.functional.leaky_relu(self.fc3(x))
        x = nn.functional.leaky_relu(self.fc4(x))
        x = nn.functional.leaky_relu(self.fc5(x))
        x = nn.functional.leaky_relu(self.fc6(x))

        return x

    def train_model(self, model, optimizer, scheduler, train_dataloader, val_dataloader, early_stopping, criterion, l2_lamdbda = 1e-5, Early = False, epoch = 2500):
        
        start = time.time()
        
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.criterion = criterion
        self.epoch = epoch
        self.Early = Early
        self.l2_lambda = l2_lamdbda
        
        train_loss = torch.zeros(epoch).to(device)
        val_loss = torch.zeros(epoch).to(device)
        
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
                
                # train loss calculate
                train_loss[ii] += loss
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                        
            #final train loss calculate
            train_loss[ii] = train_loss[ii]/num_batch
            
            # update scheduler
            if scheduler:
                scheduler.step()
            
            # Validation
            with torch.no_grad():
                num_batch = 0
                for X_val, Y_val in val_dataloader:
                    num_batch += 1
                    val_hat = model(X_val)
                    
                    loss = self.criterion(val_hat, Y_val)
                    val_loss[ii] += loss
                    
                val_loss[ii] = val_loss[ii]/num_batch
                
            # Early stopping
            if self.Early:
                early_stopping(val_loss[ii], model)
                if early_stopping.early_stop:
                    print("Early stopping at epoch {}".format(ii + 1))
                    self.epoch_stop = ii+1
                    break
            
            if ii == epoch-1:
                self.epoch_stop = ii+1
                
            # Print loss
            print("Epoch [{}/{}]  Time: {:.1f}s  Loss_train: {:.6f} Loss_val: {:.6f}".format(ii + 1, epoch,
                                                                                       time.time()-epoch_start_time,
                                                                                       train_loss[ii],
                                                                                       val_loss[ii]))
        self.train_loss = train_loss
        self.val_loss = val_loss
        
        print("==========     Gain Trained model saved     ==========\n")
        print("Training time: {:.6f} sec".format(time.time()-start))
        
        
            

if __name__ == '__main__':    
    
    x_file_a = grand_parent_dir + "/Data/active_only/structure_4QW_7_7um_valid.csv"
    y_file_a = grand_parent_dir + "/Data/active_only/values_4QW_7_7um_valid.csv"
    
    injector = [22,43,15,38,16,34,18,30,21,28,25,27,32,27,36,25]
    
    X_active_raw = np.genfromtxt(x_file_a, delimiter = ',')
    
    X_active = np.zeros([len(X_active_raw), 24])
    
    for ii in range(len(X_active)):
        X_active[ii,:] = np.hstack([X_active_raw[ii,:], injector])
    
    Y_active_raw = np.genfromtxt(y_file_a, delimiter = ',')[:len(X_active_raw)]
    
    Y1_active_raw = Y_active_raw[:,0].reshape(len(X_active),1) # gain
    Y2_active_raw = Y_active_raw[:,1].reshape(len(X_active),1) # wave
    Y3_active_raw = Y_active_raw[:,2].reshape(len(X_active),1) # LW
    Y4_active_raw = Y_active_raw[:,3].reshape(len(X_active),1) # Ein0
    Y5_active_raw = Y_active_raw[:,4].reshape(len(X_active),1) # Eout
    
    Y_active = abs(np.hstack([Y1_active_raw,Y3_active_raw,Y2_active_raw,Y4_active_raw,Y5_active_raw]))
    
    x_file = grand_parent_dir + "/Data/total/structure_4QW_tot.csv"
    y_file = grand_parent_dir + "/Data/total/values_4QW_tot.csv"
    
    X = np.genfromtxt(x_file, delimiter = ',')
    Y1 = np.genfromtxt(y_file, delimiter = ',')[:,0].reshape(len(X),1) # gain
    Y2 = np.genfromtxt(y_file, delimiter = ',')[:,1].reshape(len(X),1) # wave
    Y3 = np.genfromtxt(y_file, delimiter = ',')[:,2].reshape(len(X),1) # LW
    Y4 = np.genfromtxt(y_file, delimiter = ',')[:,3].reshape(len(X),1) # Ein0
    Y5 = np.genfromtxt(y_file, delimiter = ',')[:,4].reshape(len(X),1) # Eout
    
    Y = abs(np.hstack([Y1,Y3,Y2,Y4,Y5]))
    
    
    X = np.vstack([X_active,X])
    Y = np.vstack([Y_active,Y])
    
    # tot = np.hstack([X,Y])
    # X = tot[:,:24]
    # Y = tot[:,24:]
    
    # test_size = fraction of the remaining data
    # tot_train, tot_rest = train_test_split(tot,test_size=0.719381742,random_state=11) # 100,000 data
    # tot_train, tot_rest = train_test_split(tot,test_size=0.859690871,random_state=11) # 50,000 data
    # tot_train, tot_rest = train_test_split(tot,test_size=0.82568,random_state=11) # 65,000 data
    
    # X = tot_train[:,:24]
    # Y = tot_train[:,24:]
    
    
    # X_tot = tot_train[:,:24]
    # Y_tot = tot_train[:,24:]
    
    # X = np.vstack([X_tot, X_active]) # 100,000 data
    # Y = np.vstack([Y_tot, Y_active]) # 100,000 data
    
    
    # Max - min normalization
    X_max = X.max(axis = 0)
    X_min = X.min(axis = 0)
    
    Y_max = Y.max(axis = 0)
    Y_min = Y.min(axis = 0)
    
    for ii in range(24):
        X[:,ii] = (X[:,ii] - X_min[ii])/(X_max[ii] - X_min[ii])
    
    for ii in range(5):
        Y[:,ii] = (Y[:,ii] - Y_min[ii])/(Y_max[ii] - Y_min[ii])
    
    
    
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
    
    model = TP().to(device)
    model_name = 'TP'
    # hyper_parameters
    learning_rate = 1e-3
    l2_lambda = 0
    early_stop_patience = 50
    epoch = 200
    
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    criterion = torch.nn.MSELoss(reduction = 'mean')
    early_stopping = EarlyStopping(patience = early_stop_patience, verbose = True, path = f'./model/{model_name}.pt')
    scheduler = lr_scheduler.StepLR(optimizer, 40)
    
    # train model
    model.train_model(model, optimizer, scheduler, train_dataloader, val_dataloader, early_stopping, criterion, l2_lamdbda = l2_lambda, Early = True, epoch = epoch)

    
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
    plt.title("Multi-task model with transfer layer")
    plt.plot(x, train_loss[:epoch], c = 'red', label = 'Train loss')
    plt.plot(x, val_loss[:epoch], c = 'blue', label = 'Validation loss')
    plt.yscale('log',base = 10)
    # plt.ylim([0,3e-2])
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    
    #%% visualize the train result
    model = torch.load(f'{current_dir}/model/TP.pt', weights_only=False).to(device)
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
    
    print(f"R2 of average: {np.mean(R2):.3f}")
    print(f"PCC of average: {np.mean(PCC):.3f}")
    
    print("="*30)
    
