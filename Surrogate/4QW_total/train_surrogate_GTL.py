# -*- coding: utf-8 -*-
"""
Surrogate model GTL: total-structure model with general transfer learning (comparison model in SI).
Hidden layers of the CP model are transferred and the network is fine-tuned on the
24-layer data. Batch 32, initial lr 1e-3, early stopping (patience 50).

Output: 4QW_total/model/GTL_from_*.pt and info csv

@author: Gibaek Kim
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grand_parent_dir = os.path.dirname(parent_dir)

sys.path.append(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, '4QW_active'))
sys.path.append(grand_parent_dir)

import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from train_surrogate_active import CP
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
# "__main__.DNN_TP_GTL" (its former name). Registering that name lets torch.load() read them.
import __main__
if not hasattr(__main__, 'DNN_TP_GTL'):
    __main__.DNN_TP_GTL = GTL

if __name__ == '__main__':
    print(f'Using {device} device')

#%% DNN class
class GTL(nn.Module):
    def __init__(self, tr_layers = 0):
        super(GTL, self).__init__()        
        
        trained_model = torch.load(parent_dir + '/4QW_active/model/CP.pt', map_location = device, weights_only= False)
        
        self.fc1 = nn.Linear(24, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, 250)
        self.fc6 = nn.Linear(250, 125)
        self.fc7 = nn.Linear(125, 5)

        self.tr_layers = tr_layers
        self.tran2 = list(trained_model.children())[1]
        self.tran3 = list(trained_model.children())[2]
        self.tran4 = list(trained_model.children())[3]
        self.tran5 = list(trained_model.children())[4]
        self.tran6 = list(trained_model.children())[5]
        

    def forward(self, x):
        
        if 1 in self.tr_layers:
            raise ValueError("Invalid value entered. Please enter a number between 2, 6.")
        else:   
            x = nn.functional.leaky_relu(self.fc1(x))
        
        if 2 in self.tr_layers:
            x = nn.functional.leaky_relu(self.tran2(x))
        else:
            x = nn.functional.leaky_relu(self.fc2(x))

        if 3 in self.tr_layers:
            x = nn.functional.leaky_relu(self.tran3(x))
        else:
            x = nn.functional.leaky_relu(self.fc3(x))
        
        if 4 in self.tr_layers:
            x = nn.functional.leaky_relu(self.tran4(x))

        else:
            x = nn.functional.leaky_relu(self.fc4(x))

        if 5 in self.tr_layers:
            x = nn.functional.leaky_relu(self.tran5(x))

        else:
            x = nn.functional.leaky_relu(self.fc5(x))

        if 6 in self.tr_layers:
            x = nn.functional.leaky_relu(self.tran6(x))

        else:
            x = nn.functional.leaky_relu(self.fc6(x))
        
        x = nn.functional.leaky_relu(self.fc7(x))
        
        return x

    def train_model(self, model, optimizer, train_dataloader, val_dataloader, early_stopping, criterion, 
                    l2_lamdbda = 1e-5, Early = False, epoch = 2500):
        
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
                # print(X_train.size())
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
            

        print("==========     Gain Trained model saved     ==========\n")
        print("Training time: {:.6f} sec".format(time.time()-start))
        
        self.train_loss = train_loss
        self.val_loss = val_loss
        
if __name__ == '__main__':
    # Data prepare active
    
    # x_file_a = grand_parent_dir + "/Data/active_only/structure_4QW_7_7um_valid.csv"
    # y_file_a = grand_parent_dir + "/Data/active_only/values_4QW_7_7um_valid.csv"
    
    # injector = [22,43,15,38,16,34,18,30,21,28,25,27,32,27,36,25]
    
    # X_active_raw = np.genfromtxt(x_file_a, delimiter = ',')
    
    # X_active = np.zeros([len(X_active_raw), 24])
    
    # for ii in range(len(X_active)):
    #     X_active[ii,:] = np.hstack([X_active_raw[ii,:], injector])
    
    # Y_active_raw = np.genfromtxt(y_file_a, delimiter = ',')[:len(X_active_raw)]
    
    # Y1_active_raw = Y_active_raw[:,0].reshape(len(X_active),1) # gain
    # Y2_active_raw = Y_active_raw[:,1].reshape(len(X_active),1) # wave
    # Y3_active_raw = Y_active_raw[:,2].reshape(len(X_active),1) # LW
    # Y4_active_raw = Y_active_raw[:,3].reshape(len(X_active),1) # Ein0
    # Y5_active_raw = Y_active_raw[:,4].reshape(len(X_active),1) # Eout
    
    # Y_active = abs(np.hstack([Y1_active_raw,Y3_active_raw,Y2_active_raw,Y4_active_raw,Y5_active_raw]))
    
    # Data prepare total
    
    x_file = grand_parent_dir + "/Data/total/structure_4QW_tot.csv"
    y_file = grand_parent_dir + "/Data/total/values_4QW_tot.csv"
    
    
    X = np.genfromtxt(x_file, delimiter = ',')
    Y1 = np.genfromtxt(y_file, delimiter = ',')[:,0].reshape(len(X),1) # gain
    Y2 = np.genfromtxt(y_file, delimiter = ',')[:,1].reshape(len(X),1) # wave
    Y3 = np.genfromtxt(y_file, delimiter = ',')[:,2].reshape(len(X),1) # LW
    Y4 = np.genfromtxt(y_file, delimiter = ',')[:,3].reshape(len(X),1) # Ein0
    Y5 = np.genfromtxt(y_file, delimiter = ',')[:,4].reshape(len(X),1) # Eout
    
    
    Y = abs(np.hstack([Y1,Y3,Y2,Y4,Y5]))
    
    tot = np.hstack([X,Y])
    
    # test_size = fraction of the remaining data
    tot_train, tot_rest = train_test_split(tot,test_size=0.719381742,random_state=11) # 100,000 data
    # tot_train, tot_rest = train_test_split(tot,test_size=0.859690871,random_state=11) # 50,000 data
    # tot_train, tot_rest = train_test_split(tot,test_size=0.82568,random_state=11) # 65,000 data
    
    
    
    X = tot_train[:,:24]
    Y = tot_train[:,24:]
    
    
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
    # tr_layers = [[2,3,4],
    #              [2,3,4,5]
    #     ]
    tr_layers = [
                 [4]
        ]
    
    for tr_layer in tr_layers:
        model = GTL(tr_layers=tr_layer).to(device)
        model_name = f'GTL_from_{tr_layer[0]}_to_{tr_layer[-1]}'
        print(model_name)
        # hyperparameters
        
        learning_rate = 1e-3
        l2_lambda = 0
        early_stop_patience = 50
        epoch = 200
        
        freeze_list = []
        for ii in range(1, 7):
            freeze_list.append(f"tran{ii}.weight")
        
        for name, param in model.named_parameters():
            if name in freeze_list:
                param.requires_grad = False
                print(name, "is fixed")

        optimizer = optim.AdamW(model.parameters(), lr = learning_rate)
        
        criterion = torch.nn.MSELoss(reduction = 'mean')
        
        early_stopping = EarlyStopping(patience = early_stop_patience, verbose = True,
                                       path = os.path.join(current_dir, 'model', f'{model_name}.pt'))
        
        
        # train model
        model.train_model(model, optimizer, train_dataloader, val_dataloader, early_stopping, criterion,
                          l2_lamdbda = l2_lambda, Early = True, epoch = epoch)
    
        # fitting
        train_loss = model.train_loss.to('cpu').detach().numpy()
        val_loss = model.val_loss.to('cpu').detach().numpy()
        epoch =model.epoch_stop
        train_info = np.zeros([epoch, 4])
        train_info[:,0], train_info[:,1]= train_loss[:epoch], val_loss[:epoch]
        
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
        plt.show()
        # visualize the train result
        
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
        plt.show()
        
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
        
        train_info[:5, 2] = R2
        train_info[:5, 3] = PCC
        
        train_info = pd.DataFrame(train_info, columns=['Train loss', 'Val loss', 'R2', 'PCC'])
        train_info.to_csv(os.path.join(current_dir, "model", "info", f"{model_name}_info.csv"))
        
