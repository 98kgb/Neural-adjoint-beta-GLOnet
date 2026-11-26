#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis and figures of the surrogate models (CP, TP, GTL, PBTL): dataset distribution,
loss curves, weight distributions, computation time vs. the physical simulator, parity plots,
R2 / PCC accuracy, and distribution of optimized structures.
Some cells read result files that are not part of this repository (Algorithms/Result/*).

@author: Gibaek Kim
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(current_path+'/4QW_active')
sys.path.append(current_path+'/4QW_total')

from train_surrogate_active import CP
from train_surrogate_total_baseline import TP
from train_surrogate_GTL import GTL
from train_surrogate_PBTL import PBTL
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from matplotlib.patches import Polygon
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

if torch.backends.mps.is_available():
    device = "mps"  # Metal Performance Shaders (Mac GPU)
elif torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
else:
    device = "cpu"

def get_parameters(mdl):
    weights = torch.Tensor().to(device)
    for param_group in list(mdl.parameters()):
        weights = torch.cat((param_group.view(-1), weights))
    ws = weights.detach().cpu().numpy()
    return ws

def draw_filled_hull(points, color, alpha=0.3):
    if len(points) >= 3:
        hull = ConvexHull(points)
        polygon = Polygon(points[hull.vertices], closed=True, facecolor=color, alpha=alpha, edgecolor="black", linewidth=1.5)
        plt.gca().add_patch(polygon)

def compute_hull_area(points):
    if len(points) < 3:
        return 0  # we need at least 3 points
    hull = ConvexHull(points)
    return hull.volume

def mean_pairwise_distance(points):
    distances = pdist(points, metric='euclidean')
    return np.mean(distances)

#%% dataset distribution

# Load optimization result
active = np.genfromtxt(parent_path+'/Data/active_only/structure_4QW_7_7um_valid.csv', delimiter = ',')
inject = np.array([22,43,15,38,16,34,18,30,21,28,25,27,32,27,36,25])
inject = np.broadcast_to(inject, (37881, 16))
active = np.hstack([active, inject])
active, _ = train_test_split(active,test_size=0.8,random_state=11)

total = np.genfromtxt(parent_path+'/Data/total/structure_4QW_tot.csv', delimiter = ',')
total, _ = train_test_split(total,test_size=0.8,random_state=11) # 100,000 data


# dimension reduction
reducer = umap.UMAP(n_components=2, random_state=42, n_jobs = 1)
all_data = np.vstack([active, total])
embedding = reducer.fit_transform(all_data)
X_active = embedding[:active.shape[0]]
X_total = embedding[active.shape[0]:active.shape[0] + total.shape[0]]

#%% visulaization

fig, ax = plt.subplots(figsize = (5,5))

sc1 = ax.scatter(X_total[:, 0], X_total[:, 1], s=2, c = 'b', alpha=0.5, label='Total dataset')
sc1 = ax.scatter(X_active[:, 0], X_active[:, 1], s=2, c = 'r', alpha=0.5, label='Active dataset')
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Active dataset',
               markerfacecolor='r', markersize=6, alpha=0.5),
    plt.Line2D([0], [0], marker='o', color='w', label='Total dataset',
               markerfacecolor='b', markersize=6, alpha=0.5)
]
draw_filled_hull(X_active, "red")
draw_filled_hull(X_total, "blue")
ax.legend(handles=legend_elements, fontsize = 12)
plt.xticks([])  
plt.yticks([])
#%% DL comparison
CP_loss = pd.read_csv(f'{current_path}/4QW_active/model/info/CP_info.csv')
TP_loss = pd.read_csv(f'{current_path}/4QW_total/model/info/TP_info.csv')

linewidth = 3
fontsize = 13

plt.figure(figsize=(5,5))
plt.semilogy(CP_loss['Unnamed: 0'], CP_loss['Train loss'], c = 'r', label = 'DNN-CP train', linewidth = linewidth)
plt.semilogy(CP_loss['Unnamed: 0'], CP_loss['Val loss'], linestyle='--', c='r', label = 'DNN-CP val', linewidth = linewidth)
plt.semilogy(TP_loss['Unnamed: 0'], TP_loss['Train loss'], c = 'b', label = 'DNN-TP (DL) train', linewidth = linewidth)
plt.semilogy(TP_loss['Unnamed: 0'], TP_loss['Val loss'], linestyle='--', c='b', label = 'DNN-TP (DL) val', linewidth = linewidth)
plt.grid(True, which='major', linestyle=':', linewidth=1)
plt.grid(True, which='minor', linestyle=':', linewidth=1)
plt.xlabel('Epoch', fontsize = fontsize)
plt.ylabel('MSE', fontsize = fontsize)
plt.legend(fontsize = fontsize)
plt.tick_params(labelsize = fontsize)


#%% loss curve
CP_loss = pd.read_csv(f'{current_path}/4QW_active/model/info/CP_info.csv')
TP_loss = pd.read_csv(f'{current_path}/4QW_total/model/info/TP_info.csv')

PBTL_loss = pd.read_csv(f'{current_path}/4QW_total/model/info/PBTL_info.csv')
GTL_loss = pd.read_csv(f'{current_path}/4QW_total/model/info/GTL_info.csv')

fig, ax = plt.subplots(1,2, figsize = (10,4.5))
colors = ['r', 'g', 'b']
labels = ['DL','GTL', 'PBTL']
info_data = [TP_loss, GTL_loss, PBTL_loss]
fontsize = 18
ticksize = 15
linewidth = 2.5

for idx, info in enumerate(info_data):
    ax[0].semilogy(info['Unnamed: 0'], info['Train loss'], c = colors[idx], label = f'{labels[idx]}', linewidth = linewidth)
    ax[0].semilogy(info['Unnamed: 0'], info['Val loss'], linestyle='--', c=colors[idx], linewidth = linewidth)
    print(min(info['Val loss']))
for idx, info in enumerate(info_data):
    ax[1].semilogy(info['Unnamed: 0'], info['Val loss'], linestyle='--', c = colors[idx], label = f'{labels[idx]}',  linewidth = linewidth)
ax[1].set_ylim([3e-3, 5e-3])


for ii in range(2):
    ax[ii].grid(True, which='major', linestyle=':', linewidth=1)
    ax[ii].grid(True, which='minor', linestyle=':', linewidth=1)
    ax[ii].legend(fontsize = fontsize)
    ax[ii].tick_params(labelsize = fontsize)
    ax[ii].set_xlabel('Epoch', fontsize = fontsize)

ax[0].set_ylabel('MSE', fontsize = fontsize)
ax[1].yaxis.set_visible(False)
plt.tight_layout()


#%% weight distribution
model_CP = torch.load(f'{current_path}/4QW_active/model/CP.pt', map_location=device, weights_only=False)
model_TP = torch.load(f'{current_path}/4QW_total/model/TP.pt', map_location=device, weights_only=False)

TP_PBTL = torch.load(f'{current_path}/4QW_total/model/PBTL.pt', map_location=device, weights_only=False)
TP_GTL = torch.load(f'{current_path}/4QW_total/model/GTL.pt', map_location=device, weights_only=False)
# TP_GTL = torch.load(f'{current_path}/4QW_total/model_usage/GT33_balanced.pt', map_location=device, weights_only=False)


#%%
model_name = ['DNN-CP', 'DNN-TP (DL)', 'DNN-TP (PBTL)', 'DNN-TP (GTL)']
_, ax = plt.subplots(2,2, figsize = (9,9))
fontsize = 17
models = [model_CP, model_TP, TP_PBTL, TP_GTL]
for ii in range(4):
    r, c = divmod(ii, 2)
    model = models[ii]
    w_scheduler = get_parameters(model)
    ax[r][c].hist(w_scheduler.reshape(-1), range=(-.15, .15), bins=1500, color ='r', alpha=0.5)
    ax[r][c].set_xlabel('Weight value', fontsize = fontsize)
    ax[r][c].set_ylabel('Frequency', fontsize = fontsize)
    ax[r][c].set_title(f'{model_name[ii]}', fontsize = fontsize+3)
    ax[r][c].tick_params(axis='both', labelsize=fontsize)
    
plt.tight_layout()
plt.show()

#%% Calculation time
from qcl_simulator import QCL
import time
from tqdm import tqdm

X_min = [30,9,9,40,9,40,9,35,
         15,35,16,35,
         16,20,18,30,21,28,25,27,
         25,25,25,25]

X_max = [50,24,24,60,24,60,24,50,
         25,45,25,45,
         30,30,30,40,30,40,40,40,
         40,40,40,40]

num_sample = 100
structure = np.zeros([num_sample,24])
for ii in range(24):
    structure[:,ii] = np.random.randint(X_min[ii], X_max[ii], size = num_sample)

cal_time = np.zeros([10, 2])
ratio = np.linspace(0.1, 1, 10)
num_data = num_sample*ratio

for tt in tqdm(range(len(ratio)), desc = 'Calculating with surrogate...'):
    X = structure[:int(ratio[tt]*num_sample),:]
    start = time.time()
    DNN = torch.load(f'{current_path}/4QW_total/model/PBTL.pt', map_location=device, weights_only=False)
    result = DNN(torch.tensor(X).float().to(device))
    cal_time[tt,0] = time.time()-start

start = time.time()
aa = 0
for jj in tqdm(range(num_sample), desc = 'Calculating with numerical...'):
    CONV = QCL(structure[jj,:], LW_cal=True)
    CONV.Potential()
    CONV.EVMAT()
    CONV.CBEIGEN()
    CONV.Dipole()
    CONV.Scatter_rate()
    lamda, g43 = CONV.gain()

    if (jj+1) % 10 == 0:
        cal_time[aa,1] = time.time()-start
        aa += 1
#%%
fig, ax = plt.subplots(2,1, figsize = (5,8))
fontsize = 14

ax[0].plot(num_data, cal_time[:,0], label = 'Surrogate model', c = 'b', alpha = 0.5, marker = 'o')
ax[0].plot(num_data, cal_time[:,1], label = 'Numerical model', c = 'r', alpha = 0.5, marker = 'o')
ax[1].semilogy(num_data, cal_time[:,0], label = 'Surrogate model', c = 'b', alpha = 0.5, marker = 'o')
ax[1].semilogy(num_data, cal_time[:,1], label = 'Numerical model', c = 'r', alpha = 0.5, marker = 'o')

for aa in range(2):
    ax[aa].set_xlabel('Number of evaluated QCL structure', fontsize = fontsize)
    ax[aa].set_ylabel('Calculation time (sec)', fontsize = fontsize)
    ax[aa].grid()
    ax[aa].legend()
    ax[aa].tick_params(axis='both', labelsize=fontsize)
plt.tight_layout()
plt.show()

#%% Scatter plot for TP (PBTL)

x_file = parent_path + "/Data/total/structure_4QW_tot.csv"
y_file = parent_path + "/Data/total/values_4QW_tot.csv"

X = np.genfromtxt(x_file, delimiter = ',')
Y1 = np.genfromtxt(y_file, delimiter = ',')[:,0].reshape(len(X),1) # gain
Y2 = np.genfromtxt(y_file, delimiter = ',')[:,1].reshape(len(X),1) # wave
Y3 = np.genfromtxt(y_file, delimiter = ',')[:,2].reshape(len(X),1) # LW
Y4 = np.genfromtxt(y_file, delimiter = ',')[:,3].reshape(len(X),1) # Ein0
Y5 = np.genfromtxt(y_file, delimiter = ',')[:,4].reshape(len(X),1) # Eout

Y = abs(np.hstack([Y1,Y3,Y2,Y4,Y5]))

tot = np.hstack([X,Y])

tot_train, tot_rest = train_test_split(tot,test_size=0.719381742,random_state=11) # 100,000 data

X = tot_train[:,:24]
Y = tot_train[:,24:]

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

model = torch.load(f'{current_path}/4QW_total/model/PBTL.pt', map_location=device, weights_only=False)
y_pred = model(X_test)
y_pred = y_pred.to('cpu').detach().numpy()

y_actual = Y_test.to('cpu').detach().numpy() * (Y_max-Y_min) + Y_min
y_pred = y_pred * (Y_max-Y_min) + Y_min

label = ['$G_{43}$', '$\Gamma$', '$\lambda$', "$E_{inj}$", "$E_{ext}$"]
fontsize = 15
fig, ax = plt.subplots(1,3, figsize = (12,4))
for ii in range(3):
    ax[ii].set_xlabel(f"Actual {label[ii]}", fontsize = fontsize)
    ax[ii].set_ylabel(f"Predict {label[ii]}", fontsize = fontsize)

    ax[ii].set_xlim([min(y_actual[:,ii]),max(y_actual[:,ii])])
    ax[ii].set_ylim([min(y_actual[:,ii]),max(y_actual[:,ii])])
    ax[ii].scatter(y_actual[:,ii], y_pred[:,ii], c = 'g', s = .2, alpha = 0.3)
    ax[ii].minorticks_on()
    ax[ii].grid(which='both', linestyle=':', linewidth=0.5)
    ax[ii].plot([min(y_actual[:,ii]),max(y_actual[:,ii])], [min(y_actual[:,ii]),max(y_actual[:,ii])], c = 'r', alpha = 0.7)
    ax[ii].tick_params(axis='both', labelsize=fontsize)
    
plt.tight_layout()

fig, ax = plt.subplots(1,2, figsize = (8,4))
for ii in range(3,5):
    ax[ii-3].set_xlabel(f"Actual {label[ii]}", fontsize = fontsize)
    ax[ii-3].set_ylabel(f"Predict {label[ii]}", fontsize = fontsize)
    ax[ii-3].set_xlim([min(y_actual[:,ii]),max(y_actual[:,ii])])
    ax[ii-3].set_ylim([min(y_actual[:,ii]),max(y_actual[:,ii])])
    ax[ii-3].scatter(y_actual[:,ii], y_pred[:,ii], c = 'g', s = .2, alpha = 0.3)
    ax[ii-3].plot([min(y_actual[:,ii]),max(y_actual[:,ii])], [min(y_actual[:,ii]),max(y_actual[:,ii])], c = 'r', alpha = 0.7)
    ax[ii-3].minorticks_on()
    ax[ii-3].grid(which='both', linestyle=':', linewidth=0.5)
    ax[ii-3].tick_params(axis='both', labelsize=fontsize)
plt.tight_layout()

#%% Scatter plot for DNN-CP

x_file = parent_path + "/Data/active_only/structure_4QW_7_7um_valid.csv"
y_file = parent_path + "/Data/active_only/values_4QW_7_7um_valid.csv"

X = np.genfromtxt(x_file, delimiter = ',')
Y = np.genfromtxt(y_file, delimiter = ',')

tot = np.hstack([X,Y])

X = tot[:,:8]
Y = tot[:,8:]

# Max - min normalization
X_max = X.max(axis = 0)
X_min = X.min(axis = 0)

Y_max = Y.max(axis = 0)
Y_min = Y.min(axis = 0)

for ii in range(8):
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

model = torch.load(f'{current_path}/4QW_active/model/CP.pt', map_location=device, weights_only=False)
y_pred = model(X_test)
y_pred = y_pred.to('cpu').detach().numpy()

y_actual = Y_test.to('cpu').detach().numpy() * (Y_max-Y_min) + Y_min
y_pred = y_pred * (Y_max-Y_min) + Y_min

label = ['$G_{43}$', '$\Gamma$', '$\lambda$', "$E_{inj}$", "$E_{ext}$"]
fontsize = 15
fig, ax = plt.subplots(1,3, figsize = (12,4))
for ii in range(3):
    ax[ii].set_xlabel(f"Actual {label[ii]}", fontsize = fontsize)
    ax[ii].set_ylabel(f"Predict {label[ii]}", fontsize = fontsize)

    ax[ii].set_xlim([min(y_actual[:,ii]),max(y_actual[:,ii])])
    ax[ii].set_ylim([min(y_actual[:,ii]),max(y_actual[:,ii])])
    ax[ii].scatter(y_actual[:,ii], y_pred[:,ii], c = 'g', s = .2, alpha = 0.3)
    ax[ii].minorticks_on()
    ax[ii].grid(which='both', linestyle=':', linewidth=0.5)
    ax[ii].plot([min(y_actual[:,ii]),max(y_actual[:,ii])], [min(y_actual[:,ii]),max(y_actual[:,ii])], c = 'r', alpha = 0.7)
    ax[ii].tick_params(axis='both', labelsize=fontsize)
    
plt.tight_layout()

fig, ax = plt.subplots(1,2, figsize = (8,4))
for ii in range(3,5):
    ax[ii-3].set_xlabel(f"Actual {label[ii]}", fontsize = fontsize)
    ax[ii-3].set_ylabel(f"Predict {label[ii]}", fontsize = fontsize)
    ax[ii-3].set_xlim([min(np.abs(y_actual[:,ii])),max(y_actual[:,ii])])
    ax[ii-3].set_ylim([min(np.abs(y_actual[:,ii])),max(y_actual[:,ii])])
    ax[ii-3].scatter(np.abs(y_actual[:,ii]), np.abs(y_pred[:,ii]), c = 'g', s = .2, alpha = 0.3)
    ax[ii-3].plot([min(np.abs(y_actual[:,ii])),max(y_actual[:,ii])], [min(np.abs(y_actual[:,ii])),max(y_actual[:,ii])], c = 'r', alpha = 0.7)
    ax[ii-3].minorticks_on()
    ax[ii-3].grid(which='both', linestyle=':', linewidth=0.5)
    ax[ii-3].tick_params(axis='both', labelsize=fontsize)
plt.tight_layout()


#%% Accuracy calculation
DL = torch.load(f'{current_path}/4QW_total/model/TP.pt', map_location=device, weights_only=False)
TP_PBTL = torch.load(f'{current_path}/4QW_total/model/PBTL.pt', map_location=device, weights_only=False)
TP_GTL = torch.load(f'{current_path}/4QW_total/model/GTL.pt', map_location=device, weights_only=False)

PCC = np.zeros([3,5])
R2 = np.zeros([3,5])

label = ['G43', 'Linewidth', 'wavelength', "E inj", "E ext"]
models = [DL, TP_GTL, TP_PBTL]
model_name = ['Direct learning', 'GTL', 'PBTL']
for idx, model in enumerate(models):
    print("="*30, "\n")
    print(f"Model name: {model_name[idx]}\n")
    
    y_pred = model(X_test)
    y_pred = y_pred.to('cpu').detach().numpy()
    y_actual = Y_test.to('cpu').detach().numpy() * (Y_max-Y_min) + Y_min
    y_pred = y_pred * (Y_max-Y_min) + Y_min
    
    for jj in range(5):
        PCC[idx, jj] = np.corrcoef(y_actual[:,jj], y_pred[:,jj])[0,1]
        R2[idx, jj] = r2_score(y_actual[:,jj], y_pred[:,jj])
        
        print(f"PCC of {label[jj]}: {PCC[idx, jj]:.3f}")
        print(f"R2 of {label[jj]}: {R2[idx, jj]:.3f}\n")
        
    print(f"PCC average: {np.mean(PCC[idx,:]):.3f}")
    print(f"R2 average: {np.mean(R2[idx,:]):.3f}\n")
        
    print("="*30)


R2_ratio = (R2-R2[0,:])/R2[0,:]*100
PCC_ratio = (PCC-PCC[0,:])/PCC[0,:]*100

print(f"R2 ratio: {np.round(R2_ratio,2)}\n")
print(f"PCC ratio: {np.round(PCC_ratio,2)}")

R2_ratio_ave = (np.mean(R2, axis = 1) - np.mean(R2, axis = 1)[0])/np.mean(R2, axis = 1)[0]*100
PCC_ratio_ave = (np.mean(PCC, axis = 1) - np.mean(PCC, axis = 1)[0])/np.mean(PCC, axis = 1)[0]*100

print(R2_ratio_ave)
print(PCC_ratio_ave)

#%% Optimization result

result = pd.read_excel(f'{parent_path}/Algorithms/Result/QCL optim hist.xlsx')
fontsize = 15
plt.figure(figsize = (6,5))
plt.plot(result['100 particles'], label = 'population 100', marker = 'o',c = 'r', alpha = 0.5, linewidth = 2)
plt.plot(result['1000 particles'], label = 'population 1000', marker = 'o', c = 'g', alpha = 0.5, linewidth = 2)
plt.plot(result['2000 particles'], label = 'population 2000', marker = 'o', c = 'b', alpha = 0.5, linewidth = 2)
plt.legend(fontsize = fontsize)

plt.xlabel('$Generation$', fontsize = fontsize)
plt.ylabel('$Objective value$', fontsize = fontsize)
plt.grid()
plt.tick_params(axis='both', labelsize=fontsize-2)


#%% Optimized structure distribution

# Define initial population
X_max = [50,24,24,60,24,60,25,50,30,50,20,40,20,40,20,40,23,33,33,33,40,35,45,35]
X_min = [32,11,9,45,9,42,9,35,16,36,11,31,11,31,11,30,14,24,24,24,26,21,31,21]
n_particles = 2000
initial_population = np.zeros([n_particles,24])

for jj in range(24):
    initial_population[:,jj] = np.random.randint(X_min[jj],X_max[jj]+1, size = [n_particles])

# Define optimization population
optim_str = pd.read_csv(f'{parent_path}/Algorithms/Result/DNN_GA.csv', header=None).values[:,:24]

# Define dataset population
total = np.genfromtxt(parent_path+'/Data/total/structure_4QW_tot.csv', delimiter = ',')
total, _ = train_test_split(total,test_size=0.8,random_state=11) # 100,000 data

# dimension reduction
reducer = umap.UMAP(n_components=2, random_state=42, n_jobs = 1)
all_data = np.vstack([initial_population, optim_str, total])
embedding = reducer.fit_transform(all_data)

X_init = embedding[:initial_population.shape[0]]
X_optim = embedding[initial_population.shape[0]:initial_population.shape[0]+optim_str.shape[0]]
X_total = embedding[initial_population.shape[0]+optim_str.shape[0]:]


fontsize = 14
fig, ax = plt.subplots(figsize = (6,5))

sc1 = ax.scatter(X_total[:, 0], X_total[:, 1], s=0.1, c = 'r', alpha=0.5)
sc1 = ax.scatter(X_init[:, 0], X_init[:, 1], s=0.5, c = 'g', alpha=1)
sc1 = ax.scatter(X_optim[:, 0], X_optim[:, 1], s=2, c = 'b', alpha=1)
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Optimized structures',
               markerfacecolor='b', markersize=6, alpha=0.5),
    plt.Line2D([0], [0], marker='o', color='w', label='Initial structures',
               markerfacecolor='g', markersize=6, alpha=0.5),
    plt.Line2D([0], [0], marker='o', color='w', label='Dataset structures',
               markerfacecolor='r', markersize=6, alpha=0.5)
]
draw_filled_hull(X_optim, "blue")
draw_filled_hull(X_init, "green", alpha = 0.1)
draw_filled_hull(X_total, "orange")

ax.legend(handles=legend_elements, fontsize = 12)
plt.xticks([])  
plt.yticks([])
# plt.xlim([4,11])
# plt.ylim([4,7])

#%% Pass at 1024

optim_info = pd.read_csv(f'{parent_path}/Algorithms/Result/pass_rate.csv')
labels = ['Transfer learning', 'Direct learning']
num_100 = optim_info.iloc[0].values[2:]
num_1000 = optim_info.iloc[1].values[2:]
num_2000 = optim_info.iloc[2].values[2:]

x = np.arange(len(labels))
width = 0.2  
fontsize = 18
fig, ax = plt.subplots(figsize=(7, 6))

rects1 = ax.bar(x - width,  num_100,  width, label='population 100',  color='red',   capsize=3, alpha = 0.7)
rects2 = ax.bar(x,          num_1000, width, label='population 1000', color='blue',  capsize=3, alpha = 0.7)
rects3 = ax.bar(x + width,  num_2000, width, label='population 2000', color='green', capsize=3, alpha = 0.7)

ax.set_ylabel('Pass@128', fontsize=fontsize)
ax.set_xlabel('Model', fontsize=fontsize)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=fontsize-2)
ax.legend(fontsize = fontsize)
ax.tick_params(axis='both', labelsize=fontsize-2)

plt.tight_layout()
plt.show()

#%% RMSE distance of train

optim_str = pd.read_csv(f'{parent_path}/Algorithms/Result/DNN_GA.csv', header=None).values[:,:24]
total = np.genfromtxt(parent_path+'/Data/total/structure_4QW_tot.csv', delimiter = ',')
total, _ = train_test_split(total,test_size=0.719381742,random_state=11) # 100,000 data
#%%
L1_distance = np.zeros(optim_str.shape[0])
for ii in range(len(L1_distance)):
    L1_distance[ii] = min(np.abs(total-optim_str[ii,:]).sum(axis = 1))


fontsize = 18
plt.figure(figsize = (8, 7))
plt.hist(L1_distance, bins=25, color ='r', alpha=0.5)
plt.xlabel('$L1$ distance', fontsize = fontsize)
plt.ylabel('Number of optimized structure', fontsize = fontsize)
plt.tick_params(axis='both', labelsize=fontsize)

