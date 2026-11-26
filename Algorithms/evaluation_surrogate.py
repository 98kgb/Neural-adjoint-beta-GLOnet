# -*- coding: utf-8 -*-
"""
Surrogate figure of merit (FoM) computed with the trained PBTL surrogate model
(Surrogate/4QW_total/model/surrogate_PBTL_*.pt). Same FoM definition as Eqs. (1)-(6)
(sigmoid slope 0.5, wavelength sigma = 0.1 um).

EVAL_7_7um_total : batch FoM of 24-layer structures; used as the fitness of GA and the reward of REINFORCE.

@author: Gibaek Kim
"""
import sys
import os
path_c = os.path.dirname(os.path.abspath(__file__))   # <repo>/Algorithms
path_p = os.path.dirname(path_c)                       # <repo>

sys.path.append(path_c)
sys.path.append(os.path.join(path_p, 'Surrogate', '4QW_total'))

from train_surrogate_PBTL import PBTL

import numpy as np
import math
import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Device: ", device)

def Gaussian(x, mean, sig):
    
    y = (1/(sig*math.sqrt(2*np.pi)))*np.exp(-(1/2)*(x-mean)**2/sig**2)
    
    return y

def Dirac(x, c1, c2):
    
    y = 1/(1+np.exp(c1*(x-c2)))
    
    return y

#%% for 4QW 7.7um total
def EVAL_7_7um_total(thick, X_max, X_min, Y_max, Y_min, xfile, t_gain=24, t_LW=24, t_wave = 7.7, t_Ein=26, t_Eout=26):
    
    model = torch.load(xfile, map_location = device, weights_only= False)
    
    thick_eval = np.zeros([len(thick),24])
    
    for ii in range(24):
        thick_eval[:,ii] = (thick[:,ii] - X_min[ii])/(X_max[ii] - X_min[ii])

    structure=torch.FloatTensor(thick_eval).to(device)
    
    result = model(structure)
    
    result = result.detach().cpu().numpy()
    
    peak = result[:,0]*(Y_max[0]-Y_min[0]) + Y_min[0]
    
    LW = result[:,1]*(Y_max[1]-Y_min[1]) + Y_min[1]
    
    wave = result[:,2]*(Y_max[2]-Y_min[2]) + Y_min[2]
    Ein = result[:,3]*(Y_max[3]-Y_min[3]) + Y_min[3]
    Eout = result[:,4]*(Y_max[4]-Y_min[4]) + Y_min[4]
    
    peak_point =  Dirac(peak, -0.5, t_gain)
    LW_point = Dirac(LW, 0.5, t_LW)
    wave_point = Gaussian(wave, t_wave, 0.1)/4
    Ein_point = Dirac(Ein, 0.5, t_Ein)
    Eout_point = Dirac(Eout, 0.5, t_Eout)
    
    evaluation_value = peak_point+LW_point+wave_point+Ein_point+Eout_point
    
        
    return evaluation_value, peak_point, LW_point, wave_point, Ein_point, Eout_point



