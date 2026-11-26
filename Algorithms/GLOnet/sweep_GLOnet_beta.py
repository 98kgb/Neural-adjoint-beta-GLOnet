# -*- coding: utf-8 -*-
"""
beta sweep of beta-GLOnet (Fig. 4).

Trains GLOnet with different diversity weights beta (lr = 1e-3, batch 64, latent dim 64,
500 epochs, sigma1 = 1, sigma2 = 0.5, 50 trials per beta) and records the max / mean FoM and the
structural variance of the generated designs. Set the list `betas` to the values to run.

Outputs (analysis/beta_sweep): per-run final structures / FoM (.npy) and beta_sweep_log.csv

@author: Gibaek Kim
"""
import os, sys, time, datetime, math
import numpy as np
import pandas as pd

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

path_c = os.path.dirname(os.path.abspath(__file__))
path_p = os.path.dirname(path_c)
path_gp = os.path.dirname(path_p)
path_ggp = os.path.dirname(path_gp)

sys.path.append(path_c)
sys.path.append(path_p)
sys.path.append(path_ggp)
sys.path.append(os.path.join(path_gp, 'Surrogate', '4QW_total'))

import torch
import torch.nn as nn
import torch.optim as optim
from train_GLOnet import GLOnet

from evaluation_physical import EVAL_7_7um_tot
from train_surrogate_PBTL import PBTL

if torch.backends.mps.is_available():
    device = "mps"  # Metal Performance Shaders (Mac GPU)
elif torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
else:
    device = "cpu"
    
def evaluate_model(model,initial_condition,Y_max,Y_min):
    with torch.no_grad():
        structure,result=model(initial_condition)
        peak=result[:,0]*(Y_max[0]-Y_min[0])+Y_min[0]
        LW=result[:,1]*(Y_max[1]-Y_min[1])+Y_min[1]
        wave=result[:,2]*(Y_max[2]-Y_min[2])+Y_min[2]
        Ein=result[:,3]*(Y_max[3]-Y_min[3])+Y_min[3]
        Eout=result[:,4]*(Y_max[4]-Y_min[4])+Y_min[4]
        peak_point=1/(1+torch.exp(-0.5*(peak-model.t_gain)))
        LW_point=1/(1+torch.exp(0.5*(LW-model.t_LW)))
        wave_point=(1/(torch.sqrt(torch.tensor(2*math.pi,device=result.device))))*torch.exp(-0.5*(wave-model.t_wave)**2)*2.5
        Ein_point=1/(1+torch.exp(0.5*(Ein-model.t_Ein)))
        Eout_point=1/(1+torch.exp(0.5*(Eout-model.t_Eout)))
        total_point=peak_point+LW_point+wave_point+Ein_point+Eout_point
    return structure,total_point


models = ['DNN']

save_dir=os.path.join(path_c,'analysis','beta_sweep')
os.makedirs(save_dir,exist_ok=True)

for model_name in models:
    # =============================================================
    log_path = os.path.join(path_c, "beta_sweep_log.csv")
    log_columns = [
        'timestamp', 'save_name', 'learning_rate', 'beta',
        'n_batch', 'dim_latent', 'Obj_max', 'Obj_mean', 'Str_variance', 'training_time']
    # =============================================================
    
    # define DNN surrogate model
    xfile = path_gp + '/Surrogate/4QW_total/model/surrogate_PBTL_394k_full.pt'
    
    X_max = np.array([50., 24., 24., 60., 24., 60., 25., 50., 30., 50., 20., 40., 20.,
           40., 20., 40., 23., 33., 33., 33., 40., 35., 45., 35.])
    X_min = np.array([32., 11.,  9., 45.,  9., 42.,  9., 35., 16., 36., 11., 31., 11.,
           31., 11., 30., 14., 24., 24., 24., 26., 21., 31., 21.])
    Y_max = [49.8 ,  30.5 ,  11.9 , 164.32, 109.5]
    Y_min = [1.   , 6.73 , 5.13 , 0.88 , 1.304]
    
    # hyperparameters
    epoch = 500 # number of episode
    # learning_rates = [5e-4, 1e-3,5e-3,1e-2]
    # n_batchs = [4,16,32,64]
    # dim_latents = [16,64,128,256]
    # betas = [0.5, 1, 1.5, 3, None]
    
    learning_rates = [1e-3]
    n_batchs = [64]
    dim_latents = [64]
    # betas = [0, 0.5, 1, 1.5, 3]
    betas = [2,2.5]
    
    sig1 = 1
    sig2 = 0.5
    training_time = []
    for trial_idx in range(50):
        for beta in betas:
            for learning_rate in learning_rates:
                for n_batch in n_batchs:
                    for dim_latent in dim_latents:
                        seed = 2026 + trial_idx
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
                        
                        save_name = f"GLO_beta_{beta:g}_lr_{learning_rate:g}_batch_{n_batch}_dim_{dim_latent}_trial_{trial_idx:02d}"
                        start = time.time()
                        
                        # model define
                        model = GLOnet(dim_latent, xfile).to(device)
                            
                        name_param = []
                        for kk in range(1,7):
                            name_param.append(f'fc{kk}.weight')
                            name_param.append(f'fc{kk}.bias')
                        name_param.append('fc11.weight')
                        name_param.append('fc12.weight')
                        name_param.append('fc11.bias')
                        name_param.append('fc12.bias')
                        
                        for name, param in model.named_parameters():
                            if name in name_param:
                                param.requires_grad = False
                            
                        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
                        
                        '''For recording'''
                        hist_FoM_max, hist_FoM_mean  = [], []
                        hist_loss_var, hist_loss_FoM, hist_loss_tot = [], [], []
                        hist_str, hist_DD = [], []
                        
                        for e in range(epoch):
                            
                            initial_condition = (torch.rand([n_batch,dim_latent]).to(device)-0.5)*2  # range -1 to 1
                            structure, total_point = model.train(model, optimizer, initial_condition,
                                                                 sig1 = sig1, sig2= sig2, beta = beta, Y_max = Y_max, Y_min = Y_min)
                            
                            # convert to numpy
                            structure = structure.detach().cpu().numpy()
                            structure = np.round(structure*(X_max-X_min)+X_min)
                            
                            if e % 50 == 0:
                                hist_str.append(structure)
                            
                            total_point = total_point.detach().cpu().numpy()
                                
                            hist_FoM_max.append(max(total_point))
                            hist_FoM_mean.append(np.mean(total_point))
                            
                            hist_DD.append(float(np.mean(structure.std(axis=0))))
                            hist_loss_var.append(model.loss_Div.mean().detach().cpu().numpy())
                            hist_loss_FoM.append(model.loss_FoM.mean().detach().cpu().numpy())
                            hist_loss_tot.append(model.loss.mean().detach().cpu().numpy())
                                    
                            print(f"Ep: {e+1}   Eval max: {float(max(total_point)):3.2f}   Eval mean: {float(np.mean(total_point)):3.2f}   Var: {hist_loss_var[e]:3.2f}   Loss:   {model.loss.mean().detach().cpu().numpy():3.3f}")
                        
                        print('Complete')
                        print('Optimization time: {}'.format(np.round(time.time()-start, 2)))
                        training_time.append(np.round(time.time()-start, 2))
                        hist_max = np.array(hist_FoM_max).reshape([-1,1])
                        hist_mean = np.array(hist_FoM_mean).reshape([-1,1])
                        hist_loss_div = np.array(hist_loss_var).reshape([-1,1])
                        hist_loss_fom = np.array(hist_loss_FoM).reshape([-1,1])
                        hist_loss_tot = np.array(hist_loss_tot).reshape([-1,1])
                        hist_DD=np.array(hist_DD).reshape([-1,1])
                        hist_tot=np.hstack([hist_max,hist_mean,hist_DD,hist_loss_div,hist_loss_fom,hist_loss_tot])
                        
                        df=pd.DataFrame(hist_tot,columns=['Max_FoM','Mean_FoM','DD','L_Div','L_FoM','L_total'])
                        df.to_csv(os.path.join(save_dir,f'{save_name}_info.csv'),index=False)

                        # generate 100 structures and evaluate variance
                        eval_seed=10000+trial_idx
                        torch.manual_seed(eval_seed)
                        if torch.cuda.is_available(): torch.cuda.manual_seed_all(eval_seed)
                        
                        initial_condition=(torch.rand([100,dim_latent],device=device)-0.5)*2
                        structure,total_point=evaluate_model(model,initial_condition,Y_max,Y_min)
                        
                        structure=structure.cpu().numpy()
                        total_point=total_point.cpu().numpy()
                        structure=np.round(structure*(X_max-X_min)+X_min)
                        str_variance=np.mean(structure.std(axis=0))
                        final_FoM_max=float(np.max(total_point))
                        final_FoM_mean=float(np.mean(total_point))
                        
                        hist_str = np.array(hist_str)
                        np.save(os.path.join(save_dir,f'{save_name}_final_structures.npy'),structure)
                        np.save(os.path.join(save_dir,f'{save_name}_final_fom.npy'),total_point)
                        
                        # =============================================================
                        log_data = {'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'save_name': save_name,
                            'learning_rate': learning_rate, 'beta': beta,
                            'n_batch': n_batch, 'dim_latent': dim_latent,
                            'Obj_max':final_FoM_max,
                            'Obj_mean':final_FoM_mean,
                            'Str_variance':str_variance, 'training_time': time.time()-start}
                        # =============================================================
                        try:
                            log_df = pd.DataFrame([log_data], columns=log_columns)
                            
                            log_df.to_csv(
                                log_path, 
                                mode='a', 
                                header=not os.path.exists(log_path), 
                                index=False
                            )
                        except Exception as e:
                            print(f"Error writing to log file: {e}")



