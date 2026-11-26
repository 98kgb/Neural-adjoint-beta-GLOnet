# -*- coding: utf-8 -*-
"""
Neural-adjoint GLOnet / beta-GLOnet model definition (class GLOnet).

Generator: latent z (dim_latent) -> FC(512) -> 24 normalized layer thicknesses.
The frozen PBTL surrogate model provides the differentiable performance prediction.
Loss: FoM loss L_FoM(sigma1, sigma2) minus beta * batch variance of the generated
structures (Eqs. 8-10); beta = None gives the neural-adjoint GLOnet without the diversity term.

The class is imported by the sweep scripts. Running this file trains a single GLOnet with a
physical check of the elite candidates (early exploration run, not the paper settings).

@author: Gibaek Kim
"""
import os, sys, time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

path_c = os.path.dirname(os.path.abspath(__file__))
path_p = os.path.dirname(path_c)
path_gp = os.path.dirname(path_p)
path_ggp = os.path.dirname(path_gp)
path_gggp = os.path.dirname(path_ggp)


sys.path.append(path_p)
sys.path.append(path_ggp)
sys.path.append(path_gggp)
sys.path.append(os.path.join(path_gp, 'Surrogate', '4QW_total'))

import torch
import torch.nn as nn
import torch.optim as optim
from evaluation_physical import EVAL_7_7um_tot
from train_surrogate_PBTL import PBTL
from qcl_simulator import QCL
if torch.backends.mps.is_available():
    device = "mps"  # Metal Performance Shaders (Mac GPU)
elif torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
else:
    device = "cpu"
    
class GLOnet(nn.Module):
    def __init__(self, dim_latent, xfile, t_gain=24, t_LW=24, t_wave = 7.7, t_Ein=26, t_Eout=26):
        super(GLOnet, self).__init__()
        
        trained_model = torch.load(xfile, map_location = device, weights_only= False)
        
        self.t_gain = t_gain
        self.t_LW = t_LW
        self.t_wave = t_wave
        self.t_Ein = t_Ein
        self.t_Eout = t_Eout
        
        self.gen1 = nn.Linear(dim_latent,512)
        self.gen2 = nn.Linear(512,24)
        
        self.fc11 = list(trained_model.children())[0]
        self.fc12 = list(trained_model.children())[1]
        
        self.fc2 = list(trained_model.children())[2]
        self.fc3 = list(trained_model.children())[3]
        self.fc4 = list(trained_model.children())[4]
        self.fc5 = list(trained_model.children())[5]
        self.fc6 = list(trained_model.children())[6]
        
    def forward(self, x):
        
        y = nn.functional.leaky_relu(self.gen1(x))
        y = nn.functional.sigmoid(self.gen2(y))
        
        thick = y
        
        y1 = nn.functional.leaky_relu(self.fc11(y[:,:8]))
        y2 = nn.functional.leaky_relu(self.fc12(y[:,8:]))
        
        y = torch.hstack([y1,y2])
        
        y = nn.functional.leaky_relu(self.fc2(y))
        y = nn.functional.leaky_relu(self.fc3(y))
        y = nn.functional.leaky_relu(self.fc4(y))
        y = nn.functional.leaky_relu(self.fc5(y))
        y = nn.functional.leaky_relu(self.fc6(y))
        
        return thick, y
   
    def train(self, model, optimizer, x, sig1, sig2, beta, Y_max, Y_min):
        
        # calculate action and advantage
        thick, result  = model(x) 
        
        peak = result[:,0]*(Y_max[0]-Y_min[0]) + Y_min[0]
        LW = result[:,1]*(Y_max[1]-Y_min[1]) + Y_min[1]
        wave = result[:,2]*(Y_max[2]-Y_min[2]) + Y_min[2]
        Ein = result[:,3]*(Y_max[3]-Y_min[3]) + Y_min[3]
        Eout = result[:,4]*(Y_max[4]-Y_min[4]) + Y_min[4]
        
        peak_point =  1/(1+torch.exp(-0.5*(peak-self.t_gain)))
        LW_point = 1/(1+torch.exp(0.5*(LW-self.t_LW)))
        wave_point = (1/(1*torch.sqrt(torch.tensor(2*math.pi))))*torch.exp(-(1/2)*(wave-self.t_wave)**2/1**2)*2.5
        Ein_point = 1/(1+torch.exp(0.5*(Ein-self.t_Ein)))
        Eout_point = 1/(1+torch.exp(0.5*(Eout-self.t_Eout)))
        
        total_point = peak_point +  wave_point + Ein_point + Eout_point + LW_point
        
        # calculate variance loss
        loss_var = 0
        for ii in range(24):
            loss_var += torch.var(thick[:,ii])
        
        loss_var = loss_var
        loss_perform = (-1/sig1*torch.exp(result[:,0]/sig1) + 1/sig1*torch.exp(result[:,1]/sig1) + 
                        1/sig1*torch.exp(result[:,3]/sig1) + 1/sig1*torch.exp(result[:,4]/sig1) +
                        1 - (1 / (sig2 * torch.sqrt(torch.tensor(2*math.pi)))) * torch.exp(-1/2*(wave-self.t_wave)**2/sig2**2)*2.5)

        if beta == None:
            self.loss = loss_perform            
        else:
            self.loss = loss_perform - beta * loss_var
        
        optimizer.zero_grad()
        
        self.loss_FoM = loss_perform
        self.loss_Div = loss_var
        
        self.loss.mean().backward()
        
        
        optimizer.step()
            
        return thick, total_point


#%% Main
if __name__ == "__main__":
    xfile = path_gp + '/Surrogate/4QW_total/model/surrogate_PBTL_394k_full.pt'
    trained_model = torch.load(xfile, map_location = device, weights_only= False)
    
    X_max = np.array([50., 24., 24., 60., 24., 60., 25., 50., 30., 50., 20., 40., 20.,
           40., 20., 40., 23., 33., 33., 33., 40., 35., 45., 35.])
    X_min = np.array([32., 11.,  9., 45.,  9., 42.,  9., 35., 16., 36., 11., 31., 11.,
           31., 11., 30., 14., 24., 24., 24., 26., 21., 31., 21.])
    Y_max = [49.8 ,  30.5 ,  11.9 , 164.32, 109.5]
    Y_min = [1.   , 6.73 , 5.13 , 0.88 , 1.304]
    
    # ================== hyperparameters ==================
    
    epoch = 1000 # number of episode
    learning_rates = [1e-3, 1e-4]
    n_batchs = [4,16,32,64]
    dim_latents = [4,16,32,64]
    betas = [50]
    sig1 = 1
    sig2 = 1
    
    learning_rate = 1e-3
    n_batch = 32
    dim_latent = 16
    beta = 3
    
    # =====================================================
    
    # Train criteria
    patient = 50
    valid_target = 2
    fail_count = 0
    end_condition = 50
    success_num = 0
    
    ii = 0
    train = True
    start = time.time()
    save_name = f"GLO_beta_{beta}_lr_{learning_rate}_batch_{n_batch}_dim_{dim_latent}_trial_{ii}"
    
    if os.path.exists(os.path.join(path_gp + f"/Algorithms/GLOnet/model/info/{save_name}_info.csv")):
        print(f'{save_name} already existed')
        ii += 1
    else:
        check = False
        print(f'\n{save_name} model training start!\n')
        
    final_structures = []
    
    for trial in range(50):
        validation = 0
        count = 0
        
        # model define
        model = GLOnet(dim_latent, xfile).to(device)
        
        name_param = []
        for ii in range(1,7):
            name_param.append(f'fc{ii}.weight')
            name_param.append(f'fc{ii}.bias')
        name_param.append('fc11.weight')
        name_param.append('fc12.weight')
        name_param.append('fc11.bias')
        name_param.append('fc12.bias')
        
        for name, param in model.named_parameters():
            if name in name_param:
                param.requires_grad = False
            
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        
        '''For recording'''
        hist_max = []
        hist_mean = []
        hist_actual = []
        hist_loss = []
        
        global_reward = 0
        global_reward_history = np.zeros([epoch,6])
        global_struct_history = np.zeros([epoch, 24])
        checked_structure = np.zeros([1,24])
        
        for e in range(epoch):
            
            initial_condition = (torch.rand([n_batch,dim_latent]).to(device)-0.5)*2  # range -1 to 1
            # initial_condition = torch.normal(0, 1, size = (n_particle,dim_latent)).to(device)
            structure, total_point = model.train(model, optimizer, initial_condition,
                                                 sig1 = sig1, sig2= sig2, beta = beta, Y_max = Y_max, Y_min = Y_min)
            
            # convert to numpy
            structure = structure.detach().cpu().numpy()
            structure = np.round(structure*(X_max-X_min)+X_min)
            total_point = total_point.detach().cpu().numpy()
            
            """determine the candidate for validation process"""
            elite_defined = False
            elite_index = np.argsort(total_point)[::-1]
            elite_particle = np.zeros([1,24])
            
            for ii in range(n_batch):
                elite_candidate = structure[int(elite_index[ii]),:]
                for jj in range(len(checked_structure[:,0])):
                    if np.array_equal(elite_candidate, checked_structure[jj,:]):
                        break
                    
                    else:
                        elite_particle = elite_candidate
                        checked_structure = np.vstack([checked_structure,elite_candidate])
                        elite_defined  = True
                        break
                if elite_defined:
                    break
            
            if np.max(total_point) > 4.8:                
                # evaluation_value, peak_point, LW_point, wave_point, Ein_point, Eout_point = EVAL_7_7um_tot(elite_particle, t_gain=24, t_LW=24, t_wave = 7.7, t_Ein=26, t_Eout=26)
                evaluation_value, peak_point, LW_point, wave_point, Ein_point, Eout_point = EVAL_7_7um_tot(elite_particle, t_gain=24, t_LW=24, t_wave=7.7, t_Ein=26, t_Eout=26)
                evaluation_value = np.asarray(evaluation_value).item()
                if evaluation_value > 4.8:
                    validation += 1
                    count = 0
                    if validation == valid_target:
                        success_num += 1
                        break
                 
                if  evaluation_value > global_reward:
                    count = 0 
                    global_reward = evaluation_value
                    global_reward_history[e,:] = [evaluation_value, peak_point, LW_point, wave_point, Ein_point[0], Eout_point]
                    global_struct_history[e,:] = elite_particle
                
                    if global_reward>4.8:
                        patient = 200
                        print("This model is worthy")
                        
                else:
                    count += 1
                    print("Early stopping count: {}  validation: {}".format(count, validation))
                    global_reward_history[e,:] = global_reward_history[e-1,:]
                    global_struct_history[e,:] = global_struct_history[e-1,:]
            else:
                global_reward_history[e,:] = global_reward_history[e-1,:]
                global_struct_history[e,:] = global_struct_history[e-1,:]
                
            hist_actual.append(float(global_reward))
            hist_max.append(max(total_point))
            hist_mean.append(np.mean(total_point))
            hist_loss.append(model.loss.mean().detach().cpu().numpy())
            
            print("Ep: {}   Eval max: {:3.2f}   Eval mean: {:3.2f}   Global: {:3.2f}   Loss:   {:3.3f}".format(e, float(max(total_point)),
                                                                                                               float(np.mean(total_point)),
                                                                                                               float(global_reward),
                                                                                                               model.loss.mean().detach().cpu().numpy()))
            if count > patient:
                print("Optimization fails")
                fail_count += 1
                if fail_count == end_condition:
                    saving = False
                break
        
        print('Complete')
        print('Optimization time: {}'.format(np.round(time.time()-start, 2)))
        # trial_num[idx,idx_trial] = success_num
        
        hist_actual = np.array(hist_actual).reshape([-1,1])
        hist_max = np.array(hist_max).reshape([-1,1])
        hist_mean = np.array(hist_mean).reshape([-1,1])
        hist_loss = np.array(hist_loss).reshape([-1,1])
        hist_tot = np.hstack([hist_actual, hist_max, hist_mean, hist_loss])
        df = pd.DataFrame(hist_tot, columns = ['Actual best', 'Max_obj', 'Mean_obj', 'loss'])
        
        model_info_saving = False
        if model_info_saving:
            df.to_csv(os.path.join(path_c, 'model', 'info', f'{save_name}_info.csv'))
            torch.save(model, os.path.join(path_c, 'model', f'{save_name}.pt'))
        
        #%%
        valid = True
        valid_counter = 0
        while valid:
            initial_condition = (torch.rand([n_batch,dim_latent]).to(device)-0.5)*2  # range -1 to 1
            structure, total_point = model.train(model, optimizer, initial_condition,
                                                 sig1 = sig1, sig2= sig2, beta = beta, Y_max = Y_max, Y_min = Y_min)
            structure = structure.detach().cpu().numpy()
            structure = np.round(structure*(X_max-X_min)+X_min)
            print(f'Trial {trial} validating...')
            print(f'{structure[0,:]} evaluating...')
            
            cal = QCL(structure[0,:], LW_cal=True)
            cal.Potential()
            cal.EVMAT()
            cal.CBEIGEN()
            cal.Dipole()
            cal.Scatter_rate()
            lamda, g43 = cal.gain()
            valid_counter += 1
            
            if np.max(g43)>1 or valid_counter > 10:
                fig, ax = plt.subplots(1, 2, figsize = (12,5))
                fontsize = 20
                plt.suptitle(f'Trial: {trial}', fontsize = 20)
                ax[1].plot(lamda, g43, 'b', linewidth = 2, alpha = 0.7, label = 'Optimized')
                ax[1].grid()
                ax[1].set_xlabel('$Wavelength\ (\mu m)$', fontsize = fontsize)
                ax[1].set_ylabel('$G_{43}\ (cm^{-1})$', fontsize = fontsize)
                ax[1].axvline(x = 7.7, linestyle = ':', c = 'r')        
                ax[0].plot(cal.pos, cal.CB_potential, 'k')
            
                # ax[0].set_title('Einj: {}  Eext: {}'.format(np.round(abs(model.Ein0[0]*1000),2), np.round(model.Eout*1000, 2)))
                for kk in range(20):
                    if kk == cal.st1:
                        ax[0].plot(cal.pos, (cal.cbwave[:, kk])**2*10 + cal.energy[kk], 'r', linewidth = 2, label = '$\Phi_{1}$')
                    elif kk == cal.st2:
                        ax[0].plot(cal.pos, (cal.cbwave[:, kk])**2*10 + cal.energy[kk], 'g', linewidth = 2, label = '$\Phi_{2}$')
                    elif kk == cal.st3:
                        ax[0].plot(cal.pos, (cal.cbwave[:, kk])**2*10 + cal.energy[kk], 'b', linewidth = 2, label = '$\Phi_{3}$')
                    elif kk == cal.st4:
                        ax[0].plot(cal.pos, (cal.cbwave[:, kk])**2*10 + cal.energy[kk], 'm', linewidth = 2, label = '$\Phi_{4}$')
                    elif kk == cal.stin0:
                        ax[0].plot(cal.pos, (cal.cbwave[:, kk])**2*10 + cal.energy[kk], 'c', linewidth = 2, label = '$\Phi_{inj}$')
                    elif kk == cal.stex:
                        ax[0].plot(cal.pos, (cal.cbwave[:, kk])**2*10 + cal.energy[kk], 'y', linewidth = 2, label = '$\Phi_{ext}$')
                    else:
                        ax[0].plot(cal.pos, (cal.cbwave[:, kk])**2*10 + cal.energy[kk], '--k', linewidth = 0.5)
            
                ax[0].set_xlim([0,1100])
                ax[0].set_xlabel('$Position\ (\AA)$', fontsize = fontsize)
                ax[0].set_ylabel('$Potential\ (eV)$', fontsize = fontsize)
                handles, labels = ax[0].get_legend_handles_labels()
            
                label_order = ['$\\Phi_{1}$', '$\\Phi_{2}$', '$\\Phi_{3}$', '$\\Phi_{4}$', '$\\Phi_{inj}$', '$\\Phi_{ext}$']
                # ordered_handles = [handles[labels.index(lab)] for lab in label_order]
                # ax[0].legend(ordered_handles, label_order, fontsize = fontsize-3)
                plt.show()    
                valid = False
                final_structures.append(structure[0,:])
    np.savetxt(os.path.join(path_c, 'Valid_designs.csv'), final_structures, delimiter = ',')