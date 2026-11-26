# -*- coding: utf-8 -*-
"""
REINFORCE policy (class Policy: GRU that outputs a categorical distribution over the
thickness of each of the 24 layers) and agent (class GRUAgent: sampling and policy-gradient
update with an entropy term weighted by beta). Imported by the REINFORCE sweep scripts.
Running this file trains a single policy (early exploration run, not the paper settings).

@author: Gibaek Kim
"""
import os, sys, time
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

path_c = os.path.dirname(os.path.abspath(__file__))
path_p = os.path.dirname(path_c)
path_gp = os.path.dirname(path_p)

sys.path.append(path_p)
sys.path.append(os.path.join(path_gp, 'Surrogate', '4QW_total'))

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from evaluation_surrogate import EVAL_7_7um_total
from evaluation_physical import EVAL_7_7um_tot
from train_surrogate_PBTL import PBTL

if torch.backends.mps.is_available():
    device = "mps"  # Metal Performance Shaders (Mac GPU)
elif torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
else:
    device = "cpu"

class Policy(nn.Module):
    def __init__(self, state_size, action_sizes, latent_dim):
        super().__init__()

        num_node = 32
        
        # policy neural network
        self.gru = nn.GRU(input_size = latent_dim, hidden_size = 1, num_layers = state_size)
                
        self.L1in, self.L1out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[0])
        self.L2in, self.L2out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[1])
        self.L3in, self.L3out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[2])
        self.L4in, self.L4out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[3])
        self.L5in, self.L5out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[4])
        self.L6in, self.L6out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[5])
        self.L7in, self.L7out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[6])
        self.L8in, self.L8out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[7])
        
        self.L9in, self.L9out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[8])
        self.L10in, self.L10out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[9])
        self.L11in, self.L11out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[10])
        self.L12in, self.L12out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[11])
        self.L13in, self.L13out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[12])
        self.L14in, self.L14out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[13])
        self.L15in, self.L15out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[14])
        self.L16in, self.L16out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[15])
        
        self.L17in, self.L17out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[16])
        self.L18in, self.L18out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[17])
        self.L19in, self.L19out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[18])
        self.L20in, self.L20out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[19])
        self.L21in, self.L21out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[20])
        self.L22in, self.L22out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[21])
        self.L23in, self.L23out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[22])
        self.L24in, self.L24out = nn.Linear(1, num_node), nn.Linear(num_node, action_sizes[23])
                
    def forward(self, x):
        
        _, hidden = self.gru(x)
        
        piL1 = F.leaky_relu(self.L1in(hidden[0]))
        piL1 = F.softmax(self.L1out(piL1), dim = -1)
        
        piL2 = F.leaky_relu(self.L2in(hidden[1]))
        piL2 = F.softmax(self.L2out(piL2), dim = -1)
        
        piL3 = F.leaky_relu(self.L3in(hidden[2]))
        piL3 = F.softmax(self.L3out(piL3), dim = -1)
        
        piL4 = F.leaky_relu(self.L4in(hidden[3]))
        piL4 = F.softmax(self.L4out(piL4), dim = -1)
        
        piL5 = F.leaky_relu(self.L5in(hidden[4]))
        piL5 = F.softmax(self.L5out(piL5), dim = -1)
        
        piL6 = F.leaky_relu(self.L6in(hidden[5]))
        piL6 = F.softmax(self.L6out(piL6), dim = -1)
        
        piL7 = F.leaky_relu(self.L7in(hidden[6]))
        piL7 = F.softmax(self.L7out(piL7), dim = -1)

        piL8 = F.leaky_relu(self.L8in(hidden[7]))
        piL8 = F.softmax(self.L8out(piL8), dim = -1)

        piL9 = F.leaky_relu(self.L9in(hidden[8]))
        piL9 = F.softmax(self.L9out(piL9), dim = -1)

        piL10 = F.leaky_relu(self.L10in(hidden[9]))
        piL10 = F.softmax(self.L10out(piL10), dim = -1)

        piL11 = F.leaky_relu(self.L11in(hidden[10]))
        piL11 = F.softmax(self.L11out(piL11), dim = -1)

        piL12 = F.leaky_relu(self.L12in(hidden[11]))
        piL12 = F.softmax(self.L12out(piL12), dim = -1)

        piL13 = F.leaky_relu(self.L13in(hidden[12]))
        piL13 = F.softmax(self.L13out(piL13), dim = -1)

        piL14 = F.leaky_relu(self.L14in(hidden[13]))
        piL14 = F.softmax(self.L14out(piL14), dim = -1)

        piL15 = F.leaky_relu(self.L15in(hidden[14]))
        piL15 = F.softmax(self.L15out(piL15), dim = -1)

        piL16 = F.leaky_relu(self.L16in(hidden[15]))
        piL16 = F.softmax(self.L16out(piL16), dim = -1)

        piL17 = F.leaky_relu(self.L17in(hidden[16]))
        piL17 = F.softmax(self.L17out(piL17), dim = -1)
        
        piL18 = F.leaky_relu(self.L18in(hidden[17]))
        piL18 = F.softmax(self.L18out(piL18), dim = -1)
        
        piL19 = F.leaky_relu(self.L19in(hidden[18]))
        piL19 = F.softmax(self.L19out(piL19), dim = -1)
        
        piL20 = F.leaky_relu(self.L20in(hidden[19]))
        piL20 = F.softmax(self.L20out(piL20), dim = -1)
        
        piL21 = F.leaky_relu(self.L21in(hidden[20]))
        piL21 = F.softmax(self.L21out(piL21), dim = -1)
        
        piL22 = F.leaky_relu(self.L22in(hidden[21]))
        piL22 = F.softmax(self.L22out(piL22), dim = -1)
        
        piL23 = F.leaky_relu(self.L23in(hidden[22]))
        piL23 = F.softmax(self.L23out(piL23), dim = -1)

        piL24 = F.leaky_relu(self.L24in(hidden[23]))
        piL24 = F.softmax(self.L24out(piL24), dim = -1)
        
        pi = [piL1,piL2,piL3,piL4,piL5,piL6,piL7,piL8
              ,piL9,piL10,piL11,piL12,piL13,piL14,piL15,piL16,piL17,piL18,piL19,piL20,piL21,piL22,piL23,piL24]
        
        return pi
      
class GRUAgent:
    def __init__(self, n_batch, state_size, action_size, model, optimizer):
        
        self.n_batch = n_batch
        self.state_size = state_size
        self.action_size = action_size
        
        self.model = model
        self.optimizer = optimizer
        
    def act(self, state):
                
        # calculate probability and next state
        state = state.reshape([1, state.shape[0], state.shape[1]])
        pi = self.model(state) # pi[state_size][n_particle])
        
        # initialize actions
        actions = np.zeros([self.n_batch, self.state_size])
        
        for ii in range(self.state_size):
            for jj in range(self.n_batch):
                actions[jj, ii] = np.random.choice(self.action_size[ii],
                                                p=pi[ii][jj].detach().cpu().numpy())
   
        return actions
   
    def learn(self, state, action, reward, beta):
        
        action = torch.Tensor(action).to(device).to(torch.int64)
        reward = torch.FloatTensor(reward).to(device)
                    
        state = state.reshape([1, state.shape[0], state.shape[1]])
        
        # calculate action and advantage
        pi = self.model(state) # pi[state_size][n_particle]
        
        # Cross enrtopy error
        for kk in range(24):
            if kk == 0:
                loss_pi_a = -torch.log(torch.gather(pi[:][kk], 1, action[:,kk].reshape([self.n_batch, 1]))).squeeze()*reward.squeeze()
                loss_H = -torch.sum(pi[:][kk]*torch.log(pi[:][kk]), 1)
            elif kk < 8:
                loss_pi_a += -torch.log(torch.gather(pi[:][kk], 1, action[:,kk].reshape([self.n_batch, 1]))).squeeze()*reward.squeeze()
                loss_H += -torch.sum(pi[:][kk]*torch.log(pi[:][kk]), 1)        
            else:
                loss_pi_a += -torch.log(torch.gather(pi[:][kk], 1, action[:,kk].reshape([self.n_batch, 1]))).squeeze()*reward.squeeze()
        
        self.loss = loss_pi_a - beta*loss_H
        
        self.optimizer.zero_grad()
        self.loss.mean().backward()
        self.optimizer.step()
    

#%% Main
if __name__ == "__main__":
    
    # Load surrogate model
    xfile = path_gp + '/Surrogate/4QW_total/model/surrogate_PBTL_394k_full.pt'
    X_max = np.array([50., 24., 24., 60., 24., 60., 25., 50., 30., 50., 20., 40., 20.,
           40., 20., 40., 23., 33., 33., 33., 40., 35., 45., 35.])
    X_min = np.array([32., 11.,  9., 45.,  9., 42.,  9., 35., 16., 36., 11., 31., 11.,
           31., 11., 30., 14., 24., 24., 24., 26., 21., 31., 21.])
    Y_max = [49.8 ,  30.5 ,  11.9 , 164.32, 109.5]
    Y_min = [1.   , 6.73 , 5.13 , 0.88 , 1.304]
    
    # Hyperparameters
    EPISODES = 1000    # number of episode
    learning_rate  = 5e-2 # 1e-2 is current optimized value (231108)
    state_size = 24
    n_batch =64
    latent_dim = 64
    beta = 1e-5
    action_size = (X_max - X_min).astype(int)

    
    # Saving
    train = True
    saving = True
    save_name = "Test_1"
    patient = 50
    valid_target = 2
    fail_count = 0
    end_condition = 50
    
    
    '''Define REINFORCE model'''
    while train:
        
        start = time.time()
        
        validation = 0
        count = 0
        
        model = Policy(state_size, action_size, latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        agent = GRUAgent(n_batch, state_size, action_size, model, optimizer)
        
        
        '''For recording'''
        history_max = []
        history_mean = []
        global_reward = 0
        global_reward_history = np.zeros([EPISODES,6])
        global_struct_history = np.zeros([EPISODES, 24])
        checked_structure = np.zeros([1,24])
        
        best_evaluation = 0
        early_count = 0
        early_patient = 100
        
        for e in range(EPISODES):
            
            # Initialization
            start_e = time.time()
            
            initial_condition = torch.normal(0, 1, size = (n_batch, latent_dim)).to(device)
            
            ''' Episode starts '''
            # Define action
            actions = agent.act(initial_condition)
            
            # Define final state
            structure = np.zeros([n_batch, state_size])
            
            for ii in range(n_batch):
                for kk in range(state_size):
                    structure[ii,kk] = X_min[kk] + actions[ii,kk]
                
            # Reward calculation
            evaluation_value, peak_point, LW_point, wave_point, Ein_point, Eout_point = EVAL_7_7um_total(structure, X_max, X_min, Y_max = Y_max, Y_min = Y_min, xfile = xfile,
                                                                                                         t_gain=24, t_LW=24, t_wave = 7.7, t_Ein=26, t_Eout=26)
            reward = (evaluation_value)/25
            
            agent.learn(initial_condition, actions, reward, beta)
            
            """determine the candidate for validation process"""
            
            elite_defined = False
            elite_index = np.argsort(evaluation_value)[::-1]
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
            
            if np.max(evaluation_value) < best_evaluation:
                early_count += 1
            elif np.max(evaluation_value)>4.5:
                early_count = 0
            else:
                early_count = 0
                best_evaluation = np.max(evaluation_value)
            
            if early_count> early_patient:
                break
            
            if np.max(evaluation_value) > 4.7:
                
                real_eval, peak_point, LW_point, wave_point, Ein_point, Eout_point = EVAL_7_7um_tot(elite_particle, t_gain=24, t_LW=24, t_wave = 7.7, t_Ein=26, t_Eout=26)
                
                if real_eval > 4.8:
                    validation += 1
                    count = 0
                    if validation == valid_target:
                        train = False
                        saving = True
                        break
                 
                if  real_eval > global_reward:
                    count = 0 
                    global_reward = real_eval
                    global_reward_history[e,:] = [real_eval[0], peak_point, LW_point, wave_point, Ein_point[0], Eout_point]
                    global_struct_history[e,:] = elite_particle
                
                    if global_reward>4.5:
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
                
            print("Ep: {}   Eval max: {:3.2f}   Eval mean: {:3.2f}   Global: {:3.2f}   Loss:   {:3.3f}   Time: {:3.3f}sec"\
                  .format(e+1, float(max(evaluation_value)),
                  float(np.mean(evaluation_value)),
                  float(global_reward),
                  agent.loss.mean().detach().cpu().numpy(),
                  time.time()-start_e))
    
            # Fitting
            history_max.append(max(evaluation_value))
            history_mean.append(np.mean(evaluation_value))
            
            
            if count > patient:
                print("Optimization fails")
                fail_count += 1
                if fail_count == end_condition:
                    saving = False
                break
                
    plt.pause(0.001)
    
    print('Complete')
    print('Optimization time: {}'.format(np.round(time.time()-start, 2)))
    #%% fitting
    if saving:
        torch.save(model, os.path.join(path_c, 'model', f'{save_name}.pt'))