# -*- coding: utf-8 -*-
"""
REINFORCE results of Section 3.2 (solver-validated success rate, Pass@K, distinct high-FoM designs).

GRU policy, batch 64, latent dim 64, lr = 3e-3, entropy weight 1e-5, 500 episodes, 50 trials.
Reward = surrogate FoM / 25 (the physical model never enters the reward).
Solver-validated protocol identical to run_GA_passK.py:
  * every episode, the best not-yet-checked candidate with surrogate FoM >= 4.8 is evaluated once
    with the physical model
  * stop after 2 physical passes, 200 solver calls, or 100 consecutive calls without a pass
  * afterwards 64 samples from the final policy -> top-K by surrogate -> physical evaluation

Outputs (Result/pass_at_K/REINFORCE_*): REINFORCE_physical_validation.csv, REINFORCE_phys_log.csv,
REINFORCE_training_validation.csv

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
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from evaluation_surrogate import EVAL_7_7um_total
from evaluation_physical import EVAL_7_7um_tot
from train_surrogate_PBTL import PBTL

device = "cpu"   # tiny policy net: cpu is faster than mps; learn() uses this module-level device

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
        
        piL1 = F.leaky_relu(self.L1in(hidden[0])); piL1 = F.softmax(self.L1out(piL1), dim = -1)
        piL2 = F.leaky_relu(self.L2in(hidden[1])); piL2 = F.softmax(self.L2out(piL2), dim = -1)
        piL3 = F.leaky_relu(self.L3in(hidden[2])); piL3 = F.softmax(self.L3out(piL3), dim = -1)        
        piL4 = F.leaky_relu(self.L4in(hidden[3]));piL4 = F.softmax(self.L4out(piL4), dim = -1)
        piL5 = F.leaky_relu(self.L5in(hidden[4])); piL5 = F.softmax(self.L5out(piL5), dim = -1)
        piL6 = F.leaky_relu(self.L6in(hidden[5])); piL6 = F.softmax(self.L6out(piL6), dim = -1)
        piL7 = F.leaky_relu(self.L7in(hidden[6])); piL7 = F.softmax(self.L7out(piL7), dim = -1)
        piL8 = F.leaky_relu(self.L8in(hidden[7])); piL8 = F.softmax(self.L8out(piL8), dim = -1)
        piL9 = F.leaky_relu(self.L9in(hidden[8])); piL9 = F.softmax(self.L9out(piL9), dim = -1)
        piL10 = F.leaky_relu(self.L10in(hidden[9])); piL10 = F.softmax(self.L10out(piL10), dim = -1)
        piL11 = F.leaky_relu(self.L11in(hidden[10])); piL11 = F.softmax(self.L11out(piL11), dim = -1)
        piL12 = F.leaky_relu(self.L12in(hidden[11])); piL12 = F.softmax(self.L12out(piL12), dim = -1)
        piL13 = F.leaky_relu(self.L13in(hidden[12])); piL13 = F.softmax(self.L13out(piL13), dim = -1)
        piL14 = F.leaky_relu(self.L14in(hidden[13])); piL14 = F.softmax(self.L14out(piL14), dim = -1)
        piL15 = F.leaky_relu(self.L15in(hidden[14])); piL15 = F.softmax(self.L15out(piL15), dim = -1)
        piL16 = F.leaky_relu(self.L16in(hidden[15])); piL16 = F.softmax(self.L16out(piL16), dim = -1)
        piL17 = F.leaky_relu(self.L17in(hidden[16])); piL17 = F.softmax(self.L17out(piL17), dim = -1)        
        piL18 = F.leaky_relu(self.L18in(hidden[17])); piL18 = F.softmax(self.L18out(piL18), dim = -1)        
        piL19 = F.leaky_relu(self.L19in(hidden[18])); piL19 = F.softmax(self.L19out(piL19), dim = -1)
        piL20 = F.leaky_relu(self.L20in(hidden[19])); piL20 = F.softmax(self.L20out(piL20), dim = -1)
        piL21 = F.leaky_relu(self.L21in(hidden[20])); piL21 = F.softmax(self.L21out(piL21), dim = -1)
        piL22 = F.leaky_relu(self.L22in(hidden[21])); piL22 = F.softmax(self.L22out(piL22), dim = -1)
        piL23 = F.leaky_relu(self.L23in(hidden[22])); piL23 = F.softmax(self.L23out(piL23), dim = -1)
        piL24 = F.leaky_relu(self.L24in(hidden[23])); piL24 = F.softmax(self.L24out(piL24), dim = -1)
        
        pi = [piL1,piL2,piL3,piL4,piL5,piL6,piL7,piL8,piL9,piL10,piL11,piL12,piL13,piL14,piL15,piL16,
              piL17,piL18,piL19,piL20,piL21,piL22,piL23,piL24]
        
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

if __name__=="__main__":
    # ---------------- experiment switches (keep identical to the GA / GLOnet scripts) ----------------
    TARGET=dict(t_gain=25,t_LW=25,t_wave=7.7,t_Ein=26,t_Eout=26)
    USE_PHYSICAL_GUIDANCE=True
    N_PHYS=200
    PATIENCE_PHYS=100          # consecutive physical calls without a pass (None = off)
    PASS_TARGET=2
    SURR_GATE=4.8
    physical_threshold=4.8

    xfile=path_gp+'/Surrogate/4QW_total/model/surrogate_PBTL_394k_full.pt'

    X_max=np.array([50.,24.,24.,60.,24.,60.,25.,50.,30.,50.,20.,40.,20.,40.,20.,40.,23.,33.,33.,33.,40.,35.,45.,35.])
    X_min=np.array([32.,11.,9.,45.,9.,42.,9.,35.,16.,36.,11.,31.,11.,31.,11.,30.,14.,24.,24.,24.,26.,21.,31.,21.])
    Y_max=[49.8,30.5,11.9,164.32,109.5]
    Y_min=[1.,6.73,5.13,0.88,1.304]

    EPISODES=500
    learning_rate=3e-3
    state_size=24
    n_batch=64
    latent_dim=64
    beta=1e-5
    # NOTE: (X_max-X_min) categories -> thickness in [X_min, X_max-1]; GA/GLOnet can reach X_max.
    # Set True to make the REINFORCE search space identical (X_max included) -- changes the tuned behaviour.
    ACTION_INCLUSIVE=False
    action_size=(X_max-X_min).astype(int)+(1 if ACTION_INCLUSIVE else 0)

    n_trials=50
    n_sample=64
    K_list=[1,4,8,16,32]
    max_K=max(K_list)

    tag='physguided' if USE_PHYSICAL_GUIDANCE else 'surronly'
    out_dir=os.path.join(path_c,'Result','pass_at_K',f'REINFORCE_lr{learning_rate:g}_{tag}_nphys{N_PHYS}')
    os.makedirs(out_dir,exist_ok=True)

    def set_seed(seed):
        np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    def evaluate(s):
        return np.asarray(EVAL_7_7um_total(s,X_max,X_min,Y_max=Y_max,Y_min=Y_min,xfile=xfile,**TARGET)[0]).reshape(-1)

    def fast_act(agent,z):
        """same categorical sampling as agent.act, one multinomial call per layer instead of 1536 np.random.choice"""
        with torch.no_grad():
            pi=agent.model(z.reshape([1,z.shape[0],z.shape[1]]))
            return np.stack([torch.multinomial(p,1).squeeze(1).cpu().numpy() for p in pi],axis=1).astype(float)

    def phys_eval(x):
        try:
            v,*comp=EVAL_7_7um_tot(x,**TARGET)
            v=np.asarray(v).item()
            valid=int(np.isfinite(v) and v>0)
            return v,valid,int(valid and v>=physical_threshold)
        except Exception:
            return np.nan,0,0

    candidate_results=[]
    phys_log=[]
    training=[]
    total_start=time.time()

    for trial in range(n_trials):
        print('\n'+'='*100)
        print(f'REINFORCE | Trial {trial+1}/{n_trials}')
        print('='*100)

        set_seed(2026+trial)
        model=Policy(state_size,action_size,latent_dim).to(device)
        optimizer=optim.Adam(model.parameters(),lr=learning_rate)
        agent=GRUAgent(n_batch,state_size,action_size,model,optimizer)

        checked=set()
        n_phys=0; n_pass=0; since_pass=0; first_pass_call=np.nan; best_phys=0.
        stop_reason='max_episode'
        t0=time.time()

        # ================= Training (surrogate reward only) + physical validation (stop rule only) =================
        for e in range(EPISODES):
            z=torch.normal(0,1,size=(n_batch,latent_dim)).to(device)
            actions=fast_act(agent,z)
            structure=X_min[None,:]+actions
            f=evaluate(structure)
            agent.learn(z,actions,f/25,beta)

            if USE_PHYSICAL_GUIDANCE:
                cand=None
                for idx in np.argsort(f)[::-1]:
                    key=tuple(structure[idx].astype(int))
                    if key not in checked:
                        checked.add(key); cand=structure[idx].copy(); s_val=float(f[idx]); break

                if cand is not None and s_val>=SURR_GATE:
                    v,valid,passed=phys_eval(cand)
                    n_phys+=1
                    if np.isfinite(v): best_phys=max(best_phys,v)
                    since_pass=0 if passed else since_pass+1
                    if passed:
                        n_pass+=1
                        if np.isnan(first_pass_call): first_pass_call=n_phys
                    phys_log.append([trial+1,e+1,n_phys,s_val,v,valid,passed])

                    if n_pass>=PASS_TARGET:
                        stop_reason='pass_target'; print(f'Pass target reached at episode {e+1}'); break
                    if n_phys>=N_PHYS:
                        stop_reason='phys_budget'; break
                    if PATIENCE_PHYS is not None and since_pass>=PATIENCE_PHYS:
                        stop_reason='no_pass_patience'; break

            if (e+1)%100==0:
                print(f'Episode {e+1}/{EPISODES} | Max={f.max():.4f} | Mean={f.mean():.4f} | '
                      f'Physical best={best_phys:.4f} | calls={n_phys} | passes={n_pass}')

        train_time=time.time()-t0
        training.append([trial+1,e+1,n_phys,n_pass,first_pass_call,best_phys,stop_reason,train_time])
        print(f'Stop={stop_reason} | episode={e+1} | phys calls={n_phys} | in-training passes={n_pass} | {train_time:.1f}s')

        # ================= Post-hoc: 64 samples -> top-K -> physical (not counted in N_PHYS) =================
        set_seed(10000+trial)
        S=[]
        for _ in range(max(1,n_sample//n_batch)):
            z=torch.normal(0,1,size=(n_batch,latent_dim)).to(device)
            S.append(X_min[None,:]+fast_act(agent,z))
        all_structure=np.vstack(S)
        all_surrogate=evaluate(all_structure)

        elite_index=np.argsort(all_surrogate)[::-1][:max_K]
        top_structure=all_structure[elite_index]
        top_surrogate=all_surrogate[elite_index]
        print(f'Top-{max_K} unique: {len(np.unique(top_structure,axis=0))}/{max_K}')

        for rank in range(max_K):
            candidate=top_structure[rank].copy()
            duplicate=int(rank>0 and any(np.array_equal(candidate,top_structure[j]) for j in range(rank)))
            physical,valid,passed=phys_eval(candidate)
            candidate_results.append([trial+1,rank+1,top_surrogate[rank],physical,valid,passed,duplicate,*candidate])
            print(f'Rank {rank+1:02d} | Surrogate={top_surrogate[rank]:.4f} | Physical={physical:.4f} | Valid={valid} | Pass={passed}')

        # save every trial (long run)
        columns=['Trial','Rank','Surrogate_FoM','Physical_FoM','Valid','Pass','Duplicate']+[f'T{i+1}' for i in range(24)]
        pd.DataFrame(candidate_results,columns=columns).to_csv(os.path.join(out_dir,'REINFORCE_physical_validation.csv'),index=False)
        pd.DataFrame(phys_log,columns=['Trial','Episode','Phys_call','Surrogate_FoM','Physical_FoM','Valid','Pass']
                     ).to_csv(os.path.join(out_dir,'REINFORCE_phys_log.csv'),index=False)
        pd.DataFrame(training,columns=['Trial','Episodes','N_phys','Train_pass_count','First_pass_call',
                                       'Train_best_physical','Stop_reason','Train_time_s']
                     ).to_csv(os.path.join(out_dir,'REINFORCE_training_validation.csv'),index=False)

    df=pd.DataFrame(training,columns=['Trial','Episodes','N_phys','Train_pass_count','First_pass_call',
                                      'Train_best_physical','Stop_reason','Train_time_s'])
    print('\nStop reasons:\n',df['Stop_reason'].value_counts().to_string())
    print(f"Trials with >=1 in-training pass: {(df['Train_pass_count']>0).sum()}/{n_trials}")
    print(f'Elapsed {time.time()-total_start:.1f}s | saved to {out_dir}')
