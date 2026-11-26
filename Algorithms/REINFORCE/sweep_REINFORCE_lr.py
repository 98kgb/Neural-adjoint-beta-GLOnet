# -*- coding: utf-8 -*-
"""
REINFORCE learning-rate study (SI): lr = 3e-4 to 1e-2 with the main-comparison settings
(batch 64, latent dim 64, entropy weight 1e-5, 500 episodes), 5 trials each.
Reports best FoM during training and best FoM / uniqueness of the final 64-sample portfolio.

Outputs (analysis/REIN_b64): REIN_b64_trials.csv, REIN_b64_summary.csv

@author: Gibaek Kim
"""
import os, sys, time
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
path_c = os.path.dirname(os.path.abspath(__file__)); path_p = os.path.dirname(path_c); path_gp = os.path.dirname(path_p)
sys.path += [path_c, path_p, os.path.join(path_gp, 'Surrogate', '4QW_total')]

import torch, torch.optim as optim, numpy as np, pandas as pd
import train_REINFORCE as RT
from train_REINFORCE import Policy, GRUAgent
from evaluation_surrogate import EVAL_7_7um_total

from train_surrogate_PBTL import PBTL

device = "cpu"          # the policy network is small, so CPU is faster than MPS (also sets the module-level device used by learn())
RT.device = device

xfile = path_gp + '/Surrogate/4QW_total/model/surrogate_PBTL_394k_full.pt'
X_max = np.array([50.,24.,24.,60.,24.,60.,25.,50.,30.,50.,20.,40.,20.,40.,20.,40.,23.,33.,33.,33.,40.,35.,45.,35.])
X_min = np.array([32.,11.,9.,45.,9.,42.,9.,35.,16.,36.,11.,31.,11.,31.,11.,30.,14.,24.,24.,24.,26.,21.,31.,21.])
Y_max = [49.8, 30.5, 11.9, 164.32, 109.5]; Y_min = [1., 6.73, 5.13, 0.88, 1.304]

TARGET = dict(t_gain=25, t_LW=25, t_wave=7.7, t_Ein=26, t_Eout=26)
EPISODES, N_BATCH, LATENT, BETA = 500, 64, 64, 1e-5
LRS, N_TRIALS = [3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2], 5          # 5 trials per learning rate
state_size = 24; action_size = (X_max - X_min).astype(int)
out_dir = os.path.join(path_c, 'analysis', 'REIN_b64'); os.makedirs(out_dir, exist_ok=True)

def fast_act(agent, z):
    """Sample from the same categorical distribution as agent.act, using one multinomial draw per layer (faster)."""
    with torch.no_grad():
        pi = agent.model(z.reshape([1, z.shape[0], z.shape[1]]))
        return np.stack([torch.multinomial(p, 1).squeeze(1).cpu().numpy() for p in pi], axis=1).astype(float)

def evaluate(s):
    return np.asarray(EVAL_7_7um_total(s, X_max, X_min, Y_max=Y_max, Y_min=Y_min, xfile=xfile, **TARGET)[0]).ravel()

def ufrac(s):
    s = np.round(np.asarray(s)).astype(int); return len({tuple(r) for r in s}) / len(s)

def sample(agent, n):
    S = []
    for _ in range(n // N_BATCH):
        a = fast_act(agent, torch.normal(0, 1, size=(N_BATCH, LATENT)).to(device)); S.append(X_min[None, :] + a)
    S = np.vstack(S); return S, evaluate(S)

rows = []
for lr in LRS:
    for idx in range(N_TRIALS):
        torch.manual_seed(idx); np.random.seed(idx)
        model = Policy(state_size, action_size, LATENT).to(device)
        agent = GRUAgent(N_BATCH, state_size, action_size, model, optim.Adam(model.parameters(), lr=lr))
        t0 = time.time(); hist_best = []
        for e in range(EPISODES):
            z = torch.normal(0, 1, size=(N_BATCH, LATENT)).to(device)
            actions = fast_act(agent, z); s = X_min[None, :] + actions; f = evaluate(s)
            agent.learn(z, actions, f / 25, BETA)
            hist_best.append(f.max())
            if (e + 1) % 50 == 0:
                print(f'lr={lr} trial={idx} ep={e+1}/{EPISODES} max={f.max():.3f} mean={f.mean():.3f} {time.time()-t0:.0f}s', flush=True)

        row = dict(lr=lr, trial=idx, best_hist=max(hist_best), sec=time.time() - t0)
        # (A) as in the paper: 64-sample portfolio from the trained policy
        s64, f64 = sample(agent, 64); o = np.argsort(-f64)
        row.update(best_final64=f64.max(), mean_final64=f64.mean(), uniq_all64=ufrac(s64))
        for K in [8, 16, 32]:
            row[f'A_uniq_top{K}'] = ufrac(s64[o[:K]])
        # (B) alternative: top 32 of 1024 samples
        s1k, f1k = sample(agent, 1024); o = np.argsort(-f1k)
        row.update(best_1024=f1k.max())
        for K in [8, 16, 32]:
            row[f'B_uniq_top{K}'] = ufrac(s1k[o[:K]])
        np.savetxt(os.path.join(out_dir, f'final64_lr{lr}_t{idx}.csv'), s64, fmt='%d', delimiter=',')
        rows.append(row); print(row, flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'REIN_b64_trials.csv'), index=False)   # saved after every trial

summ = pd.DataFrame(rows).drop(columns=['trial']).groupby('lr').agg(['mean', 'std'])
summ.to_csv(os.path.join(out_dir, 'REIN_b64_summary.csv')); print(summ.to_string())
