# -*- coding: utf-8 -*-
"""
GA mutation-rate study (SI, Fig. S6).

Surrogate-assisted GA with population 64 and 500 generations, mutation rate 0.04-0.5,
20 trials each. Records the best surrogate FoM and the fraction of unique structures.

Outputs (analysis/GA_mutation_sweep): GA_mutation_trials.csv, GA_mutation_summary.csv, final populations

@author: Gibaek Kim
"""
import os
import sys

path_c = os.path.dirname(os.path.abspath(__file__))
path_p = os.path.dirname(path_c)
path_gp = os.path.dirname(path_p)

sys.path.append(path_gp)
sys.path.append(path_gp+'/Algorithms')
sys.path.append(path_gp+'/Algorithms/GA')
sys.path.append(path_gp+'/Surrogate/4QW_total/')

import time
import numpy as np
import pandas as pd
from GA_class import GA
from evaluation_surrogate import EVAL_7_7um_total as EVAL
from evaluation_physical import EVAL_7_7um_tot
from train_surrogate_PBTL import PBTL
from train_surrogate_GTL import GTL
from train_surrogate_total_baseline import TP

from tqdm import tqdm
from qcl_simulator import QCL
import matplotlib.pyplot as plt

from GA_class import GA
from evaluation_surrogate import EVAL_7_7um_total as EVAL


xfile = path_gp + '/Surrogate/4QW_total/model/surrogate_PBTL_394k_full.pt'
generation, n_particle, n_trials = 500, 64, 20
mutations = [0.04, 0.1, 0.2, 0.3, 0.4, 0.5]   # mutation rates examined in SI Fig. S6
out_dir = os.path.join(path_c, 'analysis', 'GA_mutation_sweep')
os.makedirs(out_dir, exist_ok=True)

X_max = np.array([50.,24.,24.,60.,24.,60.,25.,50.,30.,50.,20.,40.,20.,40.,20.,40.,23.,33.,33.,33.,40.,35.,45.,35.])
X_min = np.array([32.,11.,9.,45.,9.,42.,9.,35.,16.,36.,11.,31.,11.,31.,11.,30.,14.,24.,24.,24.,26.,21.,31.,21.])
Y_max = [49.8, 30.5, 11.9, 164.32, 109.5]
Y_min = [1., 6.73, 5.13, 0.88, 1.304]
TARGET = dict(t_gain=25, t_lw=25, t_wave=7.7, t_Ein=26, t_Eout=26)

def unique_frac(s):
    s = np.round(np.asarray(s)).astype(int)
    return len({tuple(r) for r in s}) / len(s)

rows = []
for mutation in mutations:
    for idx in range(n_trials):
        np.random.seed(1000 * idx + 7)
        X = np.zeros([n_particle, 24])
        for jj in range(24):
            X[:, jj] = np.random.randint(X_min[jj], X_max[jj] + 1, size=[n_particle])
        t0 = time.time()
        ga = GA(X, X_max=X_max, X_min=X_min, Y_max=Y_max, Y_min=Y_min, EVAL=EVAL,
                xfile=xfile, DNN=True, iteration=generation, n_particles=n_particle,
                mutation=mutation, **TARGET)
        ga.iterate()

        point = np.asarray(ga.point)
        row = dict(mutation=mutation, trial=idx, best_fom=float(point.max()), sec=time.time() - t0)

        pop = fit = None   # final population and its surrogate fitness
        for name in ['X', 'population', 'str_var']:
            if hasattr(ga, name):
                c = np.asarray(getattr(ga, name))
                if c.ndim == 2 and c.shape[1] == 24:
                    pop = c[-n_particle:]; break
        for name in ['fitness', 'FoM', 'fom', 'score', 'Y']:
            if hasattr(ga, name):
                c = np.asarray(getattr(ga, name)).ravel()
                if len(c) >= n_particle:
                    fit = c[-n_particle:]; break
        if pop is not None:
            row['uniq_all64'] = unique_frac(pop)
            if fit is not None and len(fit) == len(pop):
                order = np.argsort(-fit)
                for K in [8, 16, 32]:
                    row[f'uniq_top{K}'] = unique_frac(pop[order[:K]])
                    row[f'meanfom_top{K}'] = float(fit[order[:K]].mean())
            np.savetxt(os.path.join(out_dir, f'pop_m{mutation}_t{idx}.csv'), pop, fmt='%d', delimiter=',')
        rows.append(row)
    print('mutation', mutation, 'done')

df = pd.DataFrame(rows)
df.to_csv(os.path.join(out_dir, 'GA_mutation_trials.csv'), index=False)
summ = df.drop(columns=['trial']).groupby('mutation').agg(['mean', 'std'])
summ.to_csv(os.path.join(out_dir, 'GA_mutation_summary.csv'))
print(summ.to_string())