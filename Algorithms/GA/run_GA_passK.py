# -*- coding: utf-8 -*-
"""
GA results of Section 3.2 (solver-validated success rate, Pass@K, distinct high-FoM designs).

Surrogate-assisted GA (population 64, mutation rate 0.1, 500 generations, 50 trials) with the
solver-validated protocol shared by all algorithms:
  * the optimizer only sees the SURROGATE FoM (the physical FoM never enters the fitness)
  * every generation, the best not-yet-checked candidate with surrogate FoM >= 4.8 is evaluated
    once with the physical model
  * a trial stops after 2 physical passes, 200 solver calls, or 100 consecutive calls without a pass
  * afterwards the top-K (K = 1, 4, 8, 16, 32) of the final population are evaluated physically
USE_PHYSICAL_GUIDANCE = False gives the surrogate-only run.

Outputs (Result/pass_at_K/GA_*): GA_physical_validation.csv, GA_phys_log.csv, GA_training_validation.csv

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

Begin = time.time()
 
xfile = path_gp + '/Surrogate/4QW_total/model/surrogate_PBTL_394k_full.pt'
 
X_max = np.array([50., 24., 24., 60., 24., 60., 25., 50., 30., 50., 20., 40., 20.,
       40., 20., 40., 23., 33., 33., 33., 40., 35., 45., 35.])
X_min = np.array([32., 11.,  9., 45.,  9., 42.,  9., 35., 16., 36., 11., 31., 11.,
       31., 11., 30., 14., 24., 24., 24., 26., 21., 31., 21.])
Y_max = [49.8 ,  30.5 ,  11.9 , 164.32, 109.5]
Y_min = [1.   , 6.73 , 5.13 , 0.88 , 1.304]
 
# ---------------- experiment switches ----------------
TARGET = dict(t_gain=25, t_LW=25, t_wave=7.7, t_Ein=26, t_Eout=26)   # must be identical for all algorithms
USE_PHYSICAL_GUIDANCE = True      # False -> plain 500 generations (surrogate only)
N_PHYS = 200                      # max physical calls per trial (same for all algorithms)
PATIENCE_PHYS = 100                # stop if this many consecutive physical calls give no pass (None = off)
PASS_TARGET = 2                   # stop after this many physical passes (same as GLOnet run)
SURR_GATE = 4.8                   # evaluate physically only if surrogate best >= this
physical_threshold = 4.8
 
generation = 500
n_particle = 64
mutation = 0.1
n_trials = 50
K_list = [1, 4, 8, 16, 32]
max_K = max(K_list)
 
tag = 'physguided' if USE_PHYSICAL_GUIDANCE else 'surronly'
out_dir = os.path.join(path_c, 'Result', 'pass_at_K', f'GA_mu{mutation}_{tag}_nphys{N_PHYS}')
os.makedirs(out_dir, exist_ok=True)
 
def set_seed(seed):
    np.random.seed(seed)
 
def tg(d):   # kwargs for EVAL (ANN) / EVAL_7_7um_tot (same names)
    return dict(t_gain=d['t_gain'], t_LW=d['t_LW'], t_wave=d['t_wave'], t_Ein=d['t_Ein'], t_Eout=d['t_Eout'])
 
def phys_eval(x):
    try:
        p, *_ = EVAL_7_7um_tot(x, **tg(TARGET))
        p = np.asarray(p).item()
        valid = int(np.isfinite(p) and p > 0)
        return p, valid, int(valid and p >= physical_threshold)
    except Exception:
        return np.nan, 0, 0
 
candidate_results = []
phys_log = []
training = []
 
for trial in range(n_trials):
    print('\n' + '=' * 100)
    print(f'GA | Trial {trial+1}/{n_trials}')
    print('=' * 100)
 
    set_seed(20261122 + trial)
 
    X = np.zeros([n_particle, 24])
    for jj in range(24):
        X[:, jj] = np.random.randint(X_min[jj], X_max[jj] + 1, size=n_particle)
 
    ga = GA(X, X_max=X_max, X_min=X_min, Y_max=Y_max, Y_min=Y_min,
            t_gain=TARGET['t_gain'], t_lw=TARGET['t_LW'], t_wave=TARGET['t_wave'],
            t_Ein=TARGET['t_Ein'], t_Eout=TARGET['t_Eout'],
            EVAL=EVAL, xfile=xfile, DNN=True, iteration=generation,
            n_particles=n_particle, mutation=mutation)
 
    st = dict(n_phys=0, n_pass=0, since_pass=0, first_pass_call=np.nan, best=0.0, stop='max_generation', checked=set())
    t0 = time.time()
 
    def callback(gen, Xcur, surr):
        """one physical call on the best unchecked candidate of the CURRENT (already surrogate-evaluated) population"""
        if not USE_PHYSICAL_GUIDANCE:
            return False
        order = np.argsort(surr)[::-1]
        cand = None
        for idx in order:
            key = tuple(Xcur[idx].astype(int))
            if key not in st['checked']:
                st['checked'].add(key)
                cand, s_val = Xcur[idx].copy(), float(surr[idx])
                break
        if cand is None or s_val < SURR_GATE:
            return False
        p, valid, passed = phys_eval(cand)
        st['n_phys'] += 1
        st['best'] = max(st['best'], p) if np.isfinite(p) else st['best']
        st['since_pass'] = 0 if passed else st['since_pass'] + 1
        if passed:
            st['n_pass'] += 1
            if np.isnan(st['first_pass_call']):
                st['first_pass_call'] = st['n_phys']
        phys_log.append([trial + 1, gen + 1, st['n_phys'], s_val, p, valid, passed])
        if st['n_pass'] >= PASS_TARGET:
            st['stop'] = 'pass_target'
            return True
        if st['n_phys'] >= N_PHYS:
            st['stop'] = 'phys_budget'
            return True
        if PATIENCE_PHYS is not None and st['since_pass'] >= PATIENCE_PHYS:
            st['stop'] = 'no_pass_patience'
            return True
        return False
 
    ga.iterate(callback=callback)
    train_time = time.time() - t0
    n_done = getattr(ga, 'n_done', generation)
    training.append([trial + 1, n_done, st['n_phys'], st['n_pass'], st['first_pass_call'], st['best'], st['stop'], train_time])
    print(f"Stop={st['stop']} | gen={n_done} | phys calls={st['n_phys']} | in-training passes={st['n_pass']}")
 
    # ================= Final population (evaluated population at the stop point) =================
    # NOTE: after a normal 500-generation run ga.X is the unevaluated offspring; after an early stop it is the evaluated one.
    final_structure = np.asarray(ga.X).copy()
 
    surrogate_fom, *_ = EVAL(final_structure, X_max, X_min, Y_max=Y_max, Y_min=Y_min, xfile=xfile, **tg(TARGET))
    surrogate_fom = np.asarray(surrogate_fom).reshape(-1)
    elite_index = np.argsort(surrogate_fom)[::-1][:max_K]
    top_structure = final_structure[elite_index]
    top_surrogate = surrogate_fom[elite_index]
    print(f'Top-{max_K} unique: {len(np.unique(top_structure, axis=0))}/{max_K}')
 
    # ================= Post-hoc physical validation (not counted in N_PHYS) =================
    for rank in range(max_K):
        candidate = top_structure[rank].copy()
        duplicate = int(rank > 0 and any(np.array_equal(candidate, top_structure[j]) for j in range(rank)))
        physical, valid, passed = phys_eval(candidate)
        candidate_results.append([trial + 1, rank + 1, top_surrogate[rank], physical, valid, passed, duplicate, *candidate])
        print(f'Rank {rank+1:02d} | Surrogate={top_surrogate[rank]:.4f} | Physical={physical:.4f} | Valid={valid} | Pass={passed}')
 
columns = ['Trial', 'Rank', 'Surrogate_FoM', 'Physical_FoM', 'Valid', 'Pass', 'Duplicate'] + [f'T{i+1}' for i in range(24)]
pd.DataFrame(candidate_results, columns=columns).to_csv(os.path.join(out_dir, 'GA_physical_validation.csv'), index=False)
pd.DataFrame(phys_log, columns=['Trial', 'Generation', 'Phys_call', 'Surrogate_FoM', 'Physical_FoM', 'Valid', 'Pass']
             ).to_csv(os.path.join(out_dir, 'GA_phys_log.csv'), index=False)
pd.DataFrame(training, columns=['Trial', 'Generations', 'N_phys', 'Train_pass_count', 'First_pass_call',
                                'Train_best_physical', 'Stop_reason', 'Train_time_s']
             ).to_csv(os.path.join(out_dir, 'GA_training_validation.csv'), index=False)
 
df = pd.DataFrame(training, columns=['Trial', 'Generations', 'N_phys', 'Train_pass_count', 'First_pass_call',
                                     'Train_best_physical', 'Stop_reason', 'Train_time_s'])
print('\nStop reasons:\n', df['Stop_reason'].value_counts().to_string())
print(f"Trials with >=1 in-training pass: {(df['Train_pass_count']>0).sum()}/{n_trials}")
print(f'Elapsed {time.time()-Begin:.1f}s | saved to {out_dir}')
 