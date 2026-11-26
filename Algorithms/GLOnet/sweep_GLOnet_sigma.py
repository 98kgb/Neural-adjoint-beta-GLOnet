# -*- coding: utf-8 -*-
"""
Sensitivity of beta-GLOnet to the FoM-loss parameters sigma1 and sigma2 (SI, Fig. S4).

Full-factorial grid sigma1 = 0.8-1.2 x sigma2 = 0.40-0.60 (25 combinations), 20 trials each,
beta = 1.5, lr = 1e-3, batch 64, 500 epochs. Reports mean FoM, fraction of high-FoM (>= 4.8)
candidates, pairwise L1 distance and design-bound fraction, and the paired difference against
the baseline (sigma1, sigma2) = (1.0, 0.5).

Outputs (analysis/sigma_grid): sigma_grid_all_trials.csv, sigma_grid_summary.csv,
sigma_grid_paired_vs_baseline.csv, heatmap_*.csv

@author: Gibaek Kim
"""
import os,sys,math
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

path_c=os.path.dirname(os.path.abspath(__file__))
path_p=os.path.dirname(path_c)
path_gp=os.path.dirname(path_p)
path_ggp=os.path.dirname(path_gp)
sys.path.extend([path_c,path_p,path_ggp,os.path.join(path_gp,'Surrogate','4QW_total')])

from train_GLOnet import GLOnet
from evaluation_physical import EVAL_7_7um_tot
from train_surrogate_PBTL import PBTL

device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:',device)

# ============================================================
# Fixed settings
# ============================================================
X_max=np.array([50.,24.,24.,60.,24.,60.,25.,50.,30.,50.,20.,40.,20.,40.,20.,40.,23.,33.,33.,33.,40.,35.,45.,35.])
X_min=np.array([32.,11.,9.,45.,9.,42.,9.,35.,16.,36.,11.,31.,11.,31.,11.,30.,14.,24.,24.,24.,26.,21.,31.,21.])
Y_max=np.array([49.8,30.5,11.9,164.32,109.5])
Y_min=np.array([1.,6.73,5.13,0.88,1.304])

xfile=os.path.join(path_gp,'Surrogate','4QW_total','model','surrogate_PBTL_394k_full.pt')

EPOCH=500
LR=1e-3
BATCH=64
DIM_LATENT=64
BETA=1.5
N_TRIAL=20
N_FINAL=100
HIGH_FOM=4.8

SIG1_VALUES=[0.8,0.9,1.0,1.1,1.2]
SIG2_VALUES=[0.40,0.45,0.50,0.55,0.60]

save_dir=os.path.join(path_c,'analysis','sigma_grid')
os.makedirs(save_dir,exist_ok=True)

# ============================================================
# Canonical evaluation FoM
# ============================================================
def calc_outputs_and_fom(model,z):
    structure,result=model(z)
    peak=result[:,0]*(Y_max[0]-Y_min[0])+Y_min[0]
    LW=result[:,1]*(Y_max[1]-Y_min[1])+Y_min[1]
    wave=result[:,2]*(Y_max[2]-Y_min[2])+Y_min[2]
    Ein=result[:,3]*(Y_max[3]-Y_min[3])+Y_min[3]
    Eout=result[:,4]*(Y_max[4]-Y_min[4])+Y_min[4]
    sqrt2pi=torch.sqrt(torch.tensor(2*math.pi,device=result.device,dtype=result.dtype))
    peak_point=1/(1+torch.exp(-0.5*(peak-model.t_gain)))
    LW_point=1/(1+torch.exp(0.5*(LW-model.t_LW)))
    wave_point=(2.5/sqrt2pi)*torch.exp(-0.5*(wave-model.t_wave)**2)
    Ein_point=1/(1+torch.exp(0.5*(Ein-model.t_Ein)))
    Eout_point=1/(1+torch.exp(0.5*(Eout-model.t_Eout)))
    total_point=peak_point+LW_point+wave_point+Ein_point+Eout_point
    raw=torch.stack([peak,LW,wave,Ein,Eout],dim=1)
    return structure,result,raw,total_point

# ============================================================
# Training step
# NO objective-specific weights
# ============================================================
def train_step(model,optimizer,z,sig1,sig2,beta):
    thick,result=model(z)
    wave=result[:,2]*(Y_max[2]-Y_min[2])+Y_min[2]
    sqrt2pi=torch.sqrt(torch.tensor(2*math.pi,device=result.device,dtype=result.dtype))

    L_gain=-(1/sig1)*torch.exp(result[:,0]/sig1)
    L_LW=(1/sig1)*torch.exp(result[:,1]/sig1)
    L_wave=1-(1/(sig2*sqrt2pi))*torch.exp(-0.5*((wave-model.t_wave)/sig2)**2)*2.5
    L_Ein=(1/sig1)*torch.exp(result[:,3]/sig1)
    L_Eout=(1/sig1)*torch.exp(result[:,4]/sig1)

    loss_fom=L_gain+L_LW+L_wave+L_Ein+L_Eout
    loss_div=torch.var(thick,dim=0).sum()
    loss_total=loss_fom-beta*loss_div

    optimizer.zero_grad()
    loss_total.mean().backward()
    optimizer.step()

    return thick,loss_fom.detach(),loss_div.detach(),loss_total.detach()

def mean_pairwise_l1(x):
    x=np.asarray(x,dtype=float)
    if len(x)<2:
        return np.nan
    x=(x-X_min)/(X_max-X_min)
    d=np.abs(x[:,None,:]-x[None,:,:]).mean(axis=2)
    i,j=np.triu_indices(len(x),k=1)
    return float(d[i,j].mean())

# ============================================================
# Full factorial experiment
# 5 sigma1 x 5 sigma2 x 20 trials
# ============================================================
rows=[]

for trial in range(N_TRIAL):
    for sig1 in SIG1_VALUES:
        for sig2 in SIG2_VALUES:

            case_name=f'Sig1_{sig1:.2f}_Sig2_{sig2:.2f}'

            # same initialization for a given trial across all sigma combinations
            seed=2026+trial
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model=GLOnet(DIM_LATENT,xfile).to(device)

            # surrogate freeze
            for name,param in model.named_parameters():
                if name.startswith('fc'):
                    param.requires_grad=False

            optimizer=optim.Adam(
                filter(lambda p:p.requires_grad,model.parameters()),
                lr=LR
            )

            hist_best=[]
            hist_mean=[]
            hist_dd=[]
            hist_loss_fom=[]
            hist_loss_div=[]
            hist_loss_total=[]

            for e in range(EPOCH):
                z=(torch.rand([BATCH,DIM_LATENT],device=device)-0.5)*2

                thick,loss_fom,loss_div,loss_total=train_step(
                    model=model,
                    optimizer=optimizer,
                    z=z,
                    sig1=sig1,
                    sig2=sig2,
                    beta=BETA
                )

                with torch.no_grad():
                    _,_,_,fom=calc_outputs_and_fom(model,z)

                s=thick.detach().cpu().numpy()
                s=np.round(s*(X_max-X_min)+X_min)

                hist_best.append(float(fom.max().cpu()))
                hist_mean.append(float(fom.mean().cpu()))
                hist_dd.append(float(np.mean(s.std(axis=0))))
                hist_loss_fom.append(float(loss_fom.mean().cpu()))
                hist_loss_div.append(float(loss_div.cpu()))
                hist_loss_total.append(float(loss_total.mean().cpu()))

                if (e+1)%100==0:
                    print(
                        f'Trial {trial:02d} | sig1={sig1:.2f} | sig2={sig2:.2f} | '
                        f'Ep {e+1:03d} | Best={hist_best[-1]:.4f} | '
                        f'Mean={hist_mean[-1]:.4f} | DD={hist_dd[-1]:.3f}'
                    )

            # ====================================================
            # Final population
            # ====================================================
            eval_seed=10000+trial
            torch.manual_seed(eval_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(eval_seed)

            with torch.no_grad():
                z=(torch.rand([N_FINAL,DIM_LATENT],device=device)-0.5)*2
                structure,_,raw,fom=calc_outputs_and_fom(model,z)

            structure=structure.cpu().numpy()
            structure=np.round(structure*(X_max-X_min)+X_min)
            raw=raw.cpu().numpy()
            fom=fom.cpu().numpy()

            best_idx=int(np.argmax(fom))
            best_fom=float(fom[best_idx])
            mask=fom>=HIGH_FOM

            unique=np.unique(structure,axis=0)
            high_unique=np.unique(structure[mask],axis=0) if np.any(mask) else np.empty((0,24))

            dd=float(np.mean(structure.std(axis=0)))
            unique_fraction=len(unique)/N_FINAL*100
            high_fraction=float(mask.mean()*100)
            high_l1=mean_pairwise_l1(high_unique)

            boundary=np.logical_or(
                np.isclose(structure,X_min),
                np.isclose(structure,X_max)
            ).mean()*100

            rows.append({
                'trial':trial,
                'case':case_name,
                'sig1':sig1,
                'sig2':sig2,
                'best_FoM':best_fom,
                'mean_FoM':float(fom.mean()),
                'success_best_ge_4.8':int(best_fom>=HIGH_FOM),
                'DD':dd,
                'high_FoM_fraction':high_fraction,
                'unique_fraction':float(unique_fraction),
                'high_unique_n':len(high_unique),
                'high_pairwise_L1':high_l1,
                'boundary_fraction':float(boundary),
                'best_gain':float(raw[best_idx,0]),
                'best_LW':float(raw[best_idx,1]),
                'best_wave':float(raw[best_idx,2]),
                'best_wave_error':float(abs(raw[best_idx,2]-model.t_wave)),
                'best_Einj':float(raw[best_idx,3]),
                'best_Eext':float(raw[best_idx,4])
            })

            # ====================================================
            # Save individual condition
            # ====================================================
            case_dir=os.path.join(save_dir,case_name)
            os.makedirs(case_dir,exist_ok=True)

            np.save(os.path.join(case_dir,f'trial_{trial:02d}_structures.npy'),structure)
            np.save(os.path.join(case_dir,f'trial_{trial:02d}_raw_outputs.npy'),raw)
            np.save(os.path.join(case_dir,f'trial_{trial:02d}_fom.npy'),fom)

            pd.DataFrame({
                'epoch':np.arange(1,EPOCH+1),
                'Best_FoM':hist_best,
                'Mean_FoM':hist_mean,
                'DD':hist_dd,
                'L_FoM':hist_loss_fom,
                'L_Div':hist_loss_div,
                'L_total':hist_loss_total
            }).to_csv(
                os.path.join(case_dir,f'trial_{trial:02d}_history.csv'),
                index=False
            )

            # checkpoint
            pd.DataFrame(rows).to_csv(
                os.path.join(save_dir,'sigma_grid_all_trials.csv'),
                index=False
            )

# ============================================================
# Summary
# ============================================================
df=pd.DataFrame(rows)

metrics=[
    'best_FoM',
    'mean_FoM',
    'DD',
    'high_FoM_fraction',
    'unique_fraction',
    'high_unique_n',
    'high_pairwise_L1',
    'boundary_fraction',
    'best_gain',
    'best_LW',
    'best_wave',
    'best_wave_error',
    'best_Einj',
    'best_Eext'
]

summary=[]

for sig1 in SIG1_VALUES:
    for sig2 in SIG2_VALUES:
        d=df[(df['sig1']==sig1)&(df['sig2']==sig2)]

        row={
            'sig1':sig1,
            'sig2':sig2,
            'N':len(d),
            'success_n':int(d['success_best_ge_4.8'].sum()),
            'success_rate':float(d['success_best_ge_4.8'].mean()*100),
            'best_FoM_median':float(d['best_FoM'].median())
        }

        for m in metrics:
            x=d[m].dropna().values
            row[f'{m}_mean']=np.mean(x) if len(x)>0 else np.nan
            row[f'{m}_std']=np.std(x,ddof=1) if len(x)>1 else np.nan
            row[f'{m}_CI95']=(
                1.96*np.std(x,ddof=1)/np.sqrt(len(x))
                if len(x)>1 else np.nan
            )

        summary.append(row)

summary=pd.DataFrame(summary)
summary.to_csv(
    os.path.join(save_dir,'sigma_grid_summary.csv'),
    index=False
)

# ============================================================
# Paired vs baseline (sig1=1.0, sig2=0.5)
# ============================================================
paired_metrics=[
    'best_FoM',
    'mean_FoM',
    'high_FoM_fraction',
    'DD',
    'unique_fraction',
    'high_pairwise_L1',
    'boundary_fraction',
    'best_gain',
    'best_LW',
    'best_wave_error',
    'best_Einj',
    'best_Eext'
]

baseline=df[
    np.isclose(df['sig1'],1.0)&
    np.isclose(df['sig2'],0.5)
].set_index('trial')

paired_rows=[]

for sig1 in SIG1_VALUES:
    for sig2 in SIG2_VALUES:

        if np.isclose(sig1,1.0) and np.isclose(sig2,0.5):
            continue

        current=df[
            np.isclose(df['sig1'],sig1)&
            np.isclose(df['sig2'],sig2)
        ].set_index('trial')

        common=baseline.index.intersection(current.index)

        row={
            'sig1':sig1,
            'sig2':sig2,
            'N':len(common)
        }

        for m in paired_metrics:
            delta=(current.loc[common,m]-baseline.loc[common,m]).dropna()

            row[f'delta_{m}_mean']=delta.mean() if len(delta)>0 else np.nan
            row[f'delta_{m}_std']=delta.std(ddof=1) if len(delta)>1 else np.nan
            row[f'delta_{m}_CI95']=(
                1.96*delta.std(ddof=1)/np.sqrt(len(delta))
                if len(delta)>1 else np.nan
            )

        paired_rows.append(row)

paired=pd.DataFrame(paired_rows)
paired.to_csv(
    os.path.join(save_dir,'sigma_grid_paired_vs_baseline.csv'),
    index=False
)

# ============================================================
# Heatmap-ready pivot tables
# ============================================================
heatmap_metrics=[
    'mean_FoM',
    'high_FoM_fraction',
    'DD',
    'high_pairwise_L1',
    'boundary_fraction',
    'success_best_ge_4.8'
]

for metric in heatmap_metrics:
    pivot=df.pivot_table(
        index='sig1',
        columns='sig2',
        values=metric,
        aggfunc='mean'
    )

    pivot.to_csv(
        os.path.join(save_dir,f'heatmap_{metric}.csv')
    )

# ============================================================
# Console
# ============================================================
cols=[
    'sig1',
    'sig2',
    'success_rate',
    'best_FoM_median',
    'best_FoM_mean',
    'mean_FoM_mean',
    'high_FoM_fraction_mean',
    'DD_mean',
    'high_pairwise_L1_mean',
    'boundary_fraction_mean'
]

print('\n================ SIGMA GRID SUMMARY ================\n')
print(summary[cols].to_string(index=False))

print('\nSaved:')
print(os.path.join(save_dir,'sigma_grid_all_trials.csv'))
print(os.path.join(save_dir,'sigma_grid_summary.csv'))
print(os.path.join(save_dir,'sigma_grid_paired_vs_baseline.csv'))