# -*- coding: utf-8 -*-
"""
Effect of surrogate accuracy on optimization success (SI, Fig. S9).

Runs the same protocol as run_GLOnet_passK.py with PBTL surrogate models trained on
197k, 276k, 355k and 394k (full) samples (Surrogate/4QW_total/model/surrogate_PBTL_*.pt).

Outputs (Result/pass_at_K/GLOnet_*_surrogate_quality): per-model folders + sweep_summary.csv

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
sys.path.append(os.path.join(path_gp,'Surrogate','4QW_total'))
 
import torch
import torch.nn as nn
import torch.optim as optim
from evaluation_physical import EVAL_7_7um_tot
from train_surrogate_PBTL import PBTL
from qcl_simulator import QCL
 
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
 
def to_scalar(x):
    return np.asarray(x).item()
 
class GLOnet(nn.Module):
    def __init__(self,dim_latent,xfile,t_gain=24,t_LW=24,t_wave=7.7,t_Ein=26,t_Eout=26):
        super(GLOnet,self).__init__()
        trained_model = torch.load(xfile,map_location=device,weights_only=False)
 
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
 
    def forward(self,x):
        y = nn.functional.leaky_relu(self.gen1(x))
        y = nn.functional.sigmoid(self.gen2(y))
        thick = y
        return thick,self.surrogate(thick)
 
    def surrogate(self,y):
        # frozen NN surrogate: normalized thickness (0-1) -> normalized outputs
        y1 = nn.functional.leaky_relu(self.fc11(y[:,:8]))
        y2 = nn.functional.leaky_relu(self.fc12(y[:,8:]))
        y = torch.hstack([y1,y2])
        y = nn.functional.leaky_relu(self.fc2(y))
        y = nn.functional.leaky_relu(self.fc3(y))
        y = nn.functional.leaky_relu(self.fc4(y))
        y = nn.functional.leaky_relu(self.fc5(y))
        y = nn.functional.leaky_relu(self.fc6(y))
        return y
 
    def train(self,model,optimizer,x,sig1,sig2,beta,Y_max,Y_min):
        thick,result = model(x)
 
        peak = result[:,0]*(Y_max[0]-Y_min[0])+Y_min[0]
        LW = result[:,1]*(Y_max[1]-Y_min[1])+Y_min[1]
        wave = result[:,2]*(Y_max[2]-Y_min[2])+Y_min[2]
        Ein = result[:,3]*(Y_max[3]-Y_min[3])+Y_min[3]
        Eout = result[:,4]*(Y_max[4]-Y_min[4])+Y_min[4]
 
        sqrt_2pi = torch.sqrt(torch.tensor(2*math.pi,device=wave.device,dtype=wave.dtype))
 
        peak_point = 1/(1+torch.exp(-0.5*(peak-self.t_gain)))
        LW_point = 1/(1+torch.exp(0.5*(LW-self.t_LW)))
        wave_point = (1/sqrt_2pi)*torch.exp(-0.5*(wave-self.t_wave)**2)*2.5
        Ein_point = 1/(1+torch.exp(0.5*(Ein-self.t_Ein)))
        Eout_point = 1/(1+torch.exp(0.5*(Eout-self.t_Eout)))
 
        total_point = peak_point+wave_point+Ein_point+Eout_point+LW_point
 
        loss_var = 0
        for ii in range(24):
            loss_var += torch.var(thick[:,ii])
 
        loss_perform = (
            -1/sig1*torch.exp(result[:,0]/sig1)
            +1/sig1*torch.exp(result[:,1]/sig1)
            +1/sig1*torch.exp(result[:,3]/sig1)
            +1/sig1*torch.exp(result[:,4]/sig1)
            +1-(1/(sig2*sqrt_2pi))*torch.exp(-0.5*(wave-self.t_wave)**2/sig2**2)*2.5
        )
 
        if beta is None:
            self.loss = loss_perform
        else:
            self.loss = loss_perform-beta*loss_var
 
        optimizer.zero_grad()
        self.loss.mean().backward()
        optimizer.step()
 
        return thick,total_point
 
#%% Main
if __name__=="__main__":
    # ---------------- experiment switches (keep identical to the GA script) ----------------
    TARGET=dict(t_gain=25,t_LW=25,t_wave=7.7,t_Ein=26,t_Eout=26)
    USE_PHYSICAL_GUIDANCE=True   # True: solver-validated protocol; False: surrogate-only (physical model used only for post-hoc top-K evaluation)
    N_PHYS=200
    PATIENCE_PHYS=100          # consecutive physical calls without a pass (None = off)
    PASS_TARGET=2
    SURR_GATE=4.8
    physical_threshold=4.8
    N_SAMPLE=64                # final pool = GA population = REINFORCE batch
 
    tag='physguided' if USE_PHYSICAL_GUIDANCE else 'surronly'
    out_root=os.path.join(path_c,'Result','pass_at_K',f'GLOnet_n{N_SAMPLE}_{tag}_surrogate_quality')
    os.makedirs(out_root,exist_ok=True)
 
    # surrogate models trained with different amounts of data (same files as the old SI sweep)
    # ratio = held-out fraction of the 394,237-sample dataset -> training size (1-ratio)*394,237 (SI Fig. S9)
    RATIOS=[0.5,0.3,0.1,1]   # 1 = full dataset
    SURROGATE_FILES={0.5:'surrogate_PBTL_197k.pt',0.3:'surrogate_PBTL_276k.pt',0.1:'surrogate_PBTL_355k.pt',1:'surrogate_PBTL_394k_full.pt'}
    def xfile_of(ratio): return path_gp+'/Surrogate/4QW_total/model/'+SURROGATE_FILES[ratio]
    sweep_rows=[]
 
    X_max=np.array([50.,24.,24.,60.,24.,60.,25.,50.,30.,50.,20.,40.,20.,40.,20.,40.,23.,33.,33.,33.,40.,35.,45.,35.])
    X_min=np.array([32.,11.,9.,45.,9.,42.,9.,35.,16.,36.,11.,31.,11.,31.,11.,30.,14.,24.,24.,24.,26.,21.,31.,21.])
    Y_max=[49.8,30.5,11.9,164.32,109.5]
    Y_min=[1.,6.73,5.13,0.88,1.304]
 
    epoch=500
    sig1,sig2=1,0.5   # sigma_1, sigma_2 of the FoM loss (Eq. 7 of the paper)
    n_batch=64
    dim_latent=64
    n_trials=50
    beta_list=[1.5,None]
 
    K_list=[1,4,8,16,32]
    max_K=max(K_list)
    n_sample=N_SAMPLE
 
    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
 
    def surrogate_fom(model,result):
        peak=result[:,0]*(Y_max[0]-Y_min[0])+Y_min[0]
        LW=result[:,1]*(Y_max[1]-Y_min[1])+Y_min[1]
        wave=result[:,2]*(Y_max[2]-Y_min[2])+Y_min[2]
        Ein=result[:,3]*(Y_max[3]-Y_min[3])+Y_min[3]
        Eout=result[:,4]*(Y_max[4]-Y_min[4])+Y_min[4]
        sqrt_2pi=torch.sqrt(torch.tensor(2*math.pi,device=result.device,dtype=result.dtype))
        peak_point=1/(1+torch.exp(-0.5*(peak-model.t_gain)))
        LW_point=1/(1+torch.exp(0.5*(LW-model.t_LW)))
        wave_point=(1/sqrt_2pi)*torch.exp(-0.5*(wave-model.t_wave)**2)*2.5
        Ein_point=1/(1+torch.exp(0.5*(Ein-model.t_Ein)))
        Eout_point=1/(1+torch.exp(0.5*(Eout-model.t_Eout)))
        return peak_point+LW_point+wave_point+Ein_point+Eout_point
 
    def score_rounded(model,structure_rounded):
        """surrogate FoM of an INTEGER-thickness structure (same as GA scores its integer population)"""
        with torch.no_grad():
            t=torch.tensor((structure_rounded-X_min)/(X_max-X_min),dtype=torch.float32,device=device)
            return surrogate_fom(model,model.surrogate(t)).detach().cpu().numpy().reshape(-1)
 
    def phys_eval(x):
        try:
            v,pk,lw,wv,ei,eo=EVAL_7_7um_tot(x,**TARGET)
            v=to_scalar(v)
            valid=int(np.isfinite(v) and v>0)
            return v,valid,int(valid and v>=physical_threshold),[to_scalar(pk),to_scalar(lw),to_scalar(wv),to_scalar(ei),to_scalar(eo)]
        except Exception as error:
            print(f'Physical solver failed: {error}')
            return np.nan,0,0,[np.nan]*5
 
    total_start=time.time()
 
    for ratio in RATIOS:
        xfile=xfile_of(ratio)
        out_dir=os.path.join(out_root,f'ratio_{ratio}'); os.makedirs(out_dir,exist_ok=True)
        print('\n'+'='*100); print(f'SURROGATE MODEL: {xfile}'); print('='*100)
        for beta in beta_list:
            learning_rate=1e-3 if beta==1.5 else 1e-2
            model_type='beta_GLOnet' if beta is not None else 'neural_adjoint_GLOnet'
            candidate_results=[]
            training_results=[]
            phys_log=[]
 
            print('\n'+'#'*100)
            print(f'START MODEL: {model_type} | beta={beta} | lr={learning_rate} | TARGET={TARGET}')
            print('#'*100)
 
            for trial in range(n_trials):
                print(f'\n{model_type} | Trial {trial+1}/{n_trials}')
                set_seed(2026+trial)
 
                model=GLOnet(dim_latent,xfile,**TARGET).to(device)
                for name,param in model.named_parameters():
                    if name.startswith('fc'): param.requires_grad=False
                optimizer=optim.Adam([p for p in model.parameters() if p.requires_grad],lr=learning_rate)
 
                checked=set()
                n_phys=0; n_pass=0; since_pass=0; first_pass_call=np.nan; best_phys=0.
                stop_reason='max_epoch'
                trial_start=time.time()
 
                # ============================================================
                # Training (surrogate only) + physical validation (stop rule only)
                # ============================================================
                for e in range(epoch):
                    initial_condition=(torch.rand([n_batch,dim_latent],device=device)-0.5)*2
                    structure,total_point=model.train(model,optimizer,initial_condition,
                                                      sig1=sig1,sig2=sig2,beta=beta,
                                                      Y_max=Y_max,Y_min=Y_min)
                    total_np=total_point.detach().cpu().numpy().reshape(-1)
 
                    if USE_PHYSICAL_GUIDANCE:
                        structure_np=np.round(structure.detach().cpu().numpy()*(X_max-X_min)+X_min)
                        surr_r=score_rounded(model,structure_np)
 
                        cand=None
                        for idx in np.argsort(surr_r)[::-1]:
                            key=tuple(structure_np[idx].astype(int))
                            if key not in checked:
                                checked.add(key); cand=structure_np[idx].copy(); s_val=float(surr_r[idx]); break
 
                        if cand is not None and s_val>=SURR_GATE:
                            v,valid,passed,_=phys_eval(cand)
                            n_phys+=1
                            if np.isfinite(v): best_phys=max(best_phys,v)
                            since_pass=0 if passed else since_pass+1
                            if passed:
                                n_pass+=1
                                if np.isnan(first_pass_call): first_pass_call=n_phys
                            phys_log.append([trial+1,e+1,n_phys,s_val,v,valid,passed])
 
                            if n_pass>=PASS_TARGET:
                                stop_reason='pass_target'; print(f'Pass target reached at epoch {e+1}'); break
                            if n_phys>=N_PHYS:
                                stop_reason='phys_budget'; break
                            if PATIENCE_PHYS is not None and since_pass>=PATIENCE_PHYS:
                                stop_reason='no_pass_patience'; break
 
                    if (e+1)%100==0:
                        print(f'Epoch {e+1:3d}/{epoch} | Max={total_np.max():.4f} | Mean={total_np.mean():.4f} | '
                              f'Physical best={best_phys:.4f} | calls={n_phys} | passes={n_pass}')
 
                training_time=time.time()-trial_start
                trained_epochs=e+1
                training_results.append([trial+1,trained_epochs,n_phys,n_pass,first_pass_call,best_phys,
                                         stop_reason,training_time])
                print(f'Training complete | Epoch={trained_epochs} | calls={n_phys} | passes={n_pass} | '
                      f'Stop={stop_reason} | {training_time:.2f} s')
 
                # ============================================================
                # Post-hoc: sample pool -> round -> rescore -> top-K -> physical (not counted in N_PHYS)
                # ============================================================
                set_seed(10000+trial)
                inference_start=time.time()
                with torch.no_grad():
                    initial_condition=(torch.rand([n_sample,dim_latent],device=device)-0.5)*2
                    thick,_=model(initial_condition)
                structure=np.round(thick.detach().cpu().numpy()*(X_max-X_min)+X_min)
                surr_fom=score_rounded(model,structure)
                inference_time=time.time()-inference_start
 
                elite_index=np.argsort(surr_fom)[::-1][:max_K]
                top_structure=structure[elite_index]
                top_surrogate_fom=surr_fom[elite_index]
                print(f'Generated candidates: {n_sample} | Top-{max_K} unique: {len(np.unique(top_structure,axis=0))}/{max_K}')
 
                for rank in range(max_K):
                    candidate=top_structure[rank].copy()
                    duplicate=int(rank>0 and any(np.array_equal(candidate,top_structure[j]) for j in range(rank)))
                    v,valid,passed,comp=phys_eval(candidate)
                    candidate_results.append([trial+1,rank+1,top_surrogate_fom[rank],v,valid,passed,duplicate,
                                              training_time,inference_time,*comp,*candidate])
                    print(f'Rank {rank+1:02d} | Surrogate={top_surrogate_fom[rank]:.4f} | Physical={v:.4f} | '
                          f'Valid={valid} | Pass={passed} | Duplicate={duplicate}')
 
            # ============================================================
            # Save
            # ============================================================
            candidate_columns=['Trial','Rank','Surrogate_FoM','Physical_FoM','Physically_valid','Pass','Duplicate',
                               'Training_time_s','Inference_time_s','Peak_point','LW_point','Wave_point',
                               'Ein_point','Eout_point']+[f'T{i+1}' for i in range(24)]
            df_candidate=pd.DataFrame(candidate_results,columns=candidate_columns)
 
            df_training=pd.DataFrame(training_results,columns=[
                'Trial','Trained_epochs','N_phys','Train_pass_count','First_pass_call',
                'Train_best_physical','Stop_reason','Training_time_s'])
            df_log=pd.DataFrame(phys_log,columns=['Trial','Epoch','Phys_call','Surrogate_FoM','Physical_FoM','Valid','Pass'])
 
            thickness_cols=[f'T{i+1}' for i in range(24)]
            rows=[]
            for K in K_list:
                for trial,g in df_candidate.groupby('Trial'):
                    g=g.sort_values('Rank').iloc[:K]
                    rows.append([trial,K,int(g['Pass'].any()),g['Pass'].mean(),
                                 g['Physically_valid'].mean(),len(g[thickness_cols].drop_duplicates())])
 
            df_passk=pd.DataFrame(rows,columns=['Trial','K','Pass_at_K','Pass_fraction','Valid_fraction','Unique_count'])
            df_summary=df_passk.groupby('K').agg(
                Pass_at_K=('Pass_at_K','mean'),Pass_SEM=('Pass_at_K','sem'),
                Pass_fraction=('Pass_fraction','mean'),Valid_fraction=('Valid_fraction','mean'),
                Unique_mean=('Unique_count','mean'),Unique_SEM=('Unique_count','sem')).reset_index()
            df_summary['Unique_fraction']=df_summary['Unique_mean']/df_summary['K']*100
 
            df_candidate.to_csv(os.path.join(out_dir,f'{model_type}_physical_validation.csv'),index=False)
            df_passk.to_csv(os.path.join(out_dir,f'{model_type}_passK_trials.csv'),index=False)
            df_summary.to_csv(os.path.join(out_dir,f'{model_type}_passK_summary.csv'),index=False)
            df_training.to_csv(os.path.join(out_dir,f'{model_type}_training_validation.csv'),index=False)
            df_log.to_csv(os.path.join(out_dir,f'{model_type}_phys_log.csv'),index=False)
            # ---- sweep summary rows (per surrogate / algorithm / K) ----
            train_succ=int((df_training['Train_pass_count']>0).sum())
            sweep_rows.append(dict(Ratio=ratio,Model=model_type,K=0,Pass_count=train_succ,
                                   N_trials=n_trials,Pass_pct=100*train_succ/n_trials,Valid_pct=np.nan))
            for K in K_list:
                
                sub=df_passk[df_passk.K==K]; k=int(sub.Pass_at_K.sum()); n=len(sub)
                sweep_rows.append(dict(Ratio=ratio,Model=model_type,K=K,Pass_count=k,N_trials=n,Pass_pct=100*k/n,
                                       Valid_pct=100*(sub.Valid_fraction>0).mean()))
            pd.DataFrame(sweep_rows).to_csv(os.path.join(out_root,'sweep_summary.csv'),index=False)
 
 
            print('\n'+'='*100)
            print(f'{model_type} FINAL SUMMARY')
            print('='*100)
            print(df_summary.to_string(index=False))
            print('\nStop reasons:\n',df_training['Stop_reason'].value_counts().to_string())
            print(f"Trials with >=1 in-training pass: {(df_training['Train_pass_count']>0).sum()}/{n_trials}")
 
        print('\n'+'#'*100)
        print(f'ALL EXPERIMENTS FINISHED | Total elapsed time: {time.time()-total_start:.2f} s | saved to {out_dir}')
        print('#'*100)
 