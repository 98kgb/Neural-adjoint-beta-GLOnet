# -*- coding: utf-8 -*-
"""
Physical figure of merit (FoM) of a 24-layer QCL structure, computed with the numerical
simulator (qcl_simulator.QCL). Implements Eqs. (1)-(6) of the paper:
sigmoid terms for G43, linewidth, E_inj, E_ext (slope 0.5) and a Gaussian term for the
lasing wavelength (sigma = 0.1 um). Structures without a valid level alignment get FoM = 0.

Used by all comparison scripts to decide whether a candidate physically passes (FoM >= 4.8).
Returns: (FoM, f_gain, f_linewidth, f_wavelength, f_Einj, f_Eext)

@author: Gibaek Kim
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.append(current_dir)
sys.path.append(parent_dir)

import math
import numpy as np
from qcl_simulator import QCL as QCL_7_7_tot

def Gaussian(x, mean, sig):
    
    y = (1/(sig*math.sqrt(2*np.pi)))*np.exp(-(1/2)*(x-mean)**2/sig**2)
    
    return y

def Dirac(x, c1, c2):
    
    y = 1/(1+np.exp(c1*(x-c2)))
    
    return y

#%% objective function

# 4QW QCL objective function
def EVAL_7_7um_tot(thick, t_gain=24, t_LW=24, t_wave=7.7, t_Ein=26, t_Eout=26):
    
    qcl = QCL_7_7_tot(np.hstack([thick]), LW_cal=True)
    qcl.Potential()
    qcl.EVMAT()
    qcl.CBEIGEN()
    if qcl.st3 != 0:
        qcl.Dipole()
        if qcl.E43 > qcl.ephonon and qcl.E32 > qcl.ephonon and qcl.E21 > qcl.ephonon:
            
            qcl.Scatter_rate()
            lamda, g43 = qcl.gain()
            
            
            peak = max(g43)
            wave = lamda[np.argmax(g43)]
            LW = qcl.LW*1000
            Ein = abs(qcl.Ein0)*1000
            Eout = abs(qcl.Eout)*1000
            
            peak_point =  Dirac(peak, -0.5, t_gain)
            LW_point = Dirac(LW, 0.5, t_LW)
            wave_point = Gaussian(wave, t_wave, 0.1)/4
            Ein_point = Dirac(Ein, 0.5, t_Ein)
            Eout_point = Dirac(Eout, 0.5, t_Eout)
            
            evaluation_value = peak_point+LW_point+wave_point+Ein_point+Eout_point
            
            
        else:
            peak_point =  0
            LW_point = 0
            wave_point = 0
            Ein_point = 0
            Eout_point = 0
            
            evaluation_value = 0
    else:
        peak_point =  0
        LW_point = 0
        wave_point = 0
        Ein_point = 0
        Eout_point = 0
        
        evaluation_value = 0
    
    return evaluation_value, peak_point, LW_point, wave_point, Ein_point, Eout_point
