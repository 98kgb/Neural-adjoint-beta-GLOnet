# -*- coding: utf-8 -*-
"""
Physical QCL simulator (in-house numerical model) for the 24-layer, 4-quantum-well 7.7 um QCL.

Solves the conduction-band states of 1.5 periods of the superlattice with an effective two-band
model (non-parabolicity included), then computes dipole matrix elements, LO-phonon scattering
rates, linewidth and the gain spectrum. Used to generate the training data and as the
"physical model" that validates candidate designs (via Algorithms/evaluation_physical.py).

Usage:  qcl = QCL(thicknesses_24, LW_cal=True); qcl.Potential(); qcl.EVMAT(); qcl.CBEIGEN();
        qcl.Dipole(); qcl.Scatter_rate(); wavelength, gain = qcl.gain()

@author: Gibaek Kim
"""
import numpy as np
import math

class QCL:
    def __init__(self, thickness, field = -33e3, T = 300, absorption = False, LW_cal = True):
        
        self.LW_cal = LW_cal
        self.absorption = absorption 
        self.aung = 1.0e-10
        self.ev = 1.60219e-19
        self.T = T
        self.k = 8.617e-5 #Boltzman const, ev/K
        self.delz = 1.0
        self.thick = thickness
        self.hbar = 1.0546e-34 # J.s
        self.field = field
        self.mo = 9.1095e-31
        self.m1 = 0.043*self.mo # well
        self.m2 = 0.072*self.mo # Barrier 0.072
        self.mt = 0.0392*self.mo
        # Kane Energy
        self.kane1 = 18.4*self.ev #17.82
        self.kane2 = 18.4*self.ev
        
        # composition
        self.x = 0.47     # GaAs mole fraction
        
        epho_GA = 36.1e-3
        epho_IA = 29e-3
        
        # phonon energy
        self.ephonon = epho_IA*self.x+epho_GA*(1-self.x)
        
        # Starting from barrier
        
        aw1 = thickness[0]
        aw2 = thickness[1]
        aw3 = thickness[2]
        aw4 = thickness[3]
        aw5 = thickness[4]
        aw6 = thickness[5]
        aw7 = thickness[6]
        aw8 = thickness[7]
        
        
        self.awmat=[aw1, aw2, aw3, aw4, aw5, aw6, aw7, aw8]
        
        iw1=thickness[8]
        iw2=thickness[9]
        iw3=thickness[10]
        iw4=thickness[11]
        iw5=thickness[12]
        iw6=thickness[13]
        iw7=thickness[14]
        iw8=thickness[15]
        iw9=thickness[16]
        iw10=thickness[17]
        iw11=thickness[18]
        iw12=thickness[19]
        iw13=thickness[20]
        iw14=thickness[21]
        iw15=thickness[22]
        iw16=thickness[23]
        
        
        self.iwmat =[iw1, iw2, iw3, iw4, iw5, iw6, iw7, iw8, iw9, iw10, iw11, iw12, iw13, iw14, iw15, iw16]
        
        self.twmat = np.concatenate((self.iwmat, self.awmat, self.iwmat, [aw1]))
        self.nd = int(sum(self.twmat))
        self.npx = self.nd+1
        self.nppx = self.nd+2
        
        # for test validation of the structure
        self.min_active = np.sum(self.iwmat)
        self.max_active = self.min_active + np.sum(self.awmat)
        self.mid_inj = self.max_active + np.sum(self.iwmat)/2
        
        self.aw_upper = np.sum(self.awmat[:3])
        self.aw_tot = np.sum(self.awmat)+iw1
        
    def Potential(self):
        
        Ec = np.zeros(self.nppx)
        Ev = np.zeros(self.nppx)

        Efield = np.zeros(self.nppx)
        position = np.zeros(self.nppx)

        self.gap0 = 0.51*self.ev
        self.gap1 = self.kane1*self.m1/self.mo
        self.gap2 = self.kane2*self.m2/self.mo
        
        self.iw = np.zeros(len(self.twmat))
        
        for ii in range(0, len(self.twmat)):
            self.iw[ii] = sum(self.twmat[0:ii+1])

        
        position = np.zeros(self.nppx)
        
        for ii in range(0,self.nppx):
            position[ii] = ii*self.delz
        
        self.pposition = position[1:(self.npx)]
        self.ppposition = np.concatenate((self.pposition.T,self.pposition.T), axis =0)
        
        for ii in range(0, self.nppx):
            position[ii] = (ii)*self.delz
            if (ii<self.iw[0]):
                Efield[ii] = self.field*100
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2;
                
            elif (ii<self.iw[1]):
                Efield[ii] = self.field*100
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap1
                
            elif (ii<self.iw[2]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2;   
                
            elif (ii<self.iw[3]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap1;    
                
            elif (ii<self.iw[4]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
                
            elif (ii<self.iw[5]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap1; 
                
            elif (ii<self.iw[6]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
                
            elif (ii<self.iw[7]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap1; 
                
            elif (ii<self.iw[8]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
                
            elif (ii<self.iw[9]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap1; 
                
            elif (ii<self.iw[10]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
                
            elif (ii<self.iw[11]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap1; 
                
            elif (ii<self.iw[12]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
                
            elif (ii<self.iw[13]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev
                Ev[ii] = Ec[ii]-self.gap1 
                
            elif (ii<self.iw[14]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
                
            elif (ii<self.iw[15]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap1
                
            elif (ii<self.iw[16]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
                
            elif (ii<self.iw[17]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap1
                
            elif (ii<self.iw[18]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
                
            elif (ii<self.iw[19]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev
                Ev[ii] = Ec[ii]-self.gap1
                
            elif (ii<self.iw[20]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
                
            elif (ii<self.iw[21]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap1
                
            elif (ii<self.iw[22]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
                
            elif (ii<self.iw[23]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev
                Ev[ii] = Ec[ii]-self.gap1
            elif (ii<self.iw[24]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
                
            elif (ii<self.iw[25]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap1
                
            elif (ii<self.iw[26]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
                
            elif (ii<self.iw[27]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap1
                
            elif (ii<self.iw[28]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
                
            elif (ii<self.iw[29]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev
                Ev[ii] = Ec[ii]-self.gap1
                
            elif (ii<self.iw[30]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
                
            elif (ii<self.iw[31]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap1
                
            elif (ii<self.iw[32]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
                
            elif (ii<self.iw[33]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev
                Ev[ii] = Ec[ii]-self.gap1
                
            elif (ii<self.iw[34]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
                
            elif (ii<self.iw[35]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev
                Ev[ii] = Ec[ii]-self.gap1
            elif (ii<self.iw[36]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
                
            elif (ii<self.iw[37]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap1
                
            elif (ii<self.iw[38]):
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
                
            elif (ii<self.iw[39]):
                Efield[ii] = self.field*100;
                Ec[ii] = Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap1
            else:
                Efield[ii] = self.field*100;
                Ec[ii] = self.gap0+Efield[ii]*position[ii]*self.aung*self.ev;
                Ev[ii] = Ec[ii]-self.gap2
            
            self.pos = self.pposition
            self.Ec = Ec
            self.Ev = Ev
            
    def EVMAT(self):
        
        # Potential Energy
        Ep = self.kane1*np.ones(self.nppx)
        ratio1 = self.hbar**2/(2*self.mo)
        Pcv = np.sqrt(ratio1*Ep)
        
        MA1 = np.zeros([self.nd,self.nd]); MA2 = np.zeros([self.nd,self.nd])
        MA3 = np.zeros([self.nd,self.nd]); MA4 = np.zeros([self.nd,self.nd])
        
        ## Potential Part
        
        for ii in range(0, self.nd):
            MA1[ii, ii] = MA1[ii, ii]+self.Ec[ii+1]
            MA4[ii, ii] = MA4[ii, ii]+(self.Ev[ii+1]+self.Ev[ii+2])/2;             
        
        ## kz part
        for ii in range(0,self.nd):
            MA2[ii, ii] = MA2[ii, ii]+1/self.aung*Pcv[ii+1] # Backward kt-indep
            MA3[ii, ii] = MA3[ii, ii]+1/self.aung*Pcv[ii+1] # Forward kt-indep       
        
        
        for ii in range(0,self.nd-1):
            MA2[ii+1, ii] = MA2[ii+1, ii]-1/self.aung*(Pcv[ii+2]+Pcv[ii+1])/2 # Backward kt-indep
            MA3[ii, ii+1] = MA3[ii, ii+1]-1/self.aung*(Pcv[ii+2]+Pcv[ii+1])/2 # Forward kt-indep    
        
        M1 =  np.concatenate([MA1,MA2], axis = 1)
        M2 =  np.concatenate([MA3,MA4], axis = 1)
        
        self.MA =  np.concatenate([M1,M2], axis = 0)

    def CBEIGEN(self):
        
        ncb = 20

        self.CB_potential = self.Ec[1:self.npx]/self.ev
        self.VB_potential = self.Ev[1:self.npx]/self.ev
        
        [d, v] = np.linalg.eigh(self.MA)
        Eigen_energy = np.zeros(len(self.MA))
        
        for ii in range(0,len(self.MA)):
            Eigen_energy[ii] = d[ii]/self.ev;
        
        
        self.energy = Eigen_energy[self.nd:2*self.nd]
        
        allwave = np.zeros((len(self.ppposition), ncb))
        
        for ii in range(0, ncb):
            allwave[:,ii] = v[:, self.nd + ii] 
        
        self.cbwave = np.zeros((int(len(self.ppposition)/2), ncb))
        vbwave = np.zeros((int(len(self.ppposition)/2), ncb))
        
        for ii in range(0, ncb):
            self.cbwave[:,ii] = allwave[0:self.nd, ii]
            vbwave[:,ii] = allwave[self.nd:, ii]


        ### Automatically determining QCL states involved in lazing action ###
        
        ## Determining active region states
        self.all_zz = np.zeros((ncb, 1))
        
        # calculate position expectation value
        for ii in range(ncb):
            if self.min_active < np.argmax(abs(self.cbwave[:,ii])) and np.argmax(abs(self.cbwave[:,ii])) < self.max_active:
            
                all_wave = allwave[:, ii]
                allzz = self.delz/3*(4*sum(np.conjugate(all_wave[0::2])*self.ppposition[0::2]*all_wave[0::2])\
                                +2*sum(np.conjugate(all_wave[1::2])*self.ppposition[1::2]*all_wave[1::2]))
                
                self.all_zz[ii] = allzz
            
            else:
                self.all_zz[ii] = 1e15 # 1e15 is arbitary chosen large enough number
        
        # define the position of wavefunction peak of each states 
        self.max_wave = np.zeros(ncb)
        
        for ii in range(ncb):
            # condition 1, 2: peak of states should be located in the first well in the active region.
            # condition 3: consider only active region states
            if self.min_active < np.argmax(abs(self.cbwave[:,ii])) and np.argmax(abs(self.cbwave[:,ii])) < self.min_active + sum(self.awmat[:2]) and self.all_zz[ii] !=1e15:
                self.max_wave[ii] = np.max(abs(self.cbwave[:,ii]))
                # condition: eigen value of the state should lower than mean conduction energy of next injection region
                if self.energy[ii] > self.CB_potential[int(round(sum(self.iwmat)*1.5)+sum(self.awmat))]:
                    self.max_wave[ii] = 0
            else:
                self.max_wave[ii] = 0
        

        zindex1 = np.argmax(self.max_wave)
        
        state4 = zindex1 # upper lasing state
        
        if zindex1 > 3:
            self.inj_zz = self.all_zz[0:state4]
            # self.inj_zz = self.all_zz[0:15]
            self.zindex3 = np.argsort(self.inj_zz, axis = 0)[:3]
            self.inj_index = np.sort(self.zindex3, axis = 0)
            
            # state3 = int(self.inj_index[2])
            # state2 = int(self.inj_index[1])
            # state1 = int(self.inj_index[0])
            state3 = int(self.inj_index[2].item())
            state2 = int(self.inj_index[1].item())
            state1 = int(self.inj_index[0].item())
            
            if np.argmax(abs(self.cbwave[:,state1])) > self.max_active:
                state3 = 0
                state2 = 0
                state1 = 0
            
        else:        
            state3 = 0
            state2 = 0
            state1 = 0
        
        self.st1 = state1
        self.st2 = state2
        self.st3 = state3
        self.st4 = state4
        
        ## Determining injection region states
        
        self.all_zz_in = np.zeros((ncb, 1))
        self.all_zz_ex = np.zeros((ncb, 1))
        
        # wavefunction peark position
        for ii in range(ncb):
            if self.min_active > np.argmax(abs(self.cbwave[:,ii])):
            
                all_wave = allwave[:, ii]
                allzzin = np.argmax(abs(self.cbwave[:,ii]))
                
                self.all_zz_in[ii] = allzzin
            
            else:
                self.all_zz_in[ii] = 0 # 0 is arbitary chosen small enough number
        
        candin = np.argsort(self.all_zz_in, axis = 0)[::-1]
        cand_in = candin[:2]
        
        self.stin0 = cand_in[0]
        self.stin1 = cand_in[1]
        
        self.Ein0 = self.energy[state4] - self.energy[self.stin0]
        self.Ein1 = self.energy[state4]-self.energy[self.stin1]
        
        ## Determining extraction region states

        for ii in range(ncb):
            if np.argmax(abs(self.cbwave[:,ii])) > self.max_active:
                all_wave = allwave[:, ii]
                
                allzzex = self.delz/3*(4*sum(np.conjugate(all_wave[0::2])*self.ppposition[0::2]*all_wave[0::2])\
                                +2*sum(np.conjugate(all_wave[1::2])*self.ppposition[1::2]*all_wave[1::2]))
                
                # allzzex = np.argmax(abs(self.cbwave[:,ii]))
                
                if ii < self.st2:
                    self.all_zz_ex[ii] = allzzex
                
                else:
                    self.all_zz_ex[ii] = 1e15
            else:
                self.all_zz_ex[ii] = 1e15 # 1e15 is arbitary chosen large enough number
                
        for ii in range(ncb):
            if self.all_zz_ex[ii] < self.mid_inj:
                Eout_candidate = abs(self.energy[state1]-self.energy[ii])
            else:
                Eout_candidate = 1e15
                
                
        # self.stex = np.argmin(self.all_zz_ex)
        self.stex = np.argmin(Eout_candidate)
        self.Eout = self.energy[state1]-self.energy[self.stex]
        
        #%% Wavefunction calculation
        
        self.ALLWAV1L = allwave[:, state1]
        self.ALLWAV2L = allwave[:, state2]
        self.ALLWAV3L = allwave[:, state3]
        self.ALLWAV4L = allwave[:, state4]
        
        self.CBWAV1L = self.cbwave[:, state1]
        self.CBWAV2L = self.cbwave[:, state2]
        self.CBWAV3L = self.cbwave[:, state3]
        self.CBWAV4L = self.cbwave[:, state4]
    
        self.VBWAV1L = vbwave[:, state1]
        self.VBWAV2L = vbwave[:, state2]
        self.VBWAV3L = vbwave[:, state3]
        self.VBWAV4L = vbwave[:, state4]
    
        self.ENR1L = self.energy[state1]
        self.ENR2L = self.energy[state2]
        self.ENR3L = self.energy[state3]
        self.ENR4L = self.energy[state4]
    
        self.CW1 = abs(self.CBWAV1L)**2
        self.CW2 = abs(self.CBWAV2L)**2
        self.CW3 = abs(self.CBWAV3L)**2
        self.CW4 = abs(self.CBWAV4L)**2
    
        self.VW1 = abs(self.VBWAV1L)**2
        self.VW2 = abs(self.VBWAV2L)**2
        self.VW3 = abs(self.VBWAV3L)**2
        self.VW4 = abs(self.VBWAV4L)**2
        
        # For fitting
        
        self.cw1 = self.CW1*10+self.ENR1L
        self.cw2 = self.CW2*10+self.ENR2L
        self.cw3 = self.CW3*10+self.ENR3L
        self.cw4 = self.CW4*10+self.ENR4L
        
        self.lw1 = self.VW1*10+self.ENR1L
        self.lw2 = self.VW2*10+self.ENR2L
        self.lw3 = self.VW3*10+self.ENR3L
        self.lw4 = self.VW4*10+self.ENR4L
        
        self.E43=(self.ENR4L-self.ENR3L)
        self.E42=(self.ENR4L-self.ENR2L)
        self.E41=(self.ENR4L-self.ENR1L)          
        self.E32=(self.ENR3L-self.ENR2L) 
        self.E31=(self.ENR3L-self.ENR1L)
        self.E21=(self.ENR2L-self.ENR1L)
        
    
    def Dipole(self):
        
        self.Z43= (4.0/3.0)*sum(self.ALLWAV4L[0::2]*self.ppposition[0::2]*self.ALLWAV3L[0::2]) +\
                  (2.0/3.0)*sum(self.ALLWAV4L[1::2]*self.ppposition[1::2]*self.ALLWAV3L[1::2])

    def IFR(self, iwave, fwave, DELTA = 3, LAMDA = 50):
        
        F = 0
        LineW = 0
        
        DELLAM = 97.3 * self.aung ** 2
        
        for ii in range(len(self.iw)-2):
            F += (self.gap0 / self.aung * (iwave[int(self.iw[ii])]**2 - fwave[int(self.iw[ii])]**2))**2 # gap0 unit: [J]
        
        LineW = np.pi * self.mt / self.hbar**2 * (DELLAM)**2 * F / self.ev /2 # divide ev to convert demension(J to eV)
        
        return LineW
        
        
        
    def POP(self, Eif, iwave, fwave):
        
        dies_IA = 15.15
        dief_IA = 12.3
        
        dies_GA = 12.9
        dief_GA = 10.89
        
        dies = dies_IA*self.x+dies_GA*(1-self.x)
        dief = dief_IA*self.x+dief_GA*(1-self.x)
        
        nphonon=1.0+(1.0/(np.exp(self.ephonon/(self.k*self.T))-1.0))
        epsilon = 8.8542e-12
        DDIE = dies*dief*epsilon/(dies-dief)
        con = self.ephonon*nphonon*self.ev**3/(4*math.pi*self.hbar**2*DDIE)*math.sqrt(self.mt/2) #unit: J , mt: electron effective mass
        
        iwave = iwave.reshape(self.nd,)
        fwave = fwave.reshape(self.nd,)
        
        if Eif-self.ephonon > 0:
            
            if self.absorption:
                
                ## emission part
                
                pop_part1 = 1/math.sqrt((Eif-self.ephonon)*self.ev) if Eif > self.ephonon else 0
                pop_part2 = 0  #unit: J
                pop_part3 = math.sqrt(2*self.mt/self.hbar**2)*math.sqrt((Eif-self.ephonon)*self.ev) if Eif > self.ephonon else 0
                
                #initialization
                tau0 = 0
                s1 = 0
                
                s30 = np.zeros(self.nd)
                odd0=np.zeros(self.nd)
                even0=np.zeros(self.nd)
        
                for ii in range(self.nd):
                    
                    odd0[ii] = sum(iwave[ii]*fwave[ii]*iwave[0::2]*fwave[0::2]*pop_part1*np.exp(-abs(self.pposition[ii]-self.pposition[0::2])*self.aung*(pop_part2 +pop_part3)))
                    even0[ii] = sum(iwave[ii]*fwave[ii]*iwave[1::2]*fwave[1::2]*pop_part1*np.exp(-abs(self.pposition[ii]-self.pposition[1::2])*self.aung*(pop_part2 +pop_part3)))
                            
                
                s2 = odd0*4/3+even0*2/3
                
                s3 = sum(s2[0::2])
                s4 = sum(s2[1::2])
        
                s1 = (s3*4/3 + s4*2/3)*np.pi
                
                ## absorption part
                
                pop_part10 = 1/math.sqrt((Eif+self.ephonon)*self.ev) 
                pop_part20 = 0  #unit: J
                pop_part30 = math.sqrt(2*self.mt/self.hbar**2)*math.sqrt((Eif+self.ephonon)*self.ev)
                
                # initialization        
                tau10 = 0
                
                s10 = 0
                s30 = np.zeros(self.nd)
                
                odd1=np.zeros(self.nd)
                even1=np.zeros(self.nd)
                
                for ii in range(self.nd):
                    
                    odd1[ii] = sum(iwave[ii]*fwave[ii]*iwave[0::2]*fwave[0::2]*pop_part10*np.exp(-abs(self.pposition[ii]-self.pposition[0::2])*self.aung*(pop_part20+pop_part30)))
                    even1[ii] = sum(iwave[ii]*fwave[ii]*iwave[1::2]*fwave[1::2]*pop_part10*np.exp(-abs(self.pposition[ii]-self.pposition[1::2])*self.aung*(pop_part20+pop_part30)))
                            
                s20= odd1*4/3+even1*2/3
                
                s30 = sum(s20[0::2])
                s40 = sum(s20[1::2])
        
                s10 = (s30*4/3 + s40*2/3)*np.pi
                
                s_tot = s1 + s10
                tau = 1/(con*s_tot)
            
            else:

                pop_part1 = 1/math.sqrt((Eif-self.ephonon)*self.ev) if Eif > self.ephonon else 0
                pop_part2 = 0  #unit: J
                pop_part3 = math.sqrt(2*self.mt/self.hbar**2)*math.sqrt((Eif-self.ephonon)*self.ev) if Eif > self.ephonon else 0
                
                #initialization
                tau0 = 0
                s1 = 0
                
                s30 = np.zeros(self.nd)
                odd0=np.zeros(self.nd)
                even0=np.zeros(self.nd)
        
                for ii in range(self.nd):
                    
                    odd0[ii] = sum(iwave[ii]*fwave[ii]*iwave[0::2]*fwave[0::2]*pop_part1*np.exp(-abs(self.pposition[ii]-self.pposition[0::2])*self.aung*(pop_part2 +pop_part3)))
                    even0[ii] = sum(iwave[ii]*fwave[ii]*iwave[1::2]*fwave[1::2]*pop_part1*np.exp(-abs(self.pposition[ii]-self.pposition[1::2])*self.aung*(pop_part2 +pop_part3)))
                            
                
                s2 = odd0*4/3+even0*2/3
                
                s3 = sum(s2[0::2])
                s4 = sum(s2[1::2])
        
                s1 = (s3*4/3 + s4*2/3)*np.pi
                
                tau = 1/(con*s1)
               
        
        else:
            if self.absorption:
                nphonon=(1.0/(np.exp(self.ephonon/(self.k*self.T))-1.0))
                con = self.ephonon * nphonon*self.ev**3/(4*math.pi*self.hbar**2*DDIE)*math.sqrt(self.mt/2) #unit: J , mt: electron effective mass
                
                pop_part1 = 1/math.sqrt((Eif+self.ephonon)*self.ev) 
                pop_part2 = 0  #unit: J
                pop_part3 = math.sqrt(2*self.mt/self.hbar**2)*math.sqrt((Eif+self.ephonon)*self.ev)
                
        
                # initialization        
                tau = 0
                
                s1 = 0
                s3 = np.zeros(self.nd)
                odd=np.zeros(self.nd)
                even=np.zeros(self.nd)
                
                for ii in range(self.nd):
                    
                    odd[ii] = sum(iwave[ii]*fwave[ii]*iwave[0::2]*fwave[0::2]*pop_part1*np.exp(-abs(self.pposition[ii]-self.pposition[0::2])*self.aung*(pop_part2+pop_part3)))
                    even[ii] = sum(iwave[ii]*fwave[ii]*iwave[1::2]*fwave[1::2]*pop_part1*np.exp(-abs(self.pposition[ii]-self.pposition[1::2])*self.aung*(pop_part2+pop_part3)))
                            
                s3= odd*4/3+even*2/3
                
                s2 = sum(s3[0::2])
                s22 = sum(s3[1::2])
        
                s1 = (s2*4/3+s22*2/3)*np.pi
                tau = 1/(con*s1)
            
            else:
                
                tau = 0
            
        return tau
    
        
    def Scatter_rate(self):
        
        # POP scattering
        cb_t43, vb_t43 = self.POP(self.E43,self.CBWAV4L,self.CBWAV3L), self.POP(self.E43,self.VBWAV4L,self.VBWAV3L)
        cb_t42, vb_t42 = self.POP(self.E42,self.CBWAV4L,self.CBWAV2L), self.POP(self.E42,self.VBWAV4L,self.VBWAV2L)
        cb_t41, vb_t41 = self.POP(self.E41,self.CBWAV4L,self.CBWAV1L), self.POP(self.E41,self.VBWAV4L,self.VBWAV1L)
        
        cb_t32, vb_t32 = self.POP(self.E32,self.CBWAV3L,self.CBWAV2L), self.POP(self.E32,self.VBWAV3L,self.VBWAV2L)
        cb_t31, vb_t31 = self.POP(self.E31,self.CBWAV3L,self.CBWAV1L), self.POP(self.E31,self.VBWAV3L,self.VBWAV1L)

        cb_t21, vb_t21 = self.POP(self.E21,self.CBWAV2L,self.CBWAV1L), self.POP(self.E21,self.VBWAV2L,self.VBWAV1L)
        
        cb = [cb_t43, cb_t42, cb_t41, cb_t32, cb_t31, cb_t21]
        vb = [vb_t43, vb_t42, vb_t41, vb_t32, vb_t31, vb_t21]
        
        self.LW_LO_tot = np.zeros(len(cb))
        self.tau_tot = np.zeros(len(cb))
        
        if self.absorption:
            for ii in range(len(cb)):
                self.tau_tot[ii] = cb[ii]*vb[ii]/(cb[ii]+vb[ii])
                self.LW_LO_tot[ii] = 1/self.tau_tot[ii]*self.hbar/self.ev
            

        else:
            
            for ii in range(len(cb)):
                if cb[ii] != 0:
                    self.tau_tot[ii] = cb[ii]*vb[ii]/(cb[ii]+vb[ii])
                    self.LW_LO_tot[ii] = 1/self.tau_tot[ii]*self.hbar/self.ev

                    
        # interface roughness scattering
        if self.LW_cal:
            LW_IFR43 = self.IFR(self.CBWAV4L, self.CBWAV3L) + self.IFR(self.VBWAV4L, self.VBWAV3L)
            LW_IFR42 = self.IFR(self.CBWAV4L, self.CBWAV2L) + self.IFR(self.VBWAV4L, self.VBWAV2L)
            LW_IFR41 = self.IFR(self.CBWAV4L, self.CBWAV1L) + self.IFR(self.VBWAV4L, self.VBWAV1L)
            LW_IFR32 = self.IFR(self.CBWAV3L, self.CBWAV2L) + self.IFR(self.VBWAV3L, self.VBWAV2L) 
            LW_IFR31 = self.IFR(self.CBWAV3L, self.CBWAV1L) + self.IFR(self.VBWAV3L, self.VBWAV1L) 
            LW_IFR21 = self.IFR(self.CBWAV2L, self.CBWAV1L) + self.IFR(self.VBWAV2L, self.VBWAV1L) 
            
            self.LW_IFR_tot = [LW_IFR43, LW_IFR42, LW_IFR41, LW_IFR32, LW_IFR31, LW_IFR21] 
        
       		 # Linewidth calculation
            self.LW_LO = sum(self.LW_LO_tot)
            self.LW_IFR = sum(self.LW_IFR_tot)
            self.LW = self.LW_IFR + self.LW_LO
        
            
    def gain(self):

        
        epsilon = 8.8542e-12
        Gamma = 0.5 # optical confinement factor
        N_ref = 3.47 # ground refractive index
        Area = 1.0*10**(-2)*10*10**(-5) # cm^2
        JJ = 100*10**(-3)/Area # input current density
        ALength = 510*10**(-8) # total thickness of one preiod (cm)
        # ALength = sum(self.thick)*10**(-8) # total thickness of one preiod (cm)
        
        if self.LW_cal:
            LineW = self.LW # eV
        else:
        	LineW = 7.5*10**(-3) # eV 300 K
        
        t43 = self.tau_tot[0]
        t42 = self.tau_tot[1]
        t41 = self.tau_tot[2]
        
        t32 = self.tau_tot[3]
        t31 = self.tau_tot[2]
        
        
        if all(self.tau_tot[:5])!=0:
            self.t4 = 1/((1/t43) + (1/t42) + (1/t41))
            self.t3 = 1/((1/t32) + (1/t31))
            
            self.prf1 = Gamma*(2*np.pi**2*self.Z43**2*10**(-16))/(epsilon*ALength*N_ref)
            self.prf2 = self.t4*(1-self.t3/t43)*JJ
            self.prf3 = LineW/np.pi
        
            Nowavelength = 1000
            
            Lamda = np.zeros(Nowavelength)
            G43 = np.zeros(Nowavelength)
        
            for ii in range(0,Nowavelength):
                Lamda[ii] = ii*0.01 + 2
                G43[ii] = self.prf1/(Lamda[ii]*10**(-4))*self.prf2*self.prf3/((self.E43-1.24/Lamda[ii])**2+LineW**2)
            
        else:
            Lamda = np.zeros(1000)
            G43 =np.zeros(1000)
            
            # for evaluation
            self.t4 = 0
        
        
        return Lamda, G43
        
