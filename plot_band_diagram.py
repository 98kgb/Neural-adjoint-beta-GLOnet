# -*- coding: utf-8 -*-
"""
Plot the conduction-band diagram with the moduli-squared wave functions (left) and the gain
spectrum G43 (right) of given 24-layer structures, computed with qcl_simulator.QCL.
Edit `str_ref` / `str_opt` / `structure` to plot other designs.

@author: Gibaek Kim
"""
import os
import sys

# Add the current file path to the system path
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

from qcl_simulator import QCL

import numpy as np
import matplotlib.pyplot as plt

# structure = np.genfromtxt(current_path + "/Data/active_only/structure_4QW_7_7um_valid.csv", delimiter = ',')[0,:]
# injector = np.array([22,43,15,38,16,34,18,30,21,28,25,27,32,27,36,25])

# structure = np.hstack([structure, injector])
structure = [48,19,23,50,14,51,9,48,17,41,17,39,15,34,20,36,21,29,27,28,29,24,35,24]
str_ref = [43,18,9,55,11,53,12,47,22,43,15,38,16,34,18,30,21,28,25,27,32,27,36,25]
str_opt = [38,19,12,53,14,53,14,48,19,44,19,37,16,32,17,31,22,31,26,27,29,25,35,26]

model = QCL(structure, LW_cal=True)
model.Potential()
model.EVMAT()
model.CBEIGEN()
model.Dipole()
model.Scatter_rate()
lamda, g43 = model.gain()

ref = QCL(str_ref, LW_cal=True)
ref.Potential()
ref.EVMAT()
ref.CBEIGEN()
ref.Dipole()
ref.Scatter_rate()
lamda_ref, g43_ref = ref.gain()


opt = QCL(str_opt, LW_cal=True)
opt.Potential()
opt.EVMAT()
opt.CBEIGEN()
opt.Dipole()
opt.Scatter_rate()
lamda_opt, g43_opt = opt.gain()


#%%
fig, ax = plt.subplots(1, 2, figsize = (12,5), dpi=300)
fontsize = 18
# ax[1].plot(lamda, g43, 'b', linewidth = 2, alpha = 0.7, label = 'Optimized')
ax[1].plot(lamda_ref, g43_ref, 'b', linewidth = 2, alpha = 0.7, label = 'Ref')
# ax[1].plot(lamda_opt, g43_opt, '--r', linewidth = 2, alpha = 0.7, label = 'opt')
ax[1].grid()
ax[1].set_xlabel('$Wavelength\ (\mu m)$', fontsize = fontsize)
ax[1].set_ylabel('$G_{43}\ (cm^{-1})$', fontsize = fontsize)
# ax[1].legend(fontsize = fontsize-1)
# ax1.set_title('$Gain$: {}  Wavelength: {}'.format(np.round(np.max(g43)), np.round(lamda[np.argmax(g43)], 2)))

ax[0].plot(model.pos, model.CB_potential, 'k')

# ax[0].set_title('Einj: {}  Eext: {}'.format(np.round(abs(model.Ein0[0]*1000),2), np.round(model.Eout*1000, 2)))
for kk in range(20):
    if kk == model.st1:
        ax[0].plot(model.pos, (model.cbwave[:, kk])**2*10 + model.energy[kk], 'r', linewidth = 2, label = '$\Phi_{1}$')
    elif kk == model.st2:
        ax[0].plot(model.pos, (model.cbwave[:, kk])**2*10 + model.energy[kk], 'g', linewidth = 2, label = '$\Phi_{2}$')
    elif kk == model.st3:
        ax[0].plot(model.pos, (model.cbwave[:, kk])**2*10 + model.energy[kk], 'b', linewidth = 2, label = '$\Phi_{3}$')
    elif kk == model.st4:
        ax[0].plot(model.pos, (model.cbwave[:, kk])**2*10 + model.energy[kk], 'm', linewidth = 2, label = '$\Phi_{4}$')
    elif kk == model.stin0:
        ax[0].plot(model.pos, (model.cbwave[:, kk])**2*10 + model.energy[kk], 'c', linewidth = 2, label = '$\Phi_{inj}$')
    elif kk == model.stex:
        ax[0].plot(model.pos, (model.cbwave[:, kk])**2*10 + model.energy[kk], 'y', linewidth = 2, label = '$\Phi_{ext}$')
    else:
        ax[0].plot(model.pos, (model.cbwave[:, kk])**2*10 + model.energy[kk], '--k', linewidth = 0.5)

ax[0].set_xlim([0,1100])
ax[0].set_xlabel('$Position\ (\AA)$', fontsize = fontsize)
ax[0].set_ylabel('$Potential\ (eV)$', fontsize = fontsize)
handles, labels = ax[0].get_legend_handles_labels()

label_order = ['$\\Phi_{1}$', '$\\Phi_{2}$', '$\\Phi_{3}$', '$\\Phi_{4}$', '$\\Phi_{inj}$', '$\\Phi_{ext}$']
ordered_handles = [handles[labels.index(lab)] for lab in label_order]

# legend in the defined order
ax[0].legend(ordered_handles, label_order, fontsize = fontsize-5)
#%%
fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=300, layout='constrained')
fontsize = 18

colors = {
    'phi1': '#E64B35',
    'phi2': '#3C8D40',
    'phi3': '#5B4AE6',
    'phi4': '#C44EAD',
    'inj': '#28AEB4',
    'ext': '#B5A82D',
    'gain': '#5B4AE6'
}

# Right
ax[1].plot(lamda_ref, g43_ref, color=colors['gain'], linewidth=2.3)
ax[1].set_xlabel(r'Wavelength ($\mu$m)', fontsize=fontsize)
ax[1].set_ylabel(r'$G_{43}$ (cm$^{-1}$)', fontsize=fontsize)
ax[1].grid(True, linewidth=0.6, alpha=0.35)
ax[1].tick_params(direction='in', top=True, right=True, labelsize=fontsize-5, length=5, width=1)

# Left
ax[0].plot(model.pos, model.CB_potential, color='black', linewidth=1.4)

for kk in range(20):
    y = (model.cbwave[:, kk])**2*10 + model.energy[kk]

    if kk == model.st1:
        ax[0].plot(model.pos, y, color=colors['phi1'], linewidth=2.2, label=r'$\Phi_1$')
    elif kk == model.st2:
        ax[0].plot(model.pos, y, color=colors['phi2'], linewidth=2.2, label=r'$\Phi_2$')
    elif kk == model.st3:
        ax[0].plot(model.pos, y, color=colors['phi3'], linewidth=2.2, label=r'$\Phi_3$')
    elif kk == model.st4:
        ax[0].plot(model.pos, y, color=colors['phi4'], linewidth=2.2, label=r'$\Phi_4$')
    elif kk == model.stin0:
        ax[0].plot(model.pos, y, color=colors['inj'], linewidth=2.2, label=r'$\Phi_{\mathrm{inj}}$')
    elif kk == model.stex:
        ax[0].plot(model.pos, y, color=colors['ext'], linewidth=2.2, label=r'$\Phi_{\mathrm{ext}}$')
    else:
        ax[0].plot(model.pos, y, '--', color='black', linewidth=0.55, alpha=0.65)

ax[0].set_xlim(0, 1100)
ax[0].set_xlabel(r'Position ($\mathrm{\AA}$)', fontsize=fontsize)
ax[0].set_ylabel(r'Potential (eV)', fontsize=fontsize)
ax[0].tick_params(direction='in', top=True, right=True, labelsize=fontsize-5, length=5, width=1)

handles, labels = ax[0].get_legend_handles_labels()
label_order = [
    r'$\Phi_1$', r'$\Phi_2$', r'$\Phi_3$',
    r'$\Phi_4$', r'$\Phi_{\mathrm{inj}}$', r'$\Phi_{\mathrm{ext}}$'
]
ordered_handles = [handles[labels.index(lab)] for lab in label_order]

ax[0].legend(
    ordered_handles, label_order,
    loc='upper center', ncol=2,
    fontsize=fontsize-5,
    frameon=True, framealpha=0.9,
    handlelength=2.0, columnspacing=1.2,
    borderpad=0.5
)

for a in ax:
    for spine in a.spines.values():
        spine.set_linewidth(1.2)

plt.show()