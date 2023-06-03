# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 12:42:40 2022

@author: marcu
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

exos = pd.read_csv('data_tamanini/exoplanets.csv',header=0,skiprows=24,na_values='')
print(exos)

confirmed = exos['soltype'].to_numpy() == 'Published Confirmed'
Ps = (exos['pl_orbper'].to_numpy()/365.25)[confirmed]
Ms = exos['pl_bmassj'].to_numpy()[confirmed]
names = exos['pl_name'].to_numpy()[confirmed]
new = np.concatenate(([True],names[1:] != names[:-1]))
transit = exos['tran_flag'].to_numpy()
rv = (exos['rv_flag'].to_numpy() & (transit == 0))[confirmed]
transit = transit[confirmed]
micro = exos['micro_flag'].to_numpy()[confirmed]
imaging = exos['ima_flag'].to_numpy()[confirmed]
misc = (transit + micro + rv + imaging) == 0

plt.figure(dpi=300)
plt.tight_layout()
plt.style.use('seaborn-white')
for label in [[transit,'Transit'],[rv,'Radial Velocity'],[micro,'Microlensing'],[imaging,'Imaging'],[misc,'other']]:
    plt.scatter(Ps[(label[0] != 0)],Ms[(label[0] != 0)],label=label[1],marker='.',alpha=.4)
    print('{}: {}'.format(label[1],np.sum( label[0])))
plt.xscale('log')
plt.xlabel('Orbital Period $P$ in yr')
plt.yscale('log')
plt.ylabel('Planet mass $M_P$ or lower mass $\sin (i)M_P$ in $M_J$')
plt.grid()
plt.legend()
plt.savefig('Figures/exoplanets.pdf')