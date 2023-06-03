# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 22:04:52 2021

@author: marcu
"""

from scipy.constants import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
from tamanini_data import Tamanini
import LISA
import time
import pickle

pi = np.pi
AU = 1495978707e2 # m
pc = 3.0857e16 # 2e5 AU
yr = 24*3600*365.25 # s

M_J = 1.898e27 # kg
M_S = 1.989e30 # kg
r_S = 2*G/c**2 # m

sqth = np.sqrt(3)/2
omega_E = 2*pi/yr
fig = 'Figures/'

arr_yr = np.linspace(0,yr,1000)
from ig_binary import ig_binary

keys = enumerate(['K','P','phi_0', 'theta_S_Ec', 'phi_S_Ec', 'theta_L', 'phi_L', 'f_0', 'ln(A)'])

# Tamanini has []
n0 = 10
mP = 10
a = -1.4
b = 1.7
mode = 's'

Ps = np.logspace(a,b,n0)
uncs = np.zeros((n0,3),np.float64)
uncs2 = np.zeros((n0,3),np.float64)
times = np.zeros((2,n0))
bina = []
for n,P0 in enumerate(Ps):
    print('Now working on binary #',n+1,' / ',n0,' (P={:.2f} yr) \n'.format(P0))
    start = time.time()
    B1 = ig_binary(freq=1e-3,mP=mP,P=P0,dist=1e3,theta_P=pi/2,phi_P=pi/2,mode=mode,epsrel=1.49e-3)
    uncs[n] = B1.add_json()['Tamanini_plot']
    bina.append(B1.add_json()['binary'])
    end = time.time()
    times[0,n] = (end-start)/60
    print('Finished 10 mHz after {} min'.format(times[0,n]))
    '''
    start = time.time()
    B2 = binary(np.arccos(.3),5.,1e3,np.arccos(-.2),4.,m1=.23,m2=.23,freq=1e-3,mP=mP,P=P0,theta_P=pi/2,phi_P=pi/2,T_obs=4)
    uncs2[n] = B2.rel_uncertainty(mode)
    end = time.time()
    times[1,n] = (end-start)/60
    print('Finished 1 mHz after {} min'.format(times[1,n]))
    '''
plt.figure(dpi=300)
plt.tight_layout()

plt.loglog(Ps,uncs[:,0],'rx',label=r'$\sigma_K/K$')
plt.loglog(Ps,uncs[:,1],'bx',label=r'$\sigma_P/P$')
plt.loglog(Ps,uncs[:,2],'gx',label=r'$\sigma_\varphi$')

'''
B10 = ig_binary(freq=10e-3,P=10,mP=10,dist=1e3,mode='l').add_json()['Tamanini_plot']
plt.loglog(10,B10[0],'r^')
plt.loglog(10,B10[1],'b^')
plt.loglog(10,B10[2],'g^')

plt.loglog(Ps,uncs2[:,0],'r--')
plt.loglog(Ps,uncs2[:,1],'b--')
plt.loglog(Ps,uncs2[:,2],'g--')
'''

Tamanini(1)

plt.title(r'Positions as in [1], $M_{b1,2}=0.23M_\odot,r=1kpc,M_P=10 M_J$')
plt.legend()
plt.grid()
plt.ylabel(r'$\sqrt{(\Gamma^{-1})_{ii}}/\lambda_i\cdot$SNR$\cdot M_P/M_J$')
plt.xlabel(r'$P$ in yr')
plt.savefig(fig+'Relative_Uncertainties_ig_s.png')
plt.show()