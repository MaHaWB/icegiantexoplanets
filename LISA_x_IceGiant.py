# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 13:24:49 2021

@author: marcu
"""

from scipy.constants import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
from tamanini_data import Tamanini
import LISA
import time
import astropy
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
from binary import binary
from ig_binary import ig_binary

infile = open('dict_binaries_full.txt','rb')
binaries = pickle.load(infile)
infile.close()

infile = open('dict_binaries_ig.txt','rb')
binaries_ig = pickle.load(infile)
infile.close()

def isclose(a, b, rel_tol=1e-04, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def comparison_Tam(mode='l', weight=1e-4):
    self.P = pd.read_csv(folder+'blue_'+label_csv+'.csv').to_numpy()
    self.K = pd.read_csv(folder+'red_'+label_csv+'.csv').to_numpy()
    self.phi = pd.read_csv(folder+'green_'+label_csv+'.csv').to_numpy()

    self.Pm = np.stack([self.P[:,0],self.K[:,0],self.phi[:,0]]).mean(0)
    self.tamm = np.stack([self.K[:,1],self.P[:,1],self.phi[:,1]])
    
    
    ig_bins = binaries_ig[mode]
    P = np.array([b['pars']['P'] for b in bins])
    P_ig = np.array([b['pars']['P'] for b in ig_bins])
    P_fin = []
    F_fin = []
    Tam_fin = []
    for i, Pi in enumerate(P):
        if mode == 's':
            vals = np.array([bins[i]['pars']['K'],Pi*yr,1.])
        if mode == 'm':
            b = bins[i]['binary']
            vals = np.array([b.K,b.P,1.,1.,1.,1.,1.,np.log(b.a0),b.f_1])
        if mode == 'l':
            b = bins[i]['binary']
            vals = np.array([b.K,b.P,1.,1.,1.,1.,1.,np.log(b.a0),b.f_1,b.f_GW])
        for j, Pj in enumerate(P_ig):
            if isclose(Pi,Pj):
                P_fin.append(Pi)
                A = ig_bins[j]['SNR']**2/2
                B = bins[i]['SNR']**2/2*bins[i]['binary'].S_n()
                F = bins[i]['Fisher']/bins[i]['SNR']**2/(1+weight*A/B)/bins[i]['pars']['M_P']**2 + weight*ig_bins[j]['Fisher']/2/(weight*A+B)/ig_bins[j]['pars']['M_P']**2
                F_fin.append(F)
                Tam_fin.append(np.sqrt(np.diag(np.linalg.inv(np.array(F_fin[-1]))/np.outer(vals,vals))))
                print('SNR_L = {:.3f}, SNR_IG = {:.3f}, total = {:.3f}\n'.format(bins[i]['SNR'],weight*bins[i]['SNR'](1+bins[i]['Fisher'][7,7]/ig_bins[j]['Fisher'][7,7]),bins[i]['SNR']**2 + (weight*bins[i]['SNR'](1+bins[i]['Fisher'][7,7]/ig_bins[j]['Fisher'][7,7]))**2))
    Tam_fin = np.array(Tam_fin)
    
    plt.figure(dpi=300)
    plt.tight_layout()
    
    plt.title(r'LISA x IceGiant for $S_n^{LISA}(10 mHZ)/S_n^{IG}(10 mHz) = $'+'{:.1e}'.format(weight))
    
    plt.loglog(P_fin,Tam_fin[:,0],'rx',label=r'$\sigma_K/K$')
    plt.loglog(P_fin,Tam_fin[:,1],'bx',label=r'$\sigma_P/P$')
    plt.loglog(P_fin,Tam_fin[:,2],'gx',label=r'$\sigma_\varphi$')
    
    plt.xlabel(r'$P$ in yr')
    plt.ylabel(r'$\sqrt{(\Gamma^{-1})_{ii}}/\lambda_i\cdot$SNR$\cdot M_P/M_J$')
    
    Tamanini(10)
    
    plt.tight_layout()
    
    plt.grid()
    
    plt.savefig(fig+'rel_uncertainty_added.pdf')

def comparison(mode='l', weight=1e-4):
    bins = binaries[mode]
    ig_bins = binaries_ig[mode]
    P = [b['pars']['P'] for b in bins]
    P = np.array(P)
    P_ig = np.array([b['pars']['P'] for b in ig_bins])
    P_fin = []
    F_fin = []
    Tam_fin = []
    Tam_L = []
    for i, Pi in enumerate(P):
        if mode == 's':
            vals = np.array([bins[i]['pars']['K'],Pi*yr,1.])
        if mode == 'm':
            b = bins[i]['binary']
            vals = np.array([b.K,b.P,1.,1.,1.,1.,1.,np.log(b.a0),b.f_1])
        if mode == 'l':
            b = bins[i]['binary']
            vals = np.array([b.K,b.P,1.,1.,1.,1.,1.,np.log(b.a0),b.f_1,b.f_GW])
        for j, Pj in enumerate(P_ig):
            if isclose(Pi,Pj):
                P_fin.append(Pi)
                A = ig_bins[j]['SNR']**2/2
                B = bins[i]['SNR']**2/2*bins[i]['binary'].S_n()
                SNR = (1+weight**2)*bins[i]['SNR']**2
                FIG = ig_bins[j]['Fisher']/2 * weight**2 * bins[i]['SNR']**2/A / ig_bins[j]['pars']['M_P']**2 / SNR
                FL = bins[i]['Fisher']*bins[i]['binary'].S_n()/2*bins[j]['SNR']**2/B / bins[i]['pars']['M_P']**2 / SNR
                F = FIG + FL
                F_fin.append(F)
                Tam_fin.append(np.sqrt(np.diag(np.linalg.inv(np.array(F_fin[-1]))/np.outer(vals,vals))))
                Tam_L.append(bins[i]['Tamanini_plot'])

                #print('SNR_L = {:.3f}, SNR_IG = {:.3f}, total = \n'.format(bins[i]['SNR'], weight/Binary.S_n()*bins[i]['Fisher'][7,7])) #weight*bins[i]['SNR']*(1+(bins[i]['Fisher'][7,7]*Binary.S_n()/2)/(ig_bins[j]['Fisher'][7,7]*Binary.S_n()/weight*2)))) #bins[i]['SNR']**2 + (weight*bins[i]['SNR'](1+bins[i]['Fisher'][7,7]/ig_bins[j]['Fisher'][7,7]))**2))
    Tam_fin = np.array(Tam_fin)
    Tam_L = np.array(Tam_L)
    
    plt.figure(dpi=300)
    #plt.tight_layout()
    
   # plt.title(r'LISA x IceGiant for $S_n^{LISA}(10 mHZ)/S_n^{IG}(10 mHz) = $'+'{:.1e}'.format(weight))
    
    plt.loglog(P,Tam_L[:,0],'rx')
    plt.loglog(P,Tam_L[:,1],'bx')
    plt.loglog(P,Tam_L[:,2],'gx')
    
    plt.loglog(P_fin,Tam_fin[:,0],'rs',ms=4.4,label=r'$\sigma_K/K$')
    plt.loglog(P_fin,Tam_fin[:,1],'bs',ms=4.4,label=r'$\sigma_P/P$')
    plt.loglog(P_fin,Tam_fin[:,2],'gs',ms=4.4,label=r'$\sigma_\varphi$')
    
    plt.xlabel(r'$P$ in yr')
    plt.ylabel(r'$\sigma_i/\lambda_i\cdot S/N \cdot M_P/M_J$')
    
    plt.legend()
    
    Tamanini(10)
    
    plt.tight_layout()
    
    plt.grid()
    
    plt.savefig(fig+'rel_uncertainty_added_100x.pdf')
    

plt.figure(dpi=300)
comparison(weight=1e-1,mode='l')
plt.show()