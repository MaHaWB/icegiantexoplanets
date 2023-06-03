# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:50:28 2021

@author: marcu
"""
import pickle
import numpy as np
from binary import binary
from ig_binary import ig_binary
import matplotlib.pyplot as plt
from tamanini_data import Tamanini
file = 'dict_binaries.txt'
yr = 24*3600*365.25 # s

# reset binaries
if False:
    outfile = open('dict_binaries.txt','wb')
    pickle.dump({'s': [], 'm': [],'l':[]},outfile)
    outfile.close()
    
# reset binaries_ig
if False:
    outfile = open('dict_binaries_ig.txt','wb')
    pickle.dump({'s': [], 'm': [],'l':[]},outfile)
    outfile.close()
    
# reset binaries_full
if False:
    outfile = open('dict_binaries_full.txt','wb')
    pickle.dump({'s': [], 'm': [],'l':[]},outfile)
    outfile.close()
    

if True:
    infile = open('dict_binaries.txt','rb')
    binaries = pickle.load(infile)
    infile.close()
    
    infile = open('dict_binaries_ig.txt','rb')
    binaries_ig = pickle.load(infile)
    infile.close()
    
    infile = open('dict_binaries_full.txt','rb')
    binaries_full = pickle.load(infile)
    infile.close()

if False:
    #for mode in ['s','m','l']:
    mode = 'm'
    binaries[mode].pop(9)
    binaries[mode].pop(6)
    bins = binaries[mode]
    for bina in bins:
        vals = bina['pars']
        #print(vals)
        B = binary(freq=vals['f_0'],mP=vals['M_P'],P=vals['P'],mode=mode)
        B.Fisher = bina['Fisher']
        B.snr = bina['SNR']
        B.add_json()
        
def numbers():
    print('binaries:')
    print('#s={}, #m={}, #l={} \n'.format(np.size(binaries['s']),np.size(binaries['m']),np.size(binaries['l'])))
          
    print('binaries_ig:')
    print('#s={}, #m={}, #l={} \n'.format(np.size(binaries_ig['s']),np.size(binaries_ig['m']),np.size(binaries_ig['l'])))
    
    print('binaries_full:')
    print('#s={}, #m={}, #l={}'.format(np.size(binaries_full['s']),np.size(binaries_full['m']),np.size(binaries_full['l'])))
    pass
        
numbers()

def print_all(binas=binaries_full,mode='l'):
    Ps = []
    tams = []
    for i in binas[mode]:
        Ps.append(i['pars']['P'])
        tams.append(i['Tamanini_plot'])
        
    tams = np.array(tams)
    
    plt.figure(dpi=300)
    plt.tight_layout()
    
    plt.loglog(Ps,tams[:,0],'rx',label=r'$\sigma_K/K$')
    plt.loglog(Ps,tams[:,1],'bx',label=r'$\sigma_P/P$')
    plt.loglog(Ps,tams[:,2],'gx',label=r'$\sigma_\varphi$')
    
    Tamanini(10)
    
    plt.title(r'Positions as in [1], $M_{b1,2}=0.23M_\odot,r=1kpc,M_P=1M_J, T_{obs}=4 yr$')
    plt.legend()
    plt.grid()
    plt.ylabel(r'$\sqrt{(\Gamma^{-1})_{ii}}/\lambda_i\cdot$SNR$\cdot M_P/M_J$')
    plt.xlabel(r'$P$ in yr')
    plt.savefig('Figures/Tamanini_comp.png')
    
def kill_trash():
    l = 0
    for bina in binaries_full['l']:
        A = np.sum(np.diag(bina['Error']) < 0)
        print(A)
        if A > 0:
            B = binaries_full['l'].pop(l)
            print('Killed {} yr'.format(B['pars']['P']))
        l += 1
    
    outfile = open('dict_binaries_full.txt','wb')
    pickle.dump(binaries_full,outfile)
    outfile.close()
    
print_all(binaries_ig,'l')
#kill_trash()