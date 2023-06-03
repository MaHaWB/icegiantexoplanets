from tokenize import PseudoToken
from scipy.constants import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import LISA
import time
from astropy import units
import astropy.coordinates as coord
import pickle
from binary import binary
import pandas as pd
from tamanini_data import Tamanini

pi = np.pi
AU = 1495978707e2 # m
pc = 3.0857e16 # 2e5 AU
yr = 24*3600*365.25 # s

M_J = 1.898e27 # kg
M_E = 5.9722e24 # kg
M_S = 1.989e30 # kg
R_S = 696340e3 # m
r_S = 2*G/c**2 # m

compEJ = .1
compBDJ = 0.08*M_S/M_J

fig = 'Figures/'

def calc(args,name,snr,mode='m'):
    n0 = 10
    mP = 1
    a = -1.9
    b = 1
    
    Ps = np.logspace(a,b,n0)
    uncs = np.zeros((n0,3),np.float64)
    uncs2 = np.zeros((n0,3),np.float64)
    times = np.zeros((2,n0))
    bina = []
    for n,P0 in enumerate(Ps):
        print('Now working on binary #',n+1,' / ',n0,' (P={:.2f} yr) \n'.format(P0))
        start = time.time()
        B1 = binary(*args,mP=mP,P=P0,theta_P=pi/2,phi_P=pi/2,mode=mode)
        uncs[n] = B1.add_json()['Tamanini_plot']
        bina.append(B1.js['binary'])
        end = time.time()
        times[0,n] = (end-start)/60
        print('Finished 10 mHz after {} min'.format(times[0,n]))

    plt.figure(dpi=300)
    plt.tight_layout()
    
    plt.loglog(Ps,uncs[:,0],'rx',label=r'$\sigma_K/K$')
    plt.loglog(Ps,uncs[:,1],'bx',label=r'$\sigma_P/P$')
    plt.loglog(Ps,uncs[:,2],'gx',label=r'$\sigma_\varphi$')
    
    '''
    plt.loglog(Ps,uncs2[:,0],'r--')
    plt.loglog(Ps,uncs2[:,1],'b--')
    plt.loglog(Ps,uncs2[:,2],'g--')
    '''
    
    if B1.f_GW < np.sqrt(10)*1e-3:
        Tamanini(1)
    else:
        Tamanini(10)
    
    plt.title(r'Exoplanet uncertainties for {}'.format(name))
    plt.legend()
    plt.grid()
    plt.ylabel(r'$\sqrt{(\Gamma^{-1})_{ii}}/\lambda_i\cdot$SNR$\cdot M_P/M_J$')
    plt.xlabel(r'$P$ in yr')
    plt.savefig(fig+name.replace(" ","_")+'_'+mode+'_rel_unc.png')
    plt.show()
    
    return bina

def triangular_plot(bina=0,name='Demo 1 mHz',numM=100,numI=100,snr=200,demo=False,freq=1):
    if demo:
        assert name in ['Demo 1 mHz','Demo 10 mHz']
        folder = 'data_tamanini/'
        if name == 'Demo 1 mHz':
            label_csv = 'dashed'
            freq = 1
        else:
            label_csv = 'solid'
            freq = 10
        tamP = pd.read_csv(folder+'blue_'+label_csv+'.csv').to_numpy()[:,1]
        Ps = pd.read_csv(folder+'blue_'+label_csv+'.csv').to_numpy()[:,0]
        tamK = pd.read_csv(folder+'red_'+label_csv+'.csv').to_numpy()[:,1]
        bina = np.zeros(len(Ps))
    else:
        Ps = np.array([])
        tamP = np.array([])
        tamK = np.array([])
        for b in bina:
            Ps.append(b.js['pars']['P'])
            tamP.append(b.js['Tamanini_plot'][1])
            tamK.append(b.js['Tamanini_plot'][0])
    
    Ms = np.logspace(np.log10(compEJ),np.log10(compBDJ),numM)
    cosInc = np.linspace(-1,1,numI)
    sinInc = np.sqrt(1-cosInc**2)
    Cube_tot = np.zeros((len(bina),numM,numI)) # P, M, cosI
    Cube_P = np.zeros_like(Cube_tot)
    Cube_K = np.zeros_like(Cube_tot)
    
    Cube_P = (Cube_P.T + tamP).T/snr
    Cube_K = (Cube_K.T + tamK).T/snr
    Cube_add = np.sqrt(tamP*tamK)/snr
    

    K = np.outer(Ms,sinInc)
    Cube_P /= K
    Cube_K /= K
    
    '''
    Cube_tot = (Cube_P < 0.3) * (Cube_K < 0.3) # TODO: Geometric Mean rel. unc.
    
    total = np.sum(Cube_tot)

    print(total)
    
    # Plots #
    # Plot 1 #
    
    fig, axes = plt.subplots(figsize=(10, 10), ncols=3, nrows=3)
    for i in range(3):
        for j in range(3):
            if i<j:
                axes[i, j].axis('off')
    axes[0, 0].semilogx(Ps,np.sum(Cube_tot,(1,2)).T/np.max(np.sum(Cube_tot,(1,2))))
    axes[1, 1].semilogx(Ms,np.sum(Cube_tot,(0,2)).T/np.max(np.sum(Cube_tot,(0,2))))
    axes[2, 2].plot(cosInc,np.sum(Cube_tot,(0,1)).T/np.max(np.sum(Cube_tot,(0,1))))
    axes[2, 2].set_xlabel('cos$(i)$')
    axes[1, 0].pcolor(*np.meshgrid(Ps,Ms),np.sum(Cube_tot,2).T/np.max(np.sum(Cube_tot,2)), cmap='Greens',vmin=0,vmax=1)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_ylabel('$M_P$ in $M_J$')
    axes[2, 0].pcolor(*np.meshgrid(Ps,cosInc),np.sum(Cube_tot,1).T/np.max(np.sum(Cube_tot,1)), cmap='Greens',vmin=0,vmax=1)
    axes[2, 0].set_xscale('log')
    axes[2, 0].set_ylabel('cos$(i)$')
    axes[2, 0].set_xlabel('$P$ in yr')
    axes[2, 1].pcolor(*np.meshgrid(Ms,cosInc),np.sum(Cube_tot,0).T/np.max(np.sum(Cube_tot,0)), cmap='Greens',vmin=0,vmax=1)
    axes[2, 1].set_xscale('log')
    axes[2, 1].set_xlabel('$M_P$ in $M_J$')
    
    fig.text(.85,.85,'Exoplanets recovarable (green) and\nunobservable (white) around a binary\nwith properties as {},\nso frequency = {:.1f} mHz\nand S/N = {:.1f}'.format(name,freq,snr),fontsize=13,horizontalalignment='right',verticalalignment='top',bbox=dict(facecolor='green', alpha=0.2))
    
    plt.savefig('Figures/Triangle/'+name.replace(" ","_")+'.png')
    plt.show()
    '''

    # Plot 2 #
    
    Cube_tot = np.sqrt(Cube_P*Cube_K)

    fig, axes = plt.subplots(figsize=(10, 10), ncols=3, nrows=3)
    for i in range(3):
        for j in range(3):
            if i<j:
                axes[i, j].axis('off')
    axes[0, 0].semilogx(Ps,pi/2*(1/compEJ-1/compBDJ)*Cube_add)
    axes[0, 0].set_ylim((-.2,10))
    axes[1, 1].semilogx(Ms,np.mean(Cube_add)*pi/2/Ms)
    axes[1, 1].set_ylim((-.2,10))
    axes[2, 2].plot(cosInc,np.mean(Cube_add)*(1/compEJ-1/compBDJ)/sinInc)
    axes[2, 2].set_ylim((-.2,10))
    axes[2, 2].set_xlabel('cos$(i)$')
    axes[1, 0].pcolor(*np.meshgrid(Ps,Ms),np.outer(Cube_add,1/Ms).T*pi/2, cmap='Greens',vmin=0,vmax=3)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_ylabel('$M_P$ in $M_J$')
    axes[2, 0].contour(*np.meshgrid(Ps,cosInc),np.outer(Cube_add,1/sinInc).T*(1/compEJ-1/compBDJ),levels=[5,10,15])
    axes[2, 0].set_xscale('log')
    axes[2, 0].set_ylabel('cos$(i)$')
    axes[2, 0].set_xlabel('$P$ in yr')
    axes[2, 1].pcolor(*np.meshgrid(Ms,cosInc),np.mean(Cube_add)*1/np.outer(Ms,sinInc).T, cmap='Greens')
    axes[2, 1].set_xscale('log')
    axes[2, 1].set_xlabel('$M_P$ in $M_J$')
    #plt.colorbar()
    
    #fig.text(.85,.85,r'Expectation value of the uncertainty (green) \n $E[\sqrt{\sigma_P \cdot \sigma_K}$ around a binary\nwith properties as {},\nso frequency = {:.1f} mHz\nand S/N = {:.1f}'.format(name,freq,snr),fontsize=13,horizontalalignment='right',verticalalignment='top',bbox=dict(facecolor='green', alpha=0.2))
    
    plt.savefig('Figures/Triangle/'+name.replace(" ","_")+'_Uncertainty.png')
    plt.show()

triangular_plot(name='Demo 1 mHz',demo=True)
triangular_plot(name='Demo 10 mHz',demo=True)

assert False

# ------ HM Cancri ------- #

# Parameters:
# theta_S_Ec=23.3952,phi_S_Ec=206.9246*pi/180,dist=5e3,theta_L=38,phi_L=206.9246*pi/180,m1=.55,m2=.27,freq=6.22e-3,snr=211.1

args_HM_Cnc = [23.3952*pi/180,206.9246*pi/180,5e3,38*pi/180,206.9246*pi/180,.55,.27,6.22e-3]
HM_Cnc = binary(*args_HM_Cnc,mP=0,T_obs=4)
#snr_HM_Cnc = HM_Cnc.signal_to_noise(both=True)
bina = calc(args_HM_Cnc,'HM Cnc',211.1,'m')
triangular_plot(bina,'HM Cnc',numM=100,numI=100,snr=211.1,freq=6.22)

# ------ V407 Vul ------- #

# Parameters:
# theta_S_Ec=6.4006*pi/180,phi_S_Ec= 57.7281*pi/180,dist=1786,theta_L=60*pi/180,phi_L=57.7281*pi/180,m1=0.8,m2=0.177,freq=3.51e-3, snr=169.7

args_V407 = [6.4006*pi/180,57.7281*pi/180,1786,60*pi/180,57.7281*pi/180,0.8,0.177,3.51e-3]
snr = 169.7
V407 = binary(*args_HM_Cnc,mP=0,T_obs=4)
#snr_V407 = V407.signal_to_noise(both=True)
bina_V407 = calc(args_V407,'V407 Vul',211.1,'s')
triangular_plot(bina_V407,'V407 Vul',numM=100,numI=100,snr=snr,freq=6.22)