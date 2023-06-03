import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folder = 'data_tamanini/'

class Tamanini:
    def __init__(self,freq=10,label=False):
        assert freq in [1,10,'1','10','all']
        self.freq = freq
        if freq == 1 or freq == '1':
            label_csv = 'dashed'
            ls = '--'
        if freq == 10 or freq == '10':
            label_csv = 'solid'
            ls = '-'
        
        if freq == 'all':
            Tamanini(1)
            Tamanini(10)
        else:
            self.P = pd.read_csv(folder+'blue_'+label_csv+'.csv').to_numpy()
            self.K = pd.read_csv(folder+'red_'+label_csv+'.csv').to_numpy()
            self.phi = pd.read_csv(folder+'green_'+label_csv+'.csv').to_numpy()
    
            self.Pm = np.stack([self.P[:,0],self.K[:,0],self.phi[:,0]]).mean(0)
            self.tamm = np.stack([self.K[:,1],self.P[:,1],self.phi[:,1]])
            
            if label:
                plt.loglog(self.Pm,self.tamm[0,:],'r',ls=ls,label=r'Tam. $\sigma_K/K$')
                plt.loglog(self.Pm,self.tamm[1,:],'b',ls=ls,label=r'Tam. $\sigma_P/P$')
                plt.loglog(self.Pm,self.tamm[2,:],'g',ls=ls,label=r'Tam. $\sigma_\varphi$')
            else:
                plt.loglog(self.Pm,self.tamm[0,:],'r',ls=ls)
                plt.loglog(self.Pm,self.tamm[1,:],'b',ls=ls)
                plt.loglog(self.Pm,self.tamm[2,:],'g',ls=ls)