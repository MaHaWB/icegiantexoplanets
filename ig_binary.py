# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 15:43:10 2021

@author: marcu
"""

from tempfile import tempdir
from scipy.constants import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import LISA
import pickle
import mpmath

mpmath.mp.dps = 50

pi = np.pi
AU = 1495978707e2 # m
pc = 3.0857e16 # 2e5 AU
yr = 24*3600*365.25 # s
day = 24*3600

M_J = 1.898e27 # kg
M_S = 1.989e30 # kg
r_S = 2*G/c**2 # m
sqth = np.sqrt(3)/2

labels=['K','P','phi_0', 'theta_S_Ec', 'phi_S_Ec', 'theta_L', 'phi_L', 'ln(A)', 'f_1', 'f_0']
keys = enumerate(labels)
mode_size = {'s' : 3, 'm' : 9, 'l' : 10}

file = 'dict_binaries_ig.txt'

def isclose(a, b, rel_tol=1e-04, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

class ig_binary:

    def __init__(self,theta_S_Ec=np.arccos(.3),phi_S_Ec=5.,dist=1e3,theta_L=np.arccos(-.2),phi_L=4.,m1=.23,m2=.23,freq=2.6e-3,mP=1,P=2,theta_P=pi/2,phi_P=pi/2,mode='m',epsabs=0,epsrel=1e-2):
        '''
        Parameters
        ----------
        theta_S_Ec : float in [0,pi]
        Inclination angle of the source in ecliptic coordinates in rad
        phi_S_Ec : float in [0,2*pi]
        Azimuth angle of the source in the ecliptic coordinates in rad
        dist : float
        Distance from us  binary system in pc
        theta_L : float in [0,pi]
        The inclination angle of the normal to the binaries orbital plane in ecliptic coords in rad
        phi_L : float in [0,2*pi]
        The azimuth angle of the normal to the binaries orbital plane in ecliptic coords
        m1, m2 : float
        The mass of individual binary partners in M_S
        freq : float
        The frequency of the GW emitted
        mP : float
        The mass of the single exoplanet in M_J
        sepP : float
        The seperation from the centre of mass of the binaries to the planet in AU
        theta_P : float in [0,pi]
        The inclination angle of the normal to the planets orbital plane in ecliptic coords in rad
        phi_P : float in [0,2*pi]
        The azimuth angle of the normal to the planets orbital plane in ecliptic coords
        T_obs : float
        Mission duration in years
        mode : 's', 'm' or 'l'
        Specifies the dimensions of the Fisher-mat: 3x3, 9x9, or 10x10
        '''

        # Save the parameters of the binary system + exoplanet
        self.theta_S_Ec = theta_S_Ec
        self.phi_S_Ec = phi_S_Ec
        self.dist = dist*pc #m
        self.theta_L = theta_L
        self.phi_L = phi_L
        self.m1 = m1*M_S #kg
        self.m2 = m2*M_S #kg
        self.mP = mP*M_J #kg
        self.P = P*yr #s
        self.theta_P = theta_P
        self.phi_0 = phi_P
        self.K = (2*pi*G/self.P)**(1/3)*self.mP/(self.m1+self.m2+self.mP)**(2/3)*np.sin(theta_P)/c
        
        # Save the direction specifications of our detector
        self.ig_direction = 0. # TODO: Direction earth-neptune
        self.T_2 = 30000 # TODO: Change to a list # Change to 15000 !!!
        
        self.phi_rel = self.phi_S_Ec - self.ig_direction
        
        assert mode in ['s','m','l']
        self.mode = mode
        
        # Compute relevant parameters of out binary system
        self.chirp_mass = (self.m1*self.m2)**(3/5)/((self.m1+self.m2))**(1/5) #kg
        self.f_binary = freq/2
        self.f_GW = freq
        self.f_1 = 96/5*pi**(8/3)*freq**(11/3)*(G*self.chirp_mass/c**3)**(5/3)

        self.cos_inclination = np.cos(self.theta_L)*np.cos(self.theta_S_Ec) + np.sin(self.theta_L)*np.sin(self.theta_S_Ec)*np.cos(self.phi_L-self.phi_S_Ec) # = L*n
        self.a0 = 4/self.dist*(G*self.chirp_mass/c**2)**(5/3)*(pi*self.f_GW/c)**(2/3)
        self.gw_amplitude = self.a0*np.array([(1.+self.cos_inclination**2)/2, self.cos_inclination]) # [A_plus, A_cross]
        
        # Compute the beam pattern function
        #self.strain_amplitude = self.A(t)
        self.F = self.F_ig()
        
        # Compute the scalar product k * n = cos(angle los, detector) = mu of detector and line of sight
        self.kn = np.sin(theta_S_Ec)*(np.cos(self.ig_direction)*np.cos(phi_S_Ec)+np.sin(self.ig_direction)*np.sin(phi_S_Ec))
        print('mu={}'.format(self.kn))
        assert self.kn != 1
        
        # Set the parameters for numerical integration 
        self.epsrel = epsrel
        self.epsabs = epsabs

        # Compute the relevant phases for the GW signal in the detector
        self.polarization_phase()
        self.A()
    
    def sep(self):
        '''
        Returns the seperation for a given binary system in m
        '''
        return (G*(self.m1+self.m2)/(pi*self.f_GW)**2)**(1/6) #m
    
    def S_n(self):
        '''
        Compute the strain sensitivity - currently left out because we report per SNR
        '''
        return 1. # irrelevant atm, as we report in SNR
    
    def r_s(self,m):
        '''
        Returns
        -------
        r_s : The length of the Schwarzschild radius in m.
        '''
        return m*r_S
    
    def psi_S(self):
        '''
        Returns
        -------
        psi_S : The polarization angle psi_S of the wavefront in the detector frame used to bring the GW to amplitude-phase-form - stolen from Cutler
        '''
        Lz = .5*np.cos(self.theta_L)-sqth*np.sin(self.theta_L)*np.cos(self.ig_direction-self.phi_L)
        Ln = self.cos_inclination
        nLxz = .5*np.sin(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_L-self.phi_S_Ec) - sqth*np.cos(self.ig_direction)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec)-np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)) - sqth*np.sin(self.ig_direction)*(np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.cos(self.phi_L) - np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.cos(self.phi_S_Ec))
        return np.arctan2(Lz - Ln*np.cos(self.theta_S_Ec),nLxz)
    
    def F_ig(self):
        '''
        Returns
        -------
        [F_plus(t), F_cross(t)] : The beam pattern function of the specified geometry, already transformed, s.t. we can get to amplitude-phase-form
        '''

        # beam pattern function for one arm detector, see Maggiore
        F = [np.cos(self.theta_S_Ec)**2*np.cos(self.phi_rel)**2-np.sin(self.phi_rel)**2,2*np.cos(self.theta_S_Ec)*np.sin(self.phi_rel)*np.cos(self.phi_rel)]
        
        # add the beam pattern rotation by polarization angle
        return np.array([F[0]*np.cos(2*self.psi_S()) - F[1]*np.sin(2*self.psi_S()), 
                         F[0]*np.sin(2*self.psi_S()) + F[1]*np.cos(2*self.psi_S())])
    
    def nxhxn(self,t=0):
        '''
        Returns
        -------
        h_nn : The strain in direction of the detector
        '''
        return self.amp*np.cos(2*pi*self.freq_int(t) + self.polarization)
    
    def psi_bar(self,t):
        '''
        Returns
        ------
        psi_bar : The strain as reported by Armstrong
        '''
        return self.nxhxn(t)/(1-(self.kn)**2)

    def d_psi_bar(self,t):
        '''
        Returns
        -------
        A derivative d psi_bar / d lambda = d_psi_bar * d (2*pi*freq_int + polarization) / d lambda - if there is no dependance of the amplitude on lambda 
        '''
        return -self.amp*np.sin(2*pi*self.freq_int(t) + self.polarization)/(1-(self.kn)**2)
    
    def freq(self,t):
        '''
        Returns
        -------
        The frequency of the GW emitted by the binary with an exoplanetary doppler shift
        '''
        return (self.f_GW + self.f_1*t)*(1 - self.K*np.cos(2*pi*t/self.P + self.phi_0))
    
    def polarization_phase(self):
        '''
        Returns
        -------
        polarization : The residual phase shift of the strain in amplitude-phase form
        '''
        vec = self.gw_amplitude*self.F_ig()
        self.polarization = np.arctan2(-vec[1],vec[0])
        return self.polarization

    def A(self):
        '''
        Returns
        -------
        The amplitude of the GW signal
        '''
        self.amp = np.linalg.norm(self.gw_amplitude*self.F_ig())
        return self.amp
    
    def freq_int(self,t):
        '''
        Returns
        -------
        Analytical Integral 0 to t f(t') dt' with f(t) = self.freq(t)
        '''
        return (2*t*(2*self.f_GW + self.f_1*t) + (self.K*self.P*(self.f_1*self.P*np.cos(self.phi_0) - self.f_1*self.P*np.cos((2*np.pi*t)/self.P + self.phi_0) + 2*self.f_GW*np.pi*np.sin(self.phi_0) - 2*np.pi*(self.f_GW + self.f_1*t)*np.sin((2*np.pi*t)/self.P + self.phi_0)))/np.pi**2)/4.

    def strain(self,t):
        '''
        Returns
        -------
        y(t) as specified by Armstrong
        '''
        return (self.kn-1)/2*self.psi_bar(t) - self.kn*self.psi_bar(t-(1+self.kn)/2*self.T_2) + (1+self.kn)/2*self.psi_bar(t-self.T_2)
        
    def d_freq_int(self,t,i):
        '''
        Returns
        -------
        The derivative d freq_int / d lambda_i where:

        i |  lambda_i
        ------------
        0 |  f_0
        1 |  f_1
        2 |  K
        3 |  P
        4 |  phi_0

        Verified with Differentiation2.nb and Analytical_Frequencies_IG.txt
        '''
        if i == 0:
            return t - (self.K*self.P*np.cos((np.pi*t)/self.P + self.phi_0)*np.sin((np.pi*t)/self.P))/np.pi
        if i == 1:
            return t**2/2. - (self.K*self.P*(-(self.P*np.cos(self.phi_0)) + self.P*np.cos((2*np.pi*t)/self.P + self.phi_0) + 2*np.pi*t*np.sin((2*np.pi*t)/self.P + self.phi_0)))/(4.*np.pi**2)
        if i == 2:
            return (self.P*(self.f_1*self.P*np.cos(self.phi_0) - self.f_1*self.P*np.cos((2*np.pi*t)/self.P + self.phi_0) + 2*self.f_GW*np.pi*np.sin(self.phi_0) - 2*np.pi*(self.f_GW + self.f_1*t)*np.sin((2*np.pi*t)/self.P + self.phi_0)))/(4.*np.pi**2)
        if i == 3:
            return (self.K*(self.f_1*self.P**2*np.cos(self.phi_0) + (-(self.f_1*self.P**2) + 2*self.f_GW*np.pi**2*t + 2*self.f_1*np.pi**2*t**2)*np.cos((2*np.pi*t)/self.P + self.phi_0) + self.P*np.pi*(self.f_GW*np.sin(self.phi_0) - (self.f_GW + 2*self.f_1*t)*np.sin((2*np.pi*t)/self.P + self.phi_0))))/(2.*self.P*np.pi**2)
        if i == 4:
            return (self.K*self.P*(2*self.f_GW*np.pi*np.cos(self.phi_0) - 2*np.pi*(self.f_GW + self.f_1*t)*np.cos((2*np.pi*t)/self.P + self.phi_0) + self.f_1*self.P*(-np.sin(self.phi_0) + np.sin((2*np.pi*t)/self.P + self.phi_0))))/(4.*np.pi**2)
        else:
            print('Error in d_freq_int')
            return self.strain(t)
        
    def d_strain(self,i,bina=None,diff=1e-6): # 0: K, 1: P, 2: phi_0, 3: theta_S_Ec, 4: phi_S_Ec, 5: theta_L, 6: phi_L, 7: ln(A), 8: f_1, 9: f_0
        '''
        Returns
        -------
        The function d y(t) / d lambda_i, where:

        i |  lambda_i
        ------------
        0 |  K
        1 |  P
        2 |  phi_0
        3 |  theta_S_Ec
        4 |  phi__S_Ec
        5 |  theta_L
        6 |  phi_L
        7 |  ln(A)
        8 |  f_1
        9 |  f_0

        and derivatives 3 .. 6 have to be performed numerically via an instance of bina, which can be set up via self.h_i()
        '''
        if bina is not None:
            return lambda t: (bina.strain(t)-self.strain(t))/diff
        if i <= 2:
            return lambda t: (self.kn-1)/2*2*pi*self.d_freq_int(t,i+2)*self.d_psi_bar(t) - self.kn*2*pi*self.d_freq_int(t-(self.kn+1)/2*self.T_2,i+2)*self.d_psi_bar(t-(self.kn+1)/2*self.T_2) + (self.kn+1)/2*2*pi*self.d_freq_int(t-self.T_2,i+2)*self.d_psi_bar(t-self.T_2)
        if i == 7:
            return lambda t: self.strain(t)
        if i == 8:
            return lambda t: (self.kn-1)/2*2*pi*self.d_freq_int(t,1)*self.d_psi_bar(t) - self.kn*2*pi*self.d_freq_int(t-(self.kn+1)/2*self.T_2,1)*self.d_psi_bar(t-(self.kn+1)/2*self.T_2) + (self.kn+1)/2*2*pi*self.d_freq_int(t-self.T_2,1)*self.d_psi_bar(t-self.T_2)
        if i == 9:
            return lambda t: (self.kn-1)/2*2*pi*self.d_freq_int(t,0)*self.d_psi_bar(t) - self.kn*2*pi*self.d_freq_int(t-(self.kn+1)/2*self.T_2,0)*self.d_psi_bar(t-(self.kn+1)/2*self.T_2) + (self.kn+1)/2*2*pi*self.d_freq_int(t-self.T_2,0)*self.d_psi_bar(t-self.T_2)
        
        print('Error in d_strain')
        raise ValueError
        
    def h_ixh_j(self,i,j,diff=1e-6):
        '''
        Returns
        -------
        The relevant integral for the Fisher matrix: $\I dy/dlambda_i(t) \cdot dy/dlambda_j(t) dt$, where

        i |  lambda_i
        ------------
        0 |  K
        1 |  P
        2 |  phi_0
        3 |  theta_S_Ec
        4 |  phi__S_Ec
        5 |  theta_L
        6 |  phi_L
        7 |  ln(A)
        8 |  f_1
        9 |  f_0
        '''
        print('Integrating dh/d{} * dh/d{} ...\n'.format(labels[i],labels[j]))
        if i == j:
            funcA = self.h_i(i,diff)
            return self.integral_over_obs(funcA,None)
        if i != j:
            funcA = self.h_i(i,diff)
            funcB = self.h_i(j,diff)
            return self.integral_over_obs(funcA,funcB)

    def integral_over_obs(self,funcA,funcB):
        '''
        Returns
        -------
        The integral over the observational time for the ice giant mission (40 days each year for 10 years) of one or two functions:
        \I funcA * funcB dt or \I funcA^2 dt

        Parameters
        ----------
        funcA : function (t in sec)
        funcB : function (t in sec) or None
        If None, compute \I funcA^2 dt, else compute \I funcA * funcB dt
        A scaling for the epsabs if we are on the off-diagonal of the Fisher-matrix, where sometimes parameters are weekly correlated, i.e. we expect Int ~ 0
        '''
        res = 0.
        if funcB is not None:
            for i, start in enumerate(np.linspace(0,10,10)*yr):
                temp_res = integrate.quad(lambda t: funcA(t)*funcB(t),start,start + 40*day,limit=int(40*day*self.f_GW*50),epsrel=self.epsrel,epsabs=0)
                res += temp_res[0]
                print('Integral = {:.3e} +- {:.3e}, rel. error: {:.3f}'.format(temp_res[0],temp_res[1],abs(temp_res[1]/temp_res[0])))
            print('\n')
            return res
        else:
            for i, start in enumerate(np.linspace(0,10,10)*yr):
                temp_res = integrate.quad(lambda t: (funcA(t))**2,start,start + 40*day,limit=int(40*day*self.f_GW*50),epsrel=self.epsrel,epsabs=0)
                res += temp_res[0]
                print('Integral = {:.3e} +- {:.3e}, rel. error: {:.3f}'.format(temp_res[0],temp_res[1],abs(temp_res[1]/temp_res[0])))
            print('\n')
            return res

    def h_i(self,i,diff=1e-6):
        '''
        Returns
        -------
        A function d y / d lambda_i(t), where:

        i |  lambda_i
        ------------
        0 |  K
        1 |  P
        2 |  phi_0
        3 |  theta_S_Ec
        4 |  phi_S_Ec
        5 |  theta_L
        6 |  phi_L
        7 |  ln(A)
        8 |  f_1
        9 |  f_0

        and derivatives 3 .. 6 will be performed numerically with a setup of a close binary system in parameter space
        '''
        B = None
        if i == 3:
            if self.theta_S_Ec+diff > pi:
                # we care for an overflow in theta_S_Ec and just swap to the left-sided derivative
                diff *= -1
            B = ig_binary(self.theta_S_Ec+diff,self.phi_S_Ec,self.dist/pc,self.theta_L,self.phi_L,self.m1/M_S,self.m2/M_S,self.f_GW,self.mP/M_J,self.P/yr,self.theta_P,self.phi_0,'m')
        elif i == 4:
            B = ig_binary(self.theta_S_Ec,self.phi_S_Ec+diff,self.dist/pc,self.theta_L,self.phi_L,self.m1/M_S,self.m2/M_S,self.f_GW,self.mP/M_J,self.P/yr,self.theta_P,self.phi_0,'m')
        elif i == 5:
            if self.theta_L+diff > pi:
                # we care for an overflow in theta_S_Ec and just swap to the left-sided derivative
                diff *= -1
            B = ig_binary(self.theta_S_Ec,self.phi_S_Ec,self.dist/pc,self.theta_L+diff,self.phi_L,self.m1/M_S,self.m2/M_S,self.f_GW,self.mP/M_J,self.P/yr,self.theta_P,self.phi_0,'m')
        elif i == 6:
            B = ig_binary(self.theta_S_Ec,self.phi_S_Ec,self.dist/pc,self.theta_L,self.phi_L+diff,self.m1/M_S,self.m2/M_S,self.f_GW,self.mP/M_J,self.P/yr,self.theta_P,self.phi_0,'m')
            
        return self.d_strain(i,B,diff)

    def Fisher_mat(self,diff=1e-6):
        '''
        Returns
        -------
        The Fisher matrix Gamma[i,j] = $\I_{T_{obs}} dy/d\lambda_i cdot dy/d\lambda_j dt$ in dependance of the mode:

        mode |  Fisher matrix
        ---------------------
        's'  |  (3,3) matrix of only exoplanet parameters: lambda = (K, P, phi_0)
        'm'  |  (9,9) matrix of the uncertain parameters: lambda = (K, P, phi_0, theta_S_Ec, phi_S_Ec, theta_L, phi_L, ln(A), f_1)
        'l'  |  (10,10) matrix of all parameters: lambda = (K, P, phi_0, theta_S_Ec, phi_S_Ec, theta_L, phi_L, ln(A), f_1, f_0)

        Also writes the Fisher matrix into self.Fisher
        '''

        # First, if we didn't already compute the Fisher matrix in this binaries_ig instance, compute it
        if not hasattr(self,'Fisher'):
            # Then, look if we already performed this computation in dict_binaries_ig.txt
            sim, js, mode_sim = self.look_for_similar()
            if sim:
                self.snr = js['SNR']

            size = mode_size[self.mode]
            self.Fisher = np.zeros((size,size),np.float64)
            for i in np.arange(size):
                if sim:
                    if i < mode_size[mode_sim]:
                        self.Fisher[i][i] = js['Fisher'][i,i]
                        continue
                self.Fisher[i][i] = self.h_ixh_j(i, i,diff)
            for i in np.arange(size):
                for j in np.arange(i+1,size):
                    if sim:
                        if j < mode_size[mode_sim]:
                            self.Fisher[i][j] = js['Fisher'][i,j]
                            self.Fisher[j][i] = self.Fisher[i][j]
                            continue
                    self.Fisher[i][j] = self.h_ixh_j(i, j,diff)
                    self.Fisher[j][i] = self.Fisher[i][j]

            self.Fisher *= 2/self.S_n() # for signal to noise with S_n = 1 as it is irrelevant
            self.Fisher *= 2 #Account for two measurements / two missions approximately
        return self.Fisher
    
    def rel_uncertainty(self,diff=1e-6):
        '''
        TODO: WHY DO WE HAVE THIS?
        Returns
        -------
        [sigma_K/K, sigma_P/P, sigma_phi] The relative uncertainties of the exoplanet parameters of interest, cf. Tamanini (2020)

        Parameters
        ----------
        diff : float 
        The difference for the numerical differentiation (y(par + diff) - y(par)) / diff
        '''
        # Check if already saved in dict_binaries_ig.txt
        js = self.add_json()

        try:
            unc = np.sqrt(np.diag(np.linalg.inv(self.Fisher_mat(diff))[:3]))
            self.add_json()
            return unc/np.array([self.K,self.P,1.])
        except:
            print('Error in sqrt! Please try again with higher epsrel or epsabs')
            unc = np.sqrt(np.abs(np.diag(np.linalg.inv(self.Fisher_mat(diff)))[:3]))
            self.add_json()
            return js['Tamanini_plot']
    
    def hxh(self):
        '''
        Returns
        -------
        The integrated response $\I_T_obs y \cdot y dt$
        '''
        return self.integral_over_obs(lambda t: self.strain(t),None)
    
    def signal_to_noise(self):
        '''
        Returns
        -------
        The signal-to-noise S/N of the binary over the whole observational time
        '''
        if not hasattr(self, 'snr'):
            self.snr = np.sqrt(2*self.hxh()/self.S_n()*2) # x2 for two spacecrafts
        return self.snr
    
    def json(self):
        '''
        Returns
        -------
        The important calculated properties of the binary system which are to be saved in dict_binaries_ig.txt
        json = {'Fisher' : Fisher matrix (mode),
                'Error' : Inverse of the Fisher matrix, i.e. the expected variances,
                'pars' : {label : Relevant parameters of the binary system},
                'correlation' : Correlation matrix of the parameters, i.e. Errors[i,j]/(lambda_i * lambda_j),
                'rel_unc' : Relative Uncertainties on the parameters, i.e. diagonal(Correlation),
                'exo_rel_unc' : Relative Uncertainties on the exoplanet parameters (K, P, phi_0), i.e. diagonal(Correlation)[:3],
                'SNR' : S/N,
                'binary' : self
                'Tamanini_plot' : Rescaled Relative Uncertainties on the exoplanet parameters (K, P, phi_0) as in Tamanini (2020), i.e. Uncertainty * S/N * M_P/M_J
                }
        '''
        if not hasattr(self,'js'): # Compute it!
            self.Fisher_mat()
            err = np.array(mpmath.inverse(self.Fisher).tolist(),dtype=np.float64)
            if np.linalg.det(self.Fisher) == 0:
                print('Parameters can´t be constrained. Degeneracy detected.')
            if np.sum(np.diag(err) < 0) > 0: # any negative variances
                print('Negative variances detected! Please check again the integrals and retry them with higher accuracy. Negative variances for:')
                print(np.array(labels[:mode_size[self.mode]])[np.diag(err) < 0])
            bin_labels = ['K','P','phi_0', 'theta_S_Ec', 'phi_S_Ec', 'theta_L', 'phi_L', 'ln(A)', 'f_1', 'f_0','M_P','M_c','dist']
            vals = [self.K,self.P/yr,self.phi_0,self.theta_S_Ec,self.phi_S_Ec,self.theta_L,self.phi_L,np.log(self.a0),self.f_1,self.f_GW,self.mP/M_J,self.chirp_mass,self.dist/pc]
            if self.mode == 's':
                vals2 = [self.K,self.P,1.]
            if self.mode == 'm':
                vals2 = [self.K,self.P,1.,1.,1.,1.,1.,np.log(self.a0),self.f_1]
            if self.mode == 'l':
                vals2 = [self.K,self.P,1.,1.,1.,1.,1.,np.log(self.a0),self.f_1,self.f_GW]
            rel_unc = err / np.outer(vals2,vals2)
            self.js = {'Fisher' : self.Fisher,
                        'Error' : err,
                        'pars' : {label : vals[i] for i,label in enumerate(bin_labels)},
                        'correlation' : rel_unc,
                        'rel_unc' : np.sqrt(np.abs(np.diag(rel_unc))),
                        'exo_rel_unc' : np.sqrt(np.abs(np.diag(rel_unc)))[:3],
                        'SNR' : self.signal_to_noise(),
                        'binary' : self
                        }
            self.js['Tamanini_plot'] = self.js['SNR']*self.js['pars']['M_P']*self.js['exo_rel_unc']
        return self.js
            
        
    def add_json(self):
        '''
        Returns
        -------
        The important calculated properties of the binary system which are to be saved in dict_binaries_ig.txt
        json = {'Fisher' : Fisher matrix (mode),
                'Error' : Inverse of the Fisher matrix, i.e. the expected variances,
                'pars' : {label : Relevant parameters of the binary system},
                'correlation' : Correlation matrix of the parameters, i.e. Errors[i,j]/(lambda_i * lambda_j),
                'rel_unc' : Relative Uncertainties on the parameters, i.e. diagonal(Correlation),
                'exo_rel_unc' : Relative Uncertainties on the exoplanet parameters (K, P, phi_0), i.e. diagonal(Correlation)[:3],
                'SNR' : S/N,
                'binary' : self
                'Tamanini_plot' : Rescaled Relative Uncertainties on the exoplanet parameters (K, P, phi_0) as in Tamanini (2020), i.e. Uncertainty * S/N * M_P/M_J
                }

        Also saves this json pickled into 'dict_binaries_ig.txt if it is not already in there
        '''
        # Look if we already computed this
        in_dict, js = self.in_dict_bin()

        if not in_dict:
            if not hasattr(self,'js'):
                self.json()

            # load the saved
            infile = open('dict_binaries_ig.txt','rb')
            dict_binaries = pickle.load(infile)
            infile.close()
            dict_binaries[self.mode].append(self.js)

            # save the appended dict_binaries_ig
            outfile = open('dict_binaries_ig.txt','wb')
            pickle.dump(dict_binaries,outfile)
            outfile.close()
            return self.js
        else:
            self.Fisher = js['Fisher']
            self.snr = js['SNR']
            return self.json()
    
    def in_dict_bin(self):
        '''
        Returns
        -------
        in_dict_bin : bool
        A binary as this one with the corresponding mode is already in dict_binaries_ig.txt
        js : dict or None
        If it is, then we return the corresponding json, else we return None
        '''
        infile = open('dict_binaries_ig.txt','rb')
        dict_binaries = pickle.load(infile)[self.mode]
        infile.close()
        # Go through all of them and compare, because this step isn't the time intensive one
        for i in dict_binaries:
            pars = i['pars']
            if isclose(self.P/yr, pars['P']) and isclose(self.phi_0, pars['phi_0']) and isclose(self.theta_S_Ec, pars['theta_S_Ec']) and isclose(self.phi_S_Ec, pars['phi_S_Ec']) and isclose(self.theta_L, pars['theta_L']) and isclose(self.phi_L, pars['phi_L']) and isclose(self.mP/M_J, pars['M_P']) and isclose(self.dist/pc,pars['dist']) and isclose(self.f_GW, pars['f_0']) and isclose(self.chirp_mass, pars['M_c']):
                print('Found pickled data in dict_binaries_ig.txt')
                return True, i
        return False, None
    

    def look_for_similar(self):
        '''
        Returns
        -------
        in_dict_bin : bool
        A binary as this one with a different mode is already in dict_binaries_ig.txt
        js : dict or None
        If it is, then we return the corresponding json, else we return None
        mode : character in ['s','m','l']
        The largest mode, where this computation was already performed
        '''

        for mode in ['l','m','s']:
            B2 = self.copy()
            B2.mode = mode
            res = B2.in_dict_bin()
            if res[0]:
                print('Found similiar binary pickled with mode {}. Using it as the basis for mode {}'.format(mode,self.mode))
                return res[0], res[1], mode
        return False, None, None
        
    def create_from_json(json,mode='m'):
        '''
        Returns
        -------
        A binary_ig instance computed from a json
        '''
        pars = json['pars']
        B = ig_binary(theta_S_Ec=pars['theta_S_Ec'],phi_S_Ec=pars['phi_S_Ec'],dist=pars['dist'],theta_L=pars['theta_L'],phi_L=pars['phi_L'],m1=.23,m2=.23,freq=pars['f_0'],mP=pars['M_P'],P=pars['P'],theta_P=pi/2,phi_P=pars['phi_0'],mode=mode)
        B.snr = json['SNR']*B.a0/np.exp(pars['ln(A)'])
        B.Fisher = json['Fisher']
        return B
        
    def copy(self):
        '''
        Returns
        -------
        An ig_binary instance as this one
        '''
        return ig_binary(theta_S_Ec=self.theta_S_Ec,phi_S_Ec=self.phi_S_Ec,dist=self.dist/pc,theta_L=self.theta_L,phi_L=self.phi_L,m1=self.m1/M_S,m2=self.m2/M_S,freq=self.f_GW,mP=self.mP/M_J,P=self.P/yr,theta_P=self.theta_P,phi_P=self.phi_0,mode=self.mode)