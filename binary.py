# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:36:02 2021

@author: marcus
"""
from ssl import match_hostname
from scipy.constants import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import LISA
import pickle
import vegas
import mpmath

mpmath.mp.dps = 50

pi = np.pi
AU = 1495978707e2 # m
pc = 3.0857e16 # 2e5 AU
yr = 24*3600*365.25 # s

M_J = 1.898e27 # kg
M_S = 1.989e30 # kg
r_S = 2*G/c**2 # m

sqth = np.sqrt(3)/2
omega_E = 2*pi/yr

arr_yr = np.linspace(0,yr,1000)
labels = ['K','P','phi_0', 'theta_S_Ec', 'phi_S_Ec', 'theta_L', 'phi_L', 'ln(A)', 'f_1', 'f_0']
keys = enumerate(labels)
mode_size = {'s' : 3, 'm' : 9, 'l' : 10}
ArcTan = lambda x,y: np.arctan2(y,x)
file = 'dict_binaries.txt'
file_full = 'dict_binaries_full.txt'

def isclose(a, b, rel_tol=1e-04, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

class binary:

    def __init__(self,theta_S_Ec=np.arccos(.3),phi_S_Ec=5.,dist=1e3,theta_L=np.arccos(-.2),phi_L=4.,m1=.23,m2=.23,freq=10e-3,mP=1,P=2,theta_P=pi/2,phi_P=pi/2,T_obs=4,mode='m',key=3,epsrel=1e-2,epsabs=1e-2,num=10**6):
        '''
        Parameters
        ----------
        theta_S_Ec : float in [0,pi]
        Inclination angle of the source in ecliptic coordinates in rad
        phi_S_Ec : float in [0,2*pi]
        Azimuth angle of the source in the ecliptic coordinates in rad
        dist : float
        Distance from us to binary system in pc
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
        key : int in [1,2,3]
        Used for plots etc: Indication if we should look at length difference I (1), legth difference II (2) or both (3)
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
        self.T_obs = T_obs*yr #s
        
        assert mode in ['s','m','l']
        self.mode = mode
        self.key = key
        self.both = False
        if key == 3:
            self.both = True
        
        # Set the parameters for numerical integration 
        self.epsrel=epsrel
        self.epsabs=epsabs
        self.num=num
        
        # Compute relevant parameters of out binary system
        self.chirp_mass = (self.m1*self.m2)**(3/5)/((self.m1+self.m2))**(1/5) #kg
        self.f_binary = freq/2
        self.f_GW = freq
        self.f_1 = 96/5*pi**(8/3)*freq**(11/3)*(G*self.chirp_mass/c**3)**(5/3)

        self.cos_inclination = np.cos(self.theta_L)*np.cos(self.theta_S_Ec)+np.sin(self.theta_L)*np.sin(self.theta_S_Ec)*np.cos(self.phi_L-self.phi_S_Ec) # = L*n
        self.a0 = 4/self.dist*(G*self.chirp_mass/c**2)**(5/3)*(pi*self.f_GW/c)**(2/3)
        self.gw_amplitude = self.a0*np.array([(1.+self.cos_inclination**2)/2, -self.cos_inclination]) # [A_plus, A_cross]
    
    def sep(self):
        '''
        Returns the seperation for a given binary system in m
        '''
        return (G*(self.m1+self.m2)/(pi*self.f_GW)**2)**(1/6) #m
    
    def S_n(self):
        '''
        Compute the strain sensitivity - stolen from Travis Robson et al 2019 Class. Quantum Grav. 36 105011
        '''
        return LISA.LISA().Sn(self.f_GW)
    
    def r_s(self,m):
        '''
        Returns
        -------
        r_s : The length of the Schwarzschild radius in m.
        '''
        return m*r_S
    
    def F_LISA_det_frame(self,cos_theta_S,phi_S,psi_S):
        '''
        Returns
        -------
        [F_plus, F_cross] : LISA's beam pattern function rotated to the principal axis for a pi/2 phase difference of + and x polarizations, see Cutler
        '''
        Term_A = .5*(1.+cos_theta_S**2)*np.cos(2*phi_S)
        Term_B = cos_theta_S*np.sin(2*phi_S)
        return np.array([Term_A*np.cos(2*psi_S) - Term_B*np.sin(2*psi_S),
                         Term_A*np.sin(2*psi_S) + Term_B*np.cos(2*psi_S)])
    
    def cos_theta_S(self,t=0):
        '''
        Returns
        -------
        cos_theta_S : The cos(theta_S) in the detector frame = <n,z>, see Cutler
        '''
        return .5*np.cos(self.theta_S_Ec) - sqth*np.sin(self.theta_S_Ec)*np.cos(omega_E*t-self.phi_S_Ec)
    
    def phi_S(self,t=0):
        '''
        Returns
        -------
        phi_S_det : The azimuth angle phi_S of the source in the detector frame
        '''
        arctan_term = np.arctan2((np.sqrt(3)*np.cos(self.theta_S_Ec)+np.sin(self.theta_S_Ec)*np.cos(omega_E*t-self.phi_S_Ec)),(2*np.sin(self.theta_S_Ec)*np.sin(omega_E*t-self.phi_S_Ec)))
        return (omega_E*t + arctan_term) % (2*pi)
    
    def psi_S(self,t=0):
        '''
        Returns
        -------
        psi_S : The polarization angle psi_S of the wavefront in the detector frame
        '''
        Lz = .5*np.cos(self.theta_L)-sqth*np.sin(self.theta_L)*np.cos(omega_E*t-self.phi_L)
        Ln = self.cos_inclination
        nLxz = .5*np.sin(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_L-self.phi_S_Ec) - sqth*np.cos(omega_E*t)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec)-np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)) - sqth*np.sin(omega_E*t)*(np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.cos(self.phi_L) - np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.cos(self.phi_S_Ec))
        return np.arctan2((Lz - Ln*self.cos_theta_S(t)),nLxz)
    
    def F_LISA(self,t=0):
        '''
        Returns
        -------
        [F_plus(t), F_cross(t)] : The time dependant beam pattern function due to LISA's orbit around the sun
        '''
        return self.F_LISA_det_frame(self.cos_theta_S(t),self.phi_S(t)-pi/4*(self.key-1),self.psi_S(t))
        
    def A(self,t=0):
        '''
        Returns
        -------
        A(t) : The amplitude of our binary in LISA in the common amplitude-phase-form
        '''
        return np.linalg.norm(self.gw_amplitude*self.F_LISA(t))
    
    def polarization_phase(self,t=0):
        '''
        Returns
        -------
        phi_p(t) : The polarization phase of the binary in LISA
        '''
        vec = self.gw_amplitude*self.F_LISA(t)
        return np.arctan2(-vec[1],vec[0])
    
    def doppler_phase(self,t=0):
        '''
        Returns
        -------
        phi_D(t) : The doppler phase of the binary in LISA
        '''
        return 2*pi*self.freq(t)/c*AU*np.sin(self.theta_S_Ec)*np.cos(omega_E*t-self.phi_S_Ec)
    
    def freq(self,t):
        '''
        Returns
        -------
        The frequency of the GW emitted by the binary with an exoplanetary doppler shift
        '''
        return (self.f_GW + self.f_1*t)*(1 - self.K*np.cos(2*pi*t/self.P + self.phi_0))
    
    def freq_int(self,t):
        '''
        Returns
        -------
        Analytical Integral 0 to t f(t') dt' with f(t) = self.freq(t)
        '''
        return (2*t*(2*self.f_GW + self.f_1*t) + (self.K*self.P*(self.f_1*self.P*np.cos(self.phi_0) - self.f_1*self.P*np.cos((2*pi*t)/self.P + self.phi_0) + 2*self.f_GW*pi*np.sin(self.phi_0) - 2*pi*(self.f_GW + self.f_1*t)*np.sin((2*pi*t)/self.P + self.phi_0)))/pi**2)/4.
    
    def strain(self,t):
        '''
        Returns
        -------
        h(t) as specified by Cutler
        '''
        return sqth*self.A(t)*np.cos(2*pi*self.freq_int(t)+self.polarization_phase(t)+self.doppler_phase(t))

    def d_freq(self,t,i): 
        '''
        TODO
        Returns
        -------
        The derivative d f(t) / d lambda_i where:

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
            return 1 - self.K*np.cos((2*pi*t)/self.P + self.phi_0)
        if i == 1:
            return t*(1 - self.K*np.cos((2*pi*t)/self.P + self.phi_0))
        if i == 2:
            return -((self.f_GW + self.f_1*t)*np.cos((2*pi*t)/self.P + self.phi_0))
        if i == 3:
            return (-2*self.K*pi*t*(self.f_GW + self.f_1*t)*np.sin((2*pi*t)/self.P + self.phi_0))/self.P**2
        if i == 4:
            return self.K*(self.f_GW + self.f_1*t)*np.sin((2*pi*t)/self.P + self.phi_0)

        else:
            print('Error in d_freq')
            return self.strain(t)
        
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
            return t - (self.K*self.P*np.cos((pi*t)/self.P + self.phi_0)*np.sin((pi*t)/self.P))/pi
        if i == 1:
            return t**2/2. - (self.K*self.P*(-(self.P*np.cos(self.phi_0)) + self.P*np.cos((2*pi*t)/self.P + self.phi_0) + 2*pi*t*np.sin((2*pi*t)/self.P + self.phi_0)))/(4.*pi**2)
        if i == 2:
            return (self.P*(self.f_1*self.P*np.cos(self.phi_0) - self.f_1*self.P*np.cos((2*pi*t)/self.P + self.phi_0) + 2*self.f_GW*pi*np.sin(self.phi_0) - 2*pi*(self.f_GW + self.f_1*t)*np.sin((2*pi*t)/self.P + self.phi_0)))/(4.*pi**2)
        if i == 3:
            return (self.K*(self.f_1*self.P**2*np.cos(self.phi_0) + (-(self.f_1*self.P**2) + 2*self.f_GW*pi**2*t + 2*self.f_1*pi**2*t**2)*np.cos((2*pi*t)/self.P + self.phi_0) + self.P*pi*(self.f_GW*np.sin(self.phi_0) - (self.f_GW + 2*self.f_1*t)*np.sin((2*pi*t)/self.P + self.phi_0))))/(2.*self.P*pi**2)
        if i == 4:
            return (self.K*self.P*(2*self.f_GW*pi*np.cos(self.phi_0) - 2*pi*(self.f_GW + self.f_1*t)*np.cos((2*pi*t)/self.P + self.phi_0) + self.f_1*self.P*(-np.sin(self.phi_0) + np.sin((2*pi*t)/self.P + self.phi_0))))/(4.*pi**2)

        else:
            print('Error in d_freq_int')
            return self.strain(t)
        
    def d_strain(self,i,bina=None,diff=1e-6):         
        '''
        Returns
        -------
        d h / d lambda_i, where:

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
            return lambda t: -sqth*self.A(t)*(2*pi*self.d_freq_int(t,i+2) + 2*pi*self.d_freq(t,i+2)/c*AU*np.sin(self.theta_S_Ec)*np.cos(omega_E*t-self.phi_S_Ec))*np.sin(2*pi*self.freq_int(t)+self.polarization_phase(t)+self.doppler_phase(t))
        if i == 7:
            return lambda t: self.strain(t)
        if i == 8:
            return lambda t: -sqth*self.A(t)*(2*pi*self.d_freq_int(t,1) + 2*pi*self.d_freq(t,1)/c*AU*np.sin(self.theta_S_Ec)*np.cos(omega_E*t-self.phi_S_Ec))*np.sin(2*pi*self.freq_int(t)+self.polarization_phase(t)+self.doppler_phase(t))
        if i == 9:
            return lambda t: -sqth*self.A(t)*(2*pi*self.d_freq_int(t,0) + 2*pi*self.d_freq(t,0)/c*AU*np.sin(self.theta_S_Ec)*np.cos(omega_E*t-self.phi_S_Ec))*np.sin(2*pi*self.freq_int(t)+self.polarization_phase(t)+self.doppler_phase(t))
        print('Error in d_strain')
        raise ValueError
        
    def h_ixh_j(self,i,j,diff=1e-6):
        '''
        Returns
        -------
        The relevant integral for the Fisher matrix: $\I dh/dlambda_i(t) \cdot dh/dlambda_j(t) dt$, where

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

        The integrals are performed via MC integration (vegas)
        '''
        print('Integrating dh/d{} * dh/d{} ...\n'.format(labels[i],labels[j]))
        if i == j:
            funcA = self.h_i(i,diff)
            result = integrate.quad(lambda t: (funcA(t))**2,0,self.T_obs,epsrel=self.epsrel,epsabs=0,limit=10**6)
            print('Integral = {:.3e} +- {:.3e}, rel. uncertainty = {:.3f} \n'.format(result[0], result[1], result[1]/result[0]))
            return result[0]
        if i != j:
            funcA = self.h_i(i,diff)
            funcB = self.h_i(j,diff)
            result = integrate.quad(lambda t: funcA(t)*funcB(t),0,self.T_obs,epsrel=self.epsrel,epsabs=0,limit=self.num)
            print('\nIntegral = {:.3e} +- {:.3e}, rel. uncertainty = {:.3f} \n'.format(result[0], result[1], result[1]/abs(result[0])))
            return result[0]

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
            B = binary(self.theta_S_Ec+diff,self.phi_S_Ec,self.dist/pc,self.theta_L,self.phi_L,self.m1/M_S,self.m2/M_S,self.f_GW,self.mP/M_J,self.P/yr,self.theta_P,self.phi_0,self.T_obs/yr,'m',key=self.key)
        elif i == 4:
            B = binary(self.theta_S_Ec,self.phi_S_Ec+diff,self.dist/pc,self.theta_L,self.phi_L,self.m1/M_S,self.m2/M_S,self.f_GW,self.mP/M_J,self.P/yr,self.theta_P,self.phi_0,self.T_obs/yr,'m',key=self.key)
        elif i == 5:
            if self.theta_L+diff > pi:
                # we care for an overflow in theta_S_Ec and just swap to the left-sided derivative
                diff *= -1
            B = binary(self.theta_S_Ec,self.phi_S_Ec,self.dist/pc,self.theta_L+diff,self.phi_L,self.m1/M_S,self.m2/M_S,self.f_GW,self.mP/M_J,self.P/yr,self.theta_P,self.phi_0,self.T_obs/yr,'m',key=self.key)
        elif i == 6:
            B = binary(self.theta_S_Ec,self.phi_S_Ec,self.dist/pc,self.theta_L,self.phi_L+diff,self.m1/M_S,self.m2/M_S,self.f_GW,self.mP/M_J,self.P/yr,self.theta_P,self.phi_0,self.T_obs/yr,'m',key=self.key)
            
        return self.d_strain(i,B,diff)

    def Fisher_mat(self,diff=1e-6,both=False):
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
        # If we already have a Fisher matrix, just keep it
        if not hasattr(self,'Fisher'):
            # If we already computed the same binary in a different mode, use it as a basis
            sim, js, mode_sim = self.look_for_similar()
            if sim:
                self.snr = js['SNR']

            size = mode_size[self.mode]
            self.Fishers = np.zeros((2,size,size),np.float64)
            if self.both:
                for key in [1,2]:
                    self.key = key
                    sim = False
                    if key == 1:
                        pass
                        #sim, js, mode_sim = self.look_for_similar()
                    print('\n\n>>>  Now working on signal {} ...  <<<\n\n'.format(key))
                    for i in np.arange(size): 
                        if sim and (i < mode_size[mode_sim]):
                            self.Fishers[0][i][i] = js['Fisher'][i,i]
                            continue
                        self.Fishers[key-1][i][i] = self.h_ixh_j(i, i,diff)
                    for i in np.arange(size):
                        for j in np.arange(i+1,size):
                            if sim and (j < mode_size[mode_sim]):
                                self.Fisher[i][j] = js['Fisher'][i,j]
                                self.Fisher[j][i] = js['Fisher'][j,i]
                                continue
                            self.Fishers[key-1][i][j] = self.h_ixh_j(i, j,diff)
                            self.Fishers[key-1][j][i] = self.Fishers[key-1][i][j]
            else:
                self.key = 1
                for i in np.arange(size):
                    if sim and (i < mode_size[mode_sim]):
                        self.Fishers[0][i][i] = js['Fisher'][i,i]
                        continue
                    self.Fishers[0][i][i] = self.h_ixh_j(i, i,diff)
                for i in np.arange(size):
                    for j in np.arange(i+1,size):
                        if sim and (j < mode_size[mode_sim]):
                            self.Fisher[i][j] = js['Fisher'][i,j]
                            self.Fisher[j][i] = js['Fisher'][j,i]
                            continue
                        self.Fishers[0][i][j] = self.h_ixh_j(i, j,diff)
                        self.Fishers[0][j][i] = self.Fishers[0][i][j]
                self.Fishers[0] *= 2 # account for two measurements

            self.Fisher = self.Fishers[0] + self.Fishers[1]
            self.Fisher *= 2./self.S_n() #2*mat/S_n
        return self.Fisher
    
    def rel_uncertainty(self):
        '''
        Returns
        -------
        [sigma_K/K, sigma_P/P, sigma_phi] The relative uncertainties of the exoplanet parameters of interest, cf. Tamanini (2020)
        '''

        js = self.add_json()
        try:
            unc = np.sqrt(np.diag(np.linalg.inv(self.Fisher)[:3]))
        except:
            print('Error in sqrt! Please try again with higher epsrel or epsabs')
            unc = np.sqrt(np.abs(np.diag(np.linalg.inv(self.Fisher_mat(diff)))[:3]))
        return unc/np.array([self.K,self.P,1.])
    
    def hxh(self):
        '''
        Returns
        -------
        The integral \I_T h(t)^2 dt needed for the SNR computation
        Parameters
        ----------
        both : bool 
        If the computation should be performed for both arms individually or just approximated as sqrt(2) * Integral over 1 arm
        '''
        res = 0.
        if self.both:
            for key in [1,2]:
                self.key = key
                res += integrate.quad(lambda t: self.strain(t)**2,0,self.T_obs,limit=int(self.T_obs*self.f_GW*20),epsrel=self.epsrel,epsabs=0)[0]
        else:
            res += integrate.quad(lambda t: (self.strain(t))**2,0,self.T_obs,limit=int(self.T_obs*self.f_GW*20),epsrel=self.epsrel,epsabs=0)[0]
            res *= 2
        return res
    
    def signal_to_noise(self,both=False):
        '''
        Returns
        -------
        The signal-to-noise SNR as typically cited: sqrt[2/S_n(f_0) * Integral h(t)^2 dt]
        '''
        if not hasattr(self, 'snr'):
            self.snr = np.sqrt(2*self.hxh()/self.S_n())
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
        if not hasattr(self, 'Fisher'):
            self.Fisher_mat()
        if not hasattr(self, 'js'):
            err = np.array(mpmath.inverse(self.Fisher).tolist(),dtype=np.float64)
            if np.sum(np.diag(err) < 0) > 0: # any negative variances
                print('Negative variances detected! Please check again the integrals and retry them with higher accuracy. Negative variances for:')
                print(np.array(labels[:mode_size[self.mode]])[np.diag(err) < 0])
            bin_labels = ['K','P','phi_0', 'theta_S_Ec', 'phi_S_Ec', 'theta_L', 'phi_L', 'ln(A)', 'f_1', 'f_0','M_P','M_c','dist']
            vals = [self.K,self.P/yr,self.phi_0,self.theta_S_Ec,self.phi_S_Ec,self.theta_L,self.phi_L,np.log(self.a0),self.f_1,self.f_GW,self.mP/M_J,self.chirp_mass,self.dist/pc]
            if self.mode == 'm':
                vals2 = [self.K,self.P,1.,1.,1.,1.,1.,np.log(self.a0),self.f_1]
            if self.mode == 's':
                vals2 = [self.K,self.P,1.]
            if self.mode == 'l':
                vals2 = [self.K,self.P,1.,1.,1.,1.,1.,np.log(self.a0),self.f_1,self.f_GW]
            rel_unc = err / np.outer(vals2,vals2)
            self.js = {'Fisher' : self.Fisher,'Error' : err,
                         'pars' : {label : vals[i] for i,label in enumerate(bin_labels)},
                         'correlation' : rel_unc,
                         'rel_unc' : np.sqrt(np.diag(rel_unc)),
                         'exo_rel_unc' : np.sqrt(np.diag(rel_unc))[:3],
                         'SNR' : self.signal_to_noise(),
                         'binary' : self
                         }
            if self.both:
                self.js['Fishers I + II'] = self.Fishers
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
        in_dict, js = self.in_dict_bin()

        if not in_dict:
            if not hasattr(self,'js'):
                self.json()

            if self.both:
                infile = open(file_full,'rb')
            else:
                infile = open(file,'rb')
            dict_binaries = pickle.load(infile)
            infile.close()
            dict_binaries[self.mode].append(self.json())

            if self.both:
                outfile = open(file_full,'wb')
            else:
                outfile = open(file,'wb')
            pickle.dump(dict_binaries,outfile)
            outfile.close()
            return self.js
        else:
            self.js = js
            return js
    
    def in_dict_bin(self):
        '''
        Returns
        -------
        in_dict_bin : bool
        A binary as this one with the corresponding mode is already in dict_binaries_ig.txt
        js : dict or None
        If it is, then we return the corresponding json, else we return None
        '''
        if self.both:
            infile = open('dict_binaries_full.txt','rb')
        else:
            infile = open('dict_binaries.txt','rb')
        dict_binaries = pickle.load(infile)[self.mode]
        infile.close()
        for i in dict_binaries:
            pars = i['pars']
            if isclose(self.P/yr, pars['P']) and isclose(self.phi_0, pars['phi_0']) and isclose(self.theta_S_Ec, pars['theta_S_Ec']) and isclose(self.phi_S_Ec, pars['phi_S_Ec']) and isclose(self.theta_L, pars['theta_L']) and isclose(self.phi_L, pars['phi_L']) and isclose(self.mP/M_J, pars['M_P']) and isclose(self.dist/pc,pars['dist']) and isclose(self.f_GW, pars['f_0']) and isclose(self.chirp_mass, pars['M_c']):
                print('Found pickled data in dict_binaries.txt')
                return True, i
        return False, None
    
    def reduced_fisher_mat(self, keys = ['ln(A)'], only_Fisher=True,rescaled_exo=True):
        '''
        Returns
        -------
        The rescaled correlation matrix * S/N * M_P / M_J or the Fisher_matrix without the measurement of the paramaters as specified by keys
        
        Parameters
        ----------
        keys : list of strings
        A list specifiying the parameters to be left out (prior to inversion), allowed ones are:
        ['K', 'P', 'phi_0', 'theta_S_Ec', 'phi_S_Ec', 'theta_L', 'phi_L', 'ln(A)', 'f_1', 'f_0']
        '''
        # Find the corresponding indices
        inds = []
        for key in keys:
            if not key in labels:
                print('No label like {}. Will be ignored.'.format(key))
            inds.append(labels.index(key))
        if self.mode == 'm':
            vals = [self.K,self.P,1.,1.,1.,1.,1.,np.log(self.a0),self.f_1]
        if self.mode == 's':
            vals = [self.K,self.P,1.]
        if self.mode == 'l':
            vals = [self.K,self.P,1.,1.,1.,1.,1.,np.log(self.a0),self.f_1,self.f_GW]

        def delete_mat(mat,i):
            return np.delete(np.delete(mat,i,0),i,1)
        
        # Delete rows and columns
        Fish = self.Fisher_mat()
        inds.sort(reverse=True)
        for i in inds:
            Fish = delete_mat(Fish,i)
            vals.pop(i)

        if only_Fisher:
            return Fish
        else:
            if rescaled_exo:
                print(np.abs(np.diag(np.linalg.inv(Fish) / np.outer(vals,vals))[:3]))
                
                return np.sqrt(np.abs(np.diag(np.linalg.inv(Fish) / np.outer(vals,vals))[:3])) * self.snr * self.mP / M_J
            return np.linalg.inv(Fish) / np.outer(vals,vals)
        
    def print_tamanini(self,delete=['ln(A)']):
        rescaled_exo = self.reduced_fisher_mat(delete,False,True)
        
        plt.loglog(self.P/yr,rescaled_exo[0],'rx')
        plt.loglog(self.P/yr,rescaled_exo[1],'bx')
        plt.loglog(self.P/yr,rescaled_exo[2],'gx')
    
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
        B = binary(theta_S_Ec=pars['theta_S_Ec'],phi_S_Ec=pars['phi_S_Ec'],dist=pars['dist'],theta_L=pars['theta_L'],phi_L=pars['phi_L'],m1=.23,m2=.23,freq=pars['f_0'],mP=pars['M_P'],P=pars['P'],theta_P=pi/2,phi_P=pars['phi_0'],T_obs=4,mode=mode,key=1)
        B.snr = json['SNR']*B.a0/np.exp(pars['ln(A)'])
        B.Fisher = json['Fisher']
        return B
        
    def copy(self):
        '''
        Returns
        -------
        An ig_binary instance as this one
        '''
        return binary(theta_S_Ec=self.theta_S_Ec,phi_S_Ec=self.phi_S_Ec,dist=self.dist/pc,theta_L=self.theta_L,phi_L=self.phi_L,m1=self.m1/M_S,m2=self.m2/M_S,freq=self.f_GW,mP=self.mP/M_J,P=self.P/yr,theta_P=self.theta_P,phi_P=self.phi_0,T_obs=self.T_obs/yr,mode=self.mode,key=self.key)
    
    # Analyical stuff
    
    def strain_analytical(self,t):
        return (np.sqrt(3)*np.cos(2*pi*self.freq_int(t) + ArcTan((self.a0*(1 + (np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))**2)*
   ((np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
            (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
            (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
           np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
            (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2))/2. - 
     (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
      np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
          (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
          (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
         np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
          (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
      np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))))/2.,
self.a0*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))*
 ((np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
      (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
      np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
          (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
          (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
         np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
          (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))))/2. + 
   np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
       np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
    (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
    np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))) + 
(2*AU*self.freq(t)*pi*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/c)*np.sqrt(self.a0**2*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))**2*
((np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
      (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
      np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
          (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
          (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
         np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
          (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))))/2. + 
   np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
       np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
    (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
    np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))**2 + 
(self.a0**2*(1 + (np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))**2)**2*
  ((np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
            (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
            (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
           np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
            (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2))/2. - 
     (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
      np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
          (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
          (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
         np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
          (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
      np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))**2)/4.))/2.
                                                                                            
    def dh_dtheta_S(self,t):
        return         (np.sqrt(3)*np.cos(2*pi*self.freq_int(t) + ArcTan((self.a0*(1 + (np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))**2)*
        ((np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))))/2.,
        self.a0*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))*
        ((np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))))/2. + 
        np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))) + 
        (2*AU*self.freq(t)*pi*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/c)*(2*self.a0**2*(-(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_L))*
        (np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))*
        ((np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))))/2. + 
        np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))**2 + 
        (self.a0**2*(1 + (np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))**2)**2*
        (np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (-0.5*(np.sqrt(3)*np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E)) - np.sin(self.theta_S_Ec)/2.)*(np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.) - 
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
        (((-((np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(-(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_L))) - 
        (-0.5*(np.sqrt(3)*np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E)) - np.sin(self.theta_S_Ec)/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))*
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.))/
        ((np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))**2 + 
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.)**2) + 
        ((-0.5*np.cos(self.theta_L) + (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. + 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))*
        (-0.5*(np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L)*np.sin(self.phi_S_Ec) + np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_S_Ec)*np.cos(self.theta_L)*np.cos(self.phi_S_Ec)) - np.cos(self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.))/
        ((np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))**2 + 
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.)**2))*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))) - 
        2*np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        ((-2*np.sin(self.theta_S_Ec)*(np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E) - np.sqrt(3)*np.sin(self.theta_S_Ec))*np.sin(self.phi_S_Ec - t*omega_E))/
        ((np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))**2 + 4*np.sin(self.theta_S_Ec)**2*np.sin(self.phi_S_Ec - t*omega_E)**2) - 
        (2*np.cos(self.theta_S_Ec)*(-(np.sqrt(3)*np.cos(self.theta_S_Ec)) - np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))*np.sin(self.phi_S_Ec - t*omega_E))/
        ((np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))**2 + 4*np.sin(self.theta_S_Ec)**2*np.sin(self.phi_S_Ec - t*omega_E)**2))*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))) - 
        2*np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        (((-((np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(-(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_L))) - 
        (-0.5*(np.sqrt(3)*np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E)) - np.sin(self.theta_S_Ec)/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))*
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.))/
        ((np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))**2 + 
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.)**2) + 
        ((-0.5*np.cos(self.theta_L) + (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. + 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))*
        (-0.5*(np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L)*np.sin(self.phi_S_Ec) + np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_S_Ec)*np.cos(self.theta_L)*np.cos(self.phi_S_Ec)) - np.cos(self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.))/
        ((np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))**2 + 
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.)**2))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))) - 
        np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
        ((-2*np.sin(self.theta_S_Ec)*(np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E) - np.sqrt(3)*np.sin(self.theta_S_Ec))*np.sin(self.phi_S_Ec - t*omega_E))/
        ((np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))**2 + 4*np.sin(self.theta_S_Ec)**2*np.sin(self.phi_S_Ec - t*omega_E)**2) - 
        (2*np.cos(self.theta_S_Ec)*(-(np.sqrt(3)*np.cos(self.theta_S_Ec)) - np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))*np.sin(self.phi_S_Ec - t*omega_E))/
        ((np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))**2 + 4*np.sin(self.theta_S_Ec)**2*np.sin(self.phi_S_Ec - t*omega_E)**2))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))) - 
        (-0.5*(np.sqrt(3)*np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E)) - np.sin(self.theta_S_Ec)/2.)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))*
        ((np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))))/2. + 
        self.a0**2*(-(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_L))*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))*
        (1 + (np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))**2)*
        ((np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))**2 + 
        2*self.a0**2*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))**2*
        ((np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))))/2. + 
        np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))*
        (np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
        (((-((np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(-(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_L))) - 
        (-0.5*(np.sqrt(3)*np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E)) - np.sin(self.theta_S_Ec)/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))*
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.))/
        ((np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))**2 + 
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.)**2) + 
        ((-0.5*np.cos(self.theta_L) + (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. + 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))*
        (-0.5*(np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L)*np.sin(self.phi_S_Ec) + np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_S_Ec)*np.cos(self.theta_L)*np.cos(self.phi_S_Ec)) - np.cos(self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.))/
        ((np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))**2 + 
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.)**2)) + 
        2*np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        ((-2*np.sin(self.theta_S_Ec)*(np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E) - np.sqrt(3)*np.sin(self.theta_S_Ec))*np.sin(self.phi_S_Ec - t*omega_E))/
        ((np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))**2 + 4*np.sin(self.theta_S_Ec)**2*np.sin(self.phi_S_Ec - t*omega_E)**2) - 
        (2*np.cos(self.theta_S_Ec)*(-(np.sqrt(3)*np.cos(self.theta_S_Ec)) - np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))*np.sin(self.phi_S_Ec - t*omega_E))/
        ((np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))**2 + 4*np.sin(self.theta_S_Ec)**2*np.sin(self.phi_S_Ec - t*omega_E)**2)) + 
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (-0.5*(np.sqrt(3)*np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E)) - np.sin(self.theta_S_Ec)/2.)*(np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))) + 
        np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        (-0.5*(np.sqrt(3)*np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E)) - np.sin(self.theta_S_Ec)/2.)*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))) - 
        2*(np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        (((-((np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(-(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_L))) - 
        (-0.5*(np.sqrt(3)*np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E)) - np.sin(self.theta_S_Ec)/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))*
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.))/
        ((np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))**2 + 
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.)**2) + 
        ((-0.5*np.cos(self.theta_L) + (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. + 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))*
        (-0.5*(np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L)*np.sin(self.phi_S_Ec) + np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_S_Ec)*np.cos(self.theta_L)*np.cos(self.phi_S_Ec)) - np.cos(self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.))/
        ((np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))**2 + 
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.)**2))*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))) - 
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
        ((-2*np.sin(self.theta_S_Ec)*(np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E) - np.sqrt(3)*np.sin(self.theta_S_Ec))*np.sin(self.phi_S_Ec - t*omega_E))/
        ((np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))**2 + 4*np.sin(self.theta_S_Ec)**2*np.sin(self.phi_S_Ec - t*omega_E)**2) - 
        (2*np.cos(self.theta_S_Ec)*(-(np.sqrt(3)*np.cos(self.theta_S_Ec)) - np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))*np.sin(self.phi_S_Ec - t*omega_E))/
        ((np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))**2 + 4*np.sin(self.theta_S_Ec)**2*np.sin(self.phi_S_Ec - t*omega_E)**2))*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))))/(4.*np.sqrt(self.a0**2*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))**2*
        ((np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))))/2. + 
        np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))**2 + 
        (self.a0**2*(1 + (np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))**2)**2*
        ((np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))**2)/4.)) - (np.sqrt(3)*np.sqrt(self.a0**2*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))**2*
        ((np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))))/2. + 
        np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))**2 + 
        (self.a0**2*(1 + (np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))**2)**2*
        ((np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))**2)/4.)*
        ((2*AU*self.freq(t)*pi*np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E))/c - (self.a0*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))*
        ((np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))))/2. + 
        np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))*
        ((self.a0*(1 + (np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))**2)*
        (np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (-0.5*(np.sqrt(3)*np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E)) - np.sin(self.theta_S_Ec)/2.)*(np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.) - 
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
        (((-((np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(-(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_L))) - 
        (-0.5*(np.sqrt(3)*np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E)) - np.sin(self.theta_S_Ec)/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))*
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.))/
        ((np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))**2 + 
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.)**2) + 
        ((-0.5*np.cos(self.theta_L) + (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. + 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))*
        (-0.5*(np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L)*np.sin(self.phi_S_Ec) + np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_S_Ec)*np.cos(self.theta_L)*np.cos(self.phi_S_Ec)) - np.cos(self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.))/
        ((np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))**2 + 
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.)**2))*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))) - 
        2*np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        ((-2*np.sin(self.theta_S_Ec)*(np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E) - np.sqrt(3)*np.sin(self.theta_S_Ec))*np.sin(self.phi_S_Ec - t*omega_E))/
        ((np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))**2 + 4*np.sin(self.theta_S_Ec)**2*np.sin(self.phi_S_Ec - t*omega_E)**2) - 
        (2*np.cos(self.theta_S_Ec)*(-(np.sqrt(3)*np.cos(self.theta_S_Ec)) - np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))*np.sin(self.phi_S_Ec - t*omega_E))/
        ((np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))**2 + 4*np.sin(self.theta_S_Ec)**2*np.sin(self.phi_S_Ec - t*omega_E)**2))*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))) - 
        2*np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        (((-((np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(-(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_L))) - 
        (-0.5*(np.sqrt(3)*np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E)) - np.sin(self.theta_S_Ec)/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))*
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.))/
        ((np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))**2 + 
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.)**2) + 
        ((-0.5*np.cos(self.theta_L) + (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. + 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))*
        (-0.5*(np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L)*np.sin(self.phi_S_Ec) + np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_S_Ec)*np.cos(self.theta_L)*np.cos(self.phi_S_Ec)) - np.cos(self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.))/
        ((np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))**2 + 
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.)**2))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))) - 
        np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
        ((-2*np.sin(self.theta_S_Ec)*(np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E) - np.sqrt(3)*np.sin(self.theta_S_Ec))*np.sin(self.phi_S_Ec - t*omega_E))/
        ((np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))**2 + 4*np.sin(self.theta_S_Ec)**2*np.sin(self.phi_S_Ec - t*omega_E)**2) - 
        (2*np.cos(self.theta_S_Ec)*(-(np.sqrt(3)*np.cos(self.theta_S_Ec)) - np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))*np.sin(self.phi_S_Ec - t*omega_E))/
        ((np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))**2 + 4*np.sin(self.theta_S_Ec)**2*np.sin(self.phi_S_Ec - t*omega_E)**2))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))) - 
        (-0.5*(np.sqrt(3)*np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E)) - np.sin(self.theta_S_Ec)/2.)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))))/2. + 
        self.a0*(-(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_L))*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))*
        ((np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))))/
        (self.a0**2*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))**2*
        ((np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))))/2. + 
        np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))**2 + 
        (self.a0**2*(1 + (np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))**2)**2*
        ((np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))**2)/4.) + 
        (self.a0*(1 + (np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))**2)*
        ((np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))*
        (self.a0*(-(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_L))*
        ((np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))))/2. + 
        np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))) + 
        self.a0*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))*
        (np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
        (((-((np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(-(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_L))) - 
        (-0.5*(np.sqrt(3)*np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E)) - np.sin(self.theta_S_Ec)/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))*
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.))/
        ((np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))**2 + 
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.)**2) + 
        ((-0.5*np.cos(self.theta_L) + (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. + 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))*
        (-0.5*(np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L)*np.sin(self.phi_S_Ec) + np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_S_Ec)*np.cos(self.theta_L)*np.cos(self.phi_S_Ec)) - np.cos(self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.))/
        ((np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))**2 + 
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.)**2)) + 
        2*np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        ((-2*np.sin(self.theta_S_Ec)*(np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E) - np.sqrt(3)*np.sin(self.theta_S_Ec))*np.sin(self.phi_S_Ec - t*omega_E))/
        ((np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))**2 + 4*np.sin(self.theta_S_Ec)**2*np.sin(self.phi_S_Ec - t*omega_E)**2) - 
        (2*np.cos(self.theta_S_Ec)*(-(np.sqrt(3)*np.cos(self.theta_S_Ec)) - np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))*np.sin(self.phi_S_Ec - t*omega_E))/
        ((np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))**2 + 4*np.sin(self.theta_S_Ec)**2*np.sin(self.phi_S_Ec - t*omega_E)**2)) + 
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (-0.5*(np.sqrt(3)*np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E)) - np.sin(self.theta_S_Ec)/2.)*(np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))) + 
        np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        (-0.5*(np.sqrt(3)*np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E)) - np.sin(self.theta_S_Ec)/2.)*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))) - 
        2*(np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        (((-((np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(-(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_L))) - 
        (-0.5*(np.sqrt(3)*np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E)) - np.sin(self.theta_S_Ec)/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))*
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.))/
        ((np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))**2 + 
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.)**2) + 
        ((-0.5*np.cos(self.theta_L) + (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. + 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))*
        (-0.5*(np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L)*np.sin(self.phi_S_Ec) + np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_S_Ec)*np.cos(self.theta_L)*np.cos(self.phi_S_Ec)) - np.cos(self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.))/
        ((np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))**2 + 
        (-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.)**2))*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))) - 
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
        ((-2*np.sin(self.theta_S_Ec)*(np.cos(self.theta_S_Ec)*np.cos(self.phi_S_Ec - t*omega_E) - np.sqrt(3)*np.sin(self.theta_S_Ec))*np.sin(self.phi_S_Ec - t*omega_E))/
        ((np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))**2 + 4*np.sin(self.theta_S_Ec)**2*np.sin(self.phi_S_Ec - t*omega_E)**2) - 
        (2*np.cos(self.theta_S_Ec)*(-(np.sqrt(3)*np.cos(self.theta_S_Ec)) - np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))*np.sin(self.phi_S_Ec - t*omega_E))/
        ((np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))**2 + 4*np.sin(self.theta_S_Ec)**2*np.sin(self.phi_S_Ec - t*omega_E)**2))*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))))/
        (2.*(self.a0**2*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))**2*
        ((np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))))/2. + 
        np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))**2 + 
        (self.a0**2*(1 + (np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))**2)**2*
        ((np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))**2)/4.)))*
        np.sin(2*pi*self.freq_int(t) + ArcTan((self.a0*(1 + (np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))**2)*
        ((np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))))/2.,
        self.a0*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))*
        ((np.cos(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))))*
        (1 + (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)**2)*
        np.sin(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L)))))/2. + 
        np.cos(2*ArcTan(-0.5*(np.sin(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_S_Ec - self.phi_L)) - 
        (np.sqrt(3)*np.cos(t*omega_E)*(np.cos(self.theta_L)*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec) - np.cos(self.theta_S_Ec)*np.sin(self.theta_L)*np.sin(self.phi_L)))/2. - 
        (np.sqrt(3)*(-(np.cos(self.theta_L)*np.cos(self.phi_S_Ec)*np.sin(self.theta_S_Ec)) + np.cos(self.theta_S_Ec)*np.cos(self.phi_L)*np.sin(self.theta_L))*np.sin(t*omega_E))/2.,
        np.cos(self.theta_L)/2. - (np.sqrt(3)*np.cos(self.phi_L - t*omega_E)*np.sin(self.theta_L))/2. - 
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*(np.cos(self.theta_S_Ec)*np.cos(self.theta_L) + np.cos(self.phi_S_Ec - self.phi_L)*np.sin(self.theta_S_Ec)*np.sin(self.theta_L))))*
        (np.cos(self.theta_S_Ec)/2. - (np.sqrt(3)*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/2.)*
        np.sin(2*(t*omega_E + ArcTan(-2*np.sin(self.theta_S_Ec)*np.sin(self.phi_S_Ec - t*omega_E),np.sqrt(3)*np.cos(self.theta_S_Ec) + np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec)))))) + 
        (2*AU*self.freq(t)*pi*np.cos(self.phi_S_Ec - t*omega_E)*np.sin(self.theta_S_Ec))/c))/2.