# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 13:36:35 2021

@author: marcu
"""
import astropy.units as u
import astropy.coordinates as coord
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pandas as pd

fig = 'Figures/'

c = coord.SkyCoord(lat=np.arccos(.3) * u.rad,
                   lon=5. * u.rad,
                   distance=0. * u.kpc,
                   frame='barycentricmeanecliptic').transform_to(coord.Galactocentric)

Sun = np.array([c.x.value,c.y.value,c.z.value]) # kpc
Centre = coord.SkyCoord(x=0 * u.kpc, y=0* u.kpc, z=0* u.kpc, frame='galactocentric').transform_to(coord.BarycentricMeanEcliptic)

def density_galactic(arr,rho_0=1/29378.,alpha=.25,R_b2=.64,R_d=2.5,Z_d=.4):
    x,y,z = arr
    u2 = x**2 + y**2
    r2 = u2 + z**2
    u = np.sqrt(u2)
    return rho_0*(alpha*np.exp(-r2/R_b2) + (1-alpha)*np.exp(-u/R_d)/np.cosh(z/Z_d)**2)

def projected_milky_way(theta_S,phi_S):
    new_point = coord.SkyCoord(lat=theta_S * u.rad, lon= phi_S * u.rad, distance= 1. * u.kpc, frame='barycentricmeanecliptic').transform_to(coord.Galactocentric)
    pointer = np.array([new_point.x.value,new_point.y.value,new_point.z.value])
    delta = pointer - Sun
    
    path = lambda s: Sun + s*delta
    
    return quad(lambda s: s**2*density_galactic(path(s)),0,np.inf)[0]

def generate_skycover(n=100):
    thetas = np.arcsin(np.linspace(1,-1,n))
    phis = np.linspace(0,2*np.pi,n)
    
    vals = np.zeros((n,n))
    for i, th in enumerate(thetas):
        for j, ph in enumerate(phis):
            vals[i,j] = projected_milky_way(th,ph)
    
    V = np.sum(vals)
    plt.imshow(vals/V)
    np.savetxt("skycover.csv", vals/V, delimiter=",")
    return vals, V
    
#print(generate_skycover())
Sky = pd.read_csv("skycover.csv",header=None).to_numpy()

plt.figure(dpi=300)
plt.imshow(np.log(Sky*.9+.1),extent=[0,2*np.pi,np.pi/2,-np.pi/2])
#plt.colorbar()
plt.title(r'log($\rho$) projected')
plt.xlabel(r'$\phi_S$')
plt.ylabel(r'$\theta_S$ uniform in sin$(\theta_S)$')
plt.xticks()
plt.savefig(fig+'SkyCover.png')