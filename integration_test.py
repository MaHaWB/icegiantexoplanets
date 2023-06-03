import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import vegas
import time

yr = 360.25*24*60*60

Osc = 4*yr/10e-3
def func(x,Osc=Osc):
    return np.sin(2*np.pi*x*Osc)**2

# Analytical Solution:
# \int_0^1 sin^2(2*pi*Osc*t) dt = 1/2 - sin(2*pi*Osc)*cos(2*pi*Osc)/(2*2*pi*Osc)

def analytic_sol(Osc=Osc):
    return 1/2 - np.sin(2*np.pi*Osc)*np.cos(2*np.pi*Osc)/(2*2*np.pi*Osc)

print(analytic_sol())

start = time.time()
integ = vegas.Integrator([[0,1]])
sol_MC = integ(func, nitn=10, neval=1000,adapt=False)
print(sol_MC.summary())
end = time.time()
print('Finished MC computation after {:.1f} s, solution: \n {:.2f} +- {:.3f}, Q = {:.2f} \n\n'.format(end-start, sol_MC[0].mean, sol_MC[0].sdev, sol_MC.Q))