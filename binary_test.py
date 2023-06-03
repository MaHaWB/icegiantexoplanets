from scipy.constants import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import LISA
import time
from astropy import units
import astropy.coordinates as coord
from tamanini_data import Tamanini
import pickle

pi = np.pi
AU = 1495978707e2 # m
pc = 3.0857e16 # 2e5 AU
yr = 24*3600*365.25 # s

M_J = 1.898e27 # kg
M_S = 1.989e30 # kg
R_S = 696340e3 # m
r_S = 2*G/c**2 # m

sqth = np.sqrt(3)/2
omega_E = 2*pi/yr
fig = 'Figures/'

arr_yr = np.linspace(0,yr,1000)
from binary import binary
from ig_binary import ig_binary

infile = open('dict_binaries.txt','rb')
binaries = pickle.load(infile)
infile.close()

infile = open('dict_binaries_ig.txt','rb')
binaries_ig = pickle.load(infile)
infile.close()

    
infile = open('dict_binaries_full.txt','rb')
binaries_full = pickle.load(infile)
infile.close()


labels = ['K','P','phi_0', 'theta_S_Ec', 'phi_S_Ec', 'theta_L', 'phi_L', 'ln(A)', 'f_1', 'f_0']
keys = enumerate(labels)
file = 'dict_binaries.txt'

#XRAY = binary(np.arccos(.3),5.,8.6e6,np.arccos(-.2),4.,m1=10,m2=100,freq=10e-3,mP=10,P=1,theta_P=pi/2,phi_P=pi/2,T_obs=4,mode='m')

# start = time.time()
# print(B.rel_uncertainty())
# end = time.time()
# print((end-start)/60)
# B.add_json()

#print(B.h_ixh_j(6, 6, 1e-6))

def sanity_plots():
    
    plt.figure(dpi=300)
    for key in [1,2]:
        B.key=key
        plt.plot(arr_yr,[B.A(t) for t in arr_yr])
    plt.legend(['$A_I$','$A_{II}$'])
    plt.xlabel('$t$ in s for one yr')
    plt.ylabel('Strain amplitude')
    plt.savefig(fig+'Strain_Amplitude.png')
    plt.show()
    
    plt.figure(dpi=300)
    for key in [1,2]:
        B.key=key
        plt.plot(arr_yr,[B.polarization_phase(t) for t in arr_yr])
    plt.legend([r'$\varphi^{p}_I$',r'$\varphi^{p}_{II}$'])
    plt.xlabel(r'$t$ in s for one yr')
    plt.ylabel(r'Polarization Phase $\varphi_p$ in rad')
    plt.savefig(fig+r'Polarization_Phase.png')
    plt.show()
    
    plt.figure(dpi=300)
    for key in [1,2]:
        B.key=key
        plt.plot(arr_yr,[B.doppler_phase(t) for t in arr_yr])
    plt.legend([r'$\varphi^{p}_I$',r'$\varphi^{p}_{II}$'])
    plt.xlabel(r'$t$ in s for one yr')
    plt.ylabel(r'Polarization Phase $\varphi_p$ in rad')
    plt.savefig(fig+r'Polarization_Phase.png')
    plt.show()
    
    add = 0.
    hour = np.linspace(0,5*60,1000) + add
    plt.figure(dpi=300)
    C = binary(np.arccos(.3),5.,1e3,np.arccos(-.2),4.,m1=.23,m2=.23,freq=10e-3,mP=0,P=100,theta_P=0,phi_P=0,T_obs=4)
    for key in [1,2]:
        B.key=key
        C.key=key
        plt.plot(hour - add,[B.strain(t) for t in hour],'-')
        plt.plot(hour - add,[C.strain(t) for t in hour],':')
    plt.legend(['$h_I$ w/ exoplanet','$h_I$ w/o exoplanet','$h_{II}$ w/ exoplanet','$h_{II}$ w/o exoplanet'])
    plt.xlabel('$t$ in s for five minutes')
    plt.ylabel('Strain $h$ dimensionless')
    plt.title(r'For $P=2$ yr and $t_0=0$ yr')
    plt.savefig(fig+'Strain_1.png')
    plt.show()
    
    for i, st in keys:
        f = C.h_i(i,1e-6)
        plt.plot(hour,[f(t) for t in hour+.8*yr])
        plt.title(st)
        plt.show()
        
def hr_d_strain():
    B = ig_binary(dist=1e3,freq=10e-3,mP=1,P=10)
    hour = np.linspace(0,5*60,1000)+5*yr
    for i, st in enumerate(['K','P','phi_0', 'theta_S_Ec', 'phi_S_Ec', 'theta_L', 'phi_L', 'ln(A)','f_1','f_0']):
        plt.figure(dpi=300)
        f = B.h_i(i)
        plt.plot(hour,[f(t)**2 for t in hour])
        #plt.legend([r'$\varphi^{p}_I$',r'$\varphi^{p}_{II}$'])
        plt.title('Five minute strain for $\lambda_i=${}'.format(st))
        plt.xlabel(r'$t$ in s')
        plt.ylabel(r'$\left(\frac{\partial h}{\partial \lambda_i}\right)^2$')
        plt.savefig(fig+'\d_strain_yr\d_strain^2_ig_{}.png'.format(st))
        plt.show()
    
def yr_d_strain(diff=1e-6):
    arr_yr = np.linspace(0,1,1000)*yr
    for i, st in enumerate(['K','P','phi_0', 'f_0', 'ln(A)', 'theta_S_Ec', 'phi_S_Ec', 'theta_L', 'phi_L']):
        plt.figure(dpi=300)
        f = B.h_i(i,diff)
        plt.plot(arr_yr/yr,[f(t) for t in arr_yr])
        #plt.legend([r'$\varphi^{p}_I$',r'$\varphi^{p}_{II}$'])
        plt.title('One year strain for $\lambda_i=${}'.format(st))
        plt.xlabel(r'$t$ in yr')
        plt.ylabel(r'$\frac{\partial h}{\partial \lambda_i}$')
        #plt.savefig(fig+'\d_strain_yr\d_strain_{}.png'.format(st))
        plt.show()
        
    for i, st in enumerate(['K','P','phi_0', 'f_0', 'ln(A)', 'theta_S_Ec', 'phi_S_Ec', 'theta_L', 'phi_L']):
        plt.figure(dpi=300)
        f = B.h_i(i,diff)
        plt.plot(arr_yr/yr,[f(t)**2 for t in arr_yr])
        #plt.legend([r'$\varphi^{p}_I$',r'$\varphi^{p}_{II}$'])
        plt.title('One year strain for $\lambda_i=${}'.format(st))
        plt.xlabel(r'$t$ in yr')
        plt.ylabel(r'$\left(\frac{\partial h}{\partial \lambda_i}\right)^2$')
        #plt.savefig(fig+'\d_strain_yr\d_strain^2_{}.png'.format(st))
        plt.show()
    
def strain_ig():
    add = 0.
    for i, add in enumerate(np.linspace(0,10,10)*yr):
        hour = np.linspace(0,5*60,1000) + add
        plt.figure(dpi=300)
        CwP = ig_binary(dist=1e3,theta_S_Ec=pi/2+1e-6,phi_S_Ec=0,freq=10e-3,mP=1,P=10,theta_P=pi/2,phi_P=pi/2)
        CwoP = ig_binary(dist=1e3,theta_S_Ec=pi,phi_S_Ec=0,freq=10e-3,mP=0,P=10,theta_P=0,phi_P=0)
        plt.plot(hour - add,[CwP.strain(t) for t in hour],'-')
        plt.plot(hour - add,[CwoP.strain(t) for t in hour],':')
        plt.legend(['$y_2$ w/ exoplanet','$y_2$ w/o exoplanet'])
        plt.xlabel('$t$ in s for five minutes')
        plt.ylabel('Strain $y_2$ dimensionless')
        plt.title(r'For $P=10$ yr, $M_P=M_J$ and $t_0$='+'{:.0f} yr'.format(i))
        plt.savefig(fig+'/ig_Strain/Strain_{}.png'.format(i))
        plt.show()
    
def one_year_degeneracy():
    C = binary(np.arccos(.3),5.,1e3,np.arccos(-.2),4.,m1=.23,m2=.23,freq=10e-3,mP=0,P=100,theta_P=0,phi_P=0,T_obs=4)
    B = binary(np.arccos(.3),5.,1e3,np.arccos(-.2),4.,m1=.23,m2=.23,freq=10e-3,mP=5,P=1,theta_P=pi/2,phi_P=pi/2,T_obs=4)
    for i,adds in enumerate([0,.25,.5,.75,1]):
        add = adds*yr
        hour = np.linspace(0,5*60,1000) + add
        plt.figure(dpi=300)
        for key in [1,2]:
            B.key=key
            C.key=key
            plt.plot(hour - add,[B.strain(t) for t in hour],'-')
            plt.plot(hour - add,[C.strain(t) for t in hour],':')
        plt.legend(['$h_I$ w/ exoplanet','$h_I$ w/o exoplanet','$h_{II}$ w/ exoplanet','$h_{II}$ w/o exoplanet'])
        plt.xlabel('$t$ in s for five minutes')
        plt.ylabel('Strain $h$ dimensionless')
        plt.title(r'For $P=1$ yr and $t_0={}$ yr'.format(adds))
        plt.savefig(fig+'OneYearPlanet/Strain_{}.png'.format(i))
        plt.show()
 
def deriv_check():
    for add in [0,.5,1]:
        B = binary(np.arccos(.3),5.,1e3,np.arccos(-.2),4.,m1=.23,m2=.23,freq=10e-3,mP=5,P=5,theta_P=pi/2,phi_P=pi/2,T_obs=4)
        hour = np.linspace(0,5*60,1000) + add*yr
        for i in [0,1,2]:
            f = B.h_i(i,1e-6)
            plt.plot(hour,[f(t) for t in hour],label='{}'.format(i))
        #plt.plot(hour,[B.f_0(t) for t in hour],label='OG')
        plt.legend()
        plt.show()
        for i in [3,4,5,6,7,8]:
            f = B.h_i(i,1e-6)
            plt.plot(hour,[f(t) for t in hour],label='{}'.format(i))
        #plt.plot(hour,[B.freq_int(t) for t in hour],label='OG')
            plt.legend()
            plt.show()
        
def pos_dep():
    rel_unc = np.zeros((20,20,3))
    for i,mu in enumerate(np.linspace(-1,1,20)):
        for j,rad in enumerate(np.linspace(0,2*pi,20)):
            rel_unc[j,i,:] = binary(np.arccos(mu),rad,1e3,np.arccos(-.2),4.,m1=.23,m2=.23,freq=1e-3,mP=5,P=1.5,theta_P=pi/2,phi_P=pi/2,T_obs=4).rel_uncertainty()
    rel_unc = np.log10(rel_unc)
    for k,label in enumerate(['K','P']):
        plt.matshow(rel_unc[:,:,k])
        plt.ylabel('$\mu$')
        plt.xlabel('$\phi$')
        plt.colorbar()
        plt.title('Relative uncertainty in '+label)
        plt.show()
    return rel_unc
    

def test(n=100):
    Ps = np.logspace(-2,1,n)
    uncs = np.zeros((n,9),np.float64)
    #uncs2 = np.zeros((n,3),np.float64)
    for n,P0 in enumerate(Ps):
        uncs[n] = [binary(np.arccos(.3),5.,1e3,np.arccos(-.2),4.,m1=.23,m2=.23,freq=10e-3,mP=10,P=P0,theta_P=pi/2,phi_P=pi/2,T_obs=4).h_ixh_j(i,i,1e-6) for i in np.arange(9)]
        #uncs2[n] = binary(np.arccos(.3),5.,1e3,np.arccos(-.2),4.,m1=.23,m2=.23,freq=1e-3,mP=mP,P=P0,theta_P=pi/2,phi_P=pi/2,T_obs=4).rel_uncertainty()
    #uncs/=np.diag(B.Fisher)
    #uncs2*=mP
    plt.figure(dpi=300)
    plt.tight_layout()
    
    plt.loglog(Ps,uncs[:,0],'r-',label=r'$\sigma_K/K$')
    plt.loglog(Ps,uncs[:,1],'b-',label=r'$\sigma_P/P$')
    plt.loglog(Ps,uncs[:,2],'g-',label=r'$\sigma_\varphi$')
    plt.loglog(Ps,uncs[:,3],'r-',label=r'3')
    plt.legend()
    plt.show()
    
    plt.loglog(Ps,uncs[:,4],'b-',label=r'4')
    plt.loglog(Ps,uncs[:,5],'g-',label=r'5')
    plt.loglog(Ps,uncs[:,6],'r-',label=r'6')
    plt.loglog(Ps,uncs[:,7],'b-',label=r'7')
    plt.loglog(Ps,uncs[:,8],'g-',label=r'8')
    plt.legend()
    plt.show()
    
    #plt.loglog(Ps,uncs2[:,0],'r--')
    #plt.loglog(Ps,uncs2[:,1],'b--')
    #plt.loglog(Ps,uncs2[:,2],'g--')
    
    return uncs

def test2(n=20,i=5,j=5):
    Ps = np.logspace(-2,1,n)
    plt.loglog(Ps,[binary(np.arccos(.3),5.,1e3,np.arccos(-.2),4.,m1=.23,m2=.23,freq=10e-3,mP=10,P=P0,theta_P=pi/2,phi_P=pi/2,T_obs=4).h_ixh_j(i,i,1e-6) for P0 in Ps])
    plt.show()

def is_analytical_truly_better():
    B = binary(np.arccos(.3),5.,1e3,np.arccos(-.2),4.,m1=.23,m2=.23,freq=10e-3,mP=5,P=2,theta_P=pi/2,phi_P=pi/2,T_obs=4)
    func = B.h_i(3,1e-6)
    arr_yr = np.linspace(0,1,10000)*yr
    plt.figure(dpi=300)
    #plt.plot(arr_yr/yr,[func(t) for t in arr_yr],'-',label='numerical')
    plt.plot(arr_yr/yr,[B.dh_dtheta_S(t) for t in arr_yr],'-',label='analytic')
    #plt.legend()
    plt.title('One year strain - analytical')
    plt.xlabel(r'$t$ in yr')
    plt.ylabel(r'$\partial h/\partial \theta_S (t)$')
    plt.savefig(fig+'\d_strain_yr\d_strain_comparison_yr.png')
    plt.show()

    for add in [.5,1,1.5]:
        hour = np.linspace(0,5*60,1000) + add*yr
        plt.figure(dpi=300)
        C = binary(np.arccos(.3),5.,1e3,np.arccos(-.2),4.,m1=.23,m2=.23,freq=10e-3,mP=0,P=100,theta_P=0,phi_P=0,T_obs=4)
        funcB = C.h_i(3,1e-6)
        plt.plot(hour-add*yr,[func(t) for t in hour],'-')
        plt.plot(hour-add*yr,[B.dh_dtheta_S(t) for t in hour],'-.')
        plt.plot(hour-add*yr,[funcB(t) for t in hour],'--')
        plt.plot(hour-add*yr,[C.dh_dtheta_S(t) for t in hour],':')
        plt.legend(['$h_I$ w/ exoplanet - num','$h_I$ w/ exoplanet - ana','$h_I$ w/o exoplanet - num','$h_I$ w/o exoplanet - ana'])
        plt.xlabel('$t$ in s for five minutes')
        plt.ylabel(r'$\partial h/\partial \theta_S (t)$')
        plt.title(r'For $P=2$ yr and $t_0=$ {} yr'.format(add))
        plt.savefig(fig+'\d_strain_yr\d_strain_comparison_5_min_{}e-1yr.png'.format(10*add))
        plt.show()

def correlation_mat(delete=True):
    infile = open('dict_binaries_ig.txt','rb')
    binaries_ig = pickle.load(infile)
    infile.close()
    
    latest = binaries_ig['l'][-1]
    corr = latest['correlation']
    labels=[r'K',r'P',r'$\phi_0$', r'$\theta_S$', '$\phi_S$', r'$\theta_L$', '$\phi_L$', 'ln(A)', '$f_1$', '$f_0$']
    rel_unc = np.abs(corr)*latest['SNR']**2
    plt.matshow(np.log10(rel_unc))
    plt.xticks(np.arange(10),labels=labels,rotation=40)
    plt.yticks(np.arange(10),labels=labels)
    plt.colorbar()
    plt.title('log$_{10}$(Correlation matrix) for IG, SNR = 1 and $M_P$ = '+str(int(latest['pars']['M_P']))+' $M_J$')
    plt.savefig(fig+'Correlation_Mat.png')
    plt.show()
    
    if delete:
        labels=[r'K',r'P',r'$\phi_0$', r'$\theta_S$', '$\phi_S$', r'$\theta_L$', '$\phi_L$', 'ln(A)', '$f_1$']
        l2 = latest['binary'].copy()
        l2.mode = 'm'
        corr2 = l2.json()['correlation']
        rel_unc2 = np.abs(corr2*(latest['SNR']))**2
        plt.matshow((rel_unc2 - rel_unc[:-1,:-1]) / rel_unc[:-1,:-1])
        plt.xticks(np.arange(9),labels=labels,rotation=40)
        plt.yticks(np.arange(9),labels=labels)
        plt.colorbar()
        plt.title(r'Relative difference between mode l and m for IG$')
        plt.savefig(fig+'Correlation_Mat_2.png')
        plt.show()
        pass

def look_at_binary(B,delete=[],txt='Correlation_Mat_new.png'):
    labels=[r'K',r'P',r'$\phi_0$', r'$\theta_S$', '$\phi_S$', r'$\theta_L$', '$\phi_L$','ln(A)',r'$f_1$',r'$f_0$']
    plt.figure(dpi=300)
    corr = B['correlation']
    corr *= B['SNR']**2*B['pars']['M_P']**2
    for i in delete:
        corr = np.delete(np.delete(corr,i,1),i,0)
        labels.pop(i)
    i = np.size(delete)
    rel_unc = np.log10(np.abs(corr))
    plt.imshow(rel_unc)
    plt.xticks(np.arange(10-i),labels=labels,rotation=40)
    plt.yticks(np.arange(10-i),labels=labels)
    plt.colorbar()
    plt.title(r'log$_{10}($Correlation matrix$\times$SNR$^2\times (M_P / M_J)^2)$, '+'$P=${:.0f} yr'.format(B['pars']['P']))
    plt.savefig(fig+txt)
    plt.show()

def plot_all_Tamanini():
    Pm = []
    Ps = []
    tamm = []
    tams = []
    for i in binaries['m']:
        Pm.append(i['pars']['P'])
        tamm.append(i['Tamanini_plot'])
    for i in binaries['s']:
        Ps.append(i['pars']['P'])
        tams.append(i['Tamanini_plot'])
    tamm = np.array(tamm)
    tams = np.array(tams)
    
    plt.figure(dpi=300)
    plt.tight_layout()
    
    plt.loglog(Pm,tamm[:,0],'rx',label=r'$\sigma_K/K$')
    plt.loglog(Pm,tamm[:,1],'bx',label=r'$\sigma_P/P$')
    plt.loglog(Pm,tamm[:,2],'gx',label=r'$\sigma_\varphi$')
    
    plt.loglog(Ps,tams[:,0],'r-')
    plt.loglog(Ps,tams[:,1],'b-')
    plt.loglog(Ps,tams[:,2],'g-')
    
    Tamanini(10)
    Tamanini(1)
    
    plt.title(r'Positions as in [1], $M_{b1,2}=0.23M_\odot,r=1kpc,M_P=1M_J, T_{obs}=4 yr$')
    plt.legend()
    plt.grid()
    plt.ylabel(r'$\sqrt{(\Gamma^{-1})_{ii}}/\lambda_i\cdot$SNR$\cdot M_P/M_J$')
    plt.xlabel(r'$P$ in yr')
    plt.savefig(fig+'Tamanini_comp.png')
    
def look_at_unc():
    infile = open('dict_binaries.txt','rb')
    binaries = pickle.load(infile)
    infile.close()
    for json in binaries['m']:
        look_at_binary(json['binary'])
        plt.show()
       
def sin_fit(b,plot=True):
    t = np.linspace(0,100/b.f_GW,1000)
    s = np.array([b.strain(t0)*1e40 for t0 in t])
    fit = lambda y: np.array([y[0]*1e40*np.cos(y[1]*t0+y[2]) for t0 in t]) - s
    x0 = [b.a0,2*pi*b.f_GW,pi]
    res = optimize.leastsq(fit,x0,ftol=1e-8)[0]
    if plot:
        plt.plot(t,s,label='strain')
        #plt.plot(t,[x0[0]*1e40*np.cos(x0[1]*t0+x0[2]) for t0 in t],label='x0')
        plt.plot(t,[res[0]*1e40*np.cos(res[1]*t0+res[2]) for t0 in t],':',label='fit')
        plt.title('mu,A,f={:.2f},{:.2e},{:.2e}'.format(b.kn,res[0],res[1]))
        plt.legend()
        plt.show()
    return res
       
def mu_dependence_ig(n=10,plot=True):
    mus = np.linspace(-1,1,n)
    phis = np.linspace(0,2*pi,n)
    amplitude = np.zeros((n,n))
    mu = np.zeros((n,n))
    for i, theta in enumerate(np.arcsin(mus)+pi/2):
        for j, phi in enumerate(phis+pi):
            b = ig_binary(theta_S_Ec=theta,phi_S_Ec=phi,theta_L=theta,phi_L=phi)
            if np.abs(b.kn) == 1:
                 amplitude[i,j] = 0.
            else:
                amplitude[i,j] = integrate.quad(lambda t: b.strain(t)**2,0,40*24*60*60,epsrel=1.49e-2,epsabs=0)[0]
            print(i,j)
            mu[i,j] = b.kn
    
    plt.figure(dpi=300)
    plt.imshow(amplitude,extent=[-pi,pi,pi,0])
    plt.colorbar()
    plt.xlabel(r'$\phi_S$')
    plt.ylabel(r'$\theta_S$ uniform in sin$(\theta_S)$')
    plt.title(r'$\int_0^{40 d}y^2 dt$ for face-on binaries')
    plt.savefig(fig+'pos_dependance_ig_2.png')
    plt.show()
    
    plt.figure(dpi=300)
    plt.plot(mu.flatten(),amplitude.flatten(),'.')
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\int_0^{40 d}y^2 dt$')
    plt.savefig(fig+'mu_dependace_ig_2.png')
    plt.show()
    
    return [amplitude,mu,mus,phis]

def XRayBinary(f=10e-3,n=10):
    ra= '+13d29m43.3s'
    dec= '+47d11m34.7s'
    c = coord.SkyCoord(ra=ra,dec=dec,frame='icrs').transform_to(coord.BarycentricMeanEcliptic)
    phi = c.lon.radian
    theta = -c.lat.radian + pi/2
    
    mStar = np.logspace(np.log10(3.4),np.log10(7.3),n+1)
    mBH = np.logspace(np.log10(2),np.log10(10),n+1)
    
    snr = np.zeros((n,n))
    for i, mA in enumerate(mStar[:-1]):
        for j, mB in enumerate(mBH[:-1]):
            bi = binary(theta_S_Ec=theta,phi_S_Ec=phi,dist=8.6e6,theta_L=theta,phi_L=phi,m1=np.sqrt(mA*mStar[i+1]),m2=np.sqrt(mB*mBH[i+1]))
            snr[i,j] = bi.sep()/R_S
            print(i,j)
            
    plt.figure(dpi=300)
    plt.pcolor(mBH,mStar,snr)
    plt.xscale('log')
    plt.yscale('log')
    plt.colorbar()
    plt.xlabel(r'$M_{BH}$ in $M_\odot$')
    plt.ylabel(r'$M_{Star}$ in $M_\odot$')
    plt.title(r'$d/R_\odot$ for M51-ULS-1b if face-on and $f_{GW}=10$ mHz')
    plt.savefig(fig+'XRay_sep.png')
    plt.show()
    
    return snr

def strain_png():
    plt.style.use('seaborn-bright')
    plt.figure(dpi=300)
    plt.tight_layout()
    B = binary()
    arr_yr = np.linspace(0,1,10**4)
    for key in [1,2]:
        B.key=key
        plt.plot(arr_yr,[B.strain(t) for t in arr_yr*yr],alpha=.6)
    plt.legend([r'$\alpha=I$',r'$\alpha=II$'])
    plt.xlabel(r'$t$ in yr')
    plt.ylabel(r'Strain $h_\alpha (t)$')
    plt.grid()
    plt.savefig(fig+r'Strain.pdf')
    plt.show()
    
def strain_ig_png():
    plt.style.use('seaborn-bright')
    plt.figure(dpi=300)
    plt.tight_layout()
    B = ig_binary()
    arr_yr = np.linspace(0,1,10**4)
    plt.plot(arr_yr,[B.strain(t) for t in arr_yr*3600])
    plt.legend([r'$\alpha=I$'])
    plt.xlabel(r'$t$ in d')
    plt.ylabel(r'Strain $y_2 (t)$')
    plt.grid()
    plt.savefig(fig+r'Strain_2.pdf')
    plt.show()
    
def LISA_png():
    plt.figure(dpi=300)
    plt.tight_layout()
    B = binary()
    arr_yr = np.linspace(0,1,10**4)
    for key in [1,2]:
        B.key=key
        plt.plot(arr_yr,[B.strain(t) for t in arr_yr*yr],alpha=.6)
    plt.legend([r'$\alpha=I$',r'$\alpha=II$'])
    plt.xlabel(r'$t$ in yr')
    plt.ylabel(r'Strain $h_\alpha (t)$')
    plt.grid()
    plt.savefig(fig+r'Strain.pdf')
    plt.show()

def integrand_ex_png():
    plt.style.use('seaborn-bright')
    plt.figure(dpi=300)
    plt.tight_layout()
    B = binary(P=8.5)
    funcA = B.h_i(0)
    funcB = B.h_i(1)
    arr_yr = np.linspace(0,4,10**5)
    for key in [1,2]:
        B.key=key
        plt.plot(arr_yr,[funcA(t)*funcB(t) for t in arr_yr*yr],alpha=.6)
    plt.legend([r'$\alpha=I$',r'$\alpha=II$'])
    plt.xlabel(r'$t$ in yr')
    plt.ylabel(r'Integrand $\partial_P h_\alpha \times \partial_K h_\alpha(t)$')
    plt.grid()
    plt.savefig(fig+r'Fisher_integrand_2.png')
    plt.show()
    
def LISA_IG_s_png():
    PI = []
    PL = []
    tamI = []
    tamL = []
    for i in binaries_full['l']:
        B = i['binary']
        PL.append(i['pars']['P'])
        tamL.append(B.reduced_fisher_mat(['theta_S_Ec', 'phi_S_Ec', 'theta_L', 'phi_L', 'ln(A)', 'f_1', 'f_0'],False,True))
    for iz, i in enumerate(binaries_ig['s']):
        if iz < 3:
            continue
        B = i['binary']
        if B.f_GW > 5e-3 and i['pars']['P'] > 0.03:
            PI.append(i['pars']['P'])
            tamI.append(i['Tamanini_plot'])
    tamI = np.array(tamI)
    tamL = np.array(tamL)
    
    plt.figure(dpi=300)
    
    res = [(PI[i],tamI[i]) for i in np.arange(len(PI))]
    res.sort(key=lambda x: x[0])
    res.pop(1)
    PI = [i[0] for i in res]
    tamI = np.array([i[1] for i in res])
    
    plt.loglog(PI,tamI[:,0],'r-.',label=r'$\sigma_K/K$')
    plt.loglog(PI,tamI[:,1],'b-.',label=r'$\sigma_P/P$')
    plt.loglog(PI,tamI[:,2],'g-.',label=r'$\sigma_\varphi$')
    
    plt.loglog(PI,tamI[:,0],'rx',markersize=5)
    plt.loglog(PI,tamI[:,1],'bx',markersize=5)
    plt.loglog(PI,tamI[:,2],'gx',markersize=5)
    
    plt.loglog(PL,tamL[:,0],'r^')
    plt.loglog(PL,tamL[:,1],'b^')
    plt.loglog(PL,tamL[:,2],'g^')
    
    Tamanini(10,False)
    
    plt.legend()
    plt.grid()
    plt.ylabel(r'$\sigma_i/\lambda_i\cdot S/N \cdot M_P/M_J$')
    plt.xlabel(r'$P$ in yr')
    plt.tight_layout()
    plt.savefig(fig+'Tamanini_comp.pdf')    
    
def IG_1_png():
    PI = []
    tamI = []
    for i in binaries_ig['s']:
        B = i['binary']
        if B.f_GW < 5e-3:
            PI.append(i['pars']['P'])
            tamI.append(i['Tamanini_plot'])
    tamI = np.array(tamI)
    
    plt.figure(dpi=300)
    
    plt.loglog(PI,tamI[:,0],'rx',label=r'$\sigma_K/K$')
    plt.loglog(PI,tamI[:,1],'bx',label=r'$\sigma_P/P$')
    plt.loglog(PI,tamI[:,2],'gx',label=r'$\sigma_\varphi$')
    
    Tamanini('all',False)
    
    plt.legend()
    plt.grid()
    plt.ylabel(r'$\sigma_i/\lambda_i\cdot S/N \cdot M_P/M_J$')
    plt.xlabel(r'$P$ in yr')
    plt.tight_layout()
    #plt.savefig(fig+'Tamanini_comp.pdf')    
    
#IG_1_png()
#LISA_IG_s_png()
#integrand_ex_png()
strain_ig_png()
#test2()
#uncs = test(20)
#pos_dep()
#sanity_plots()
#yr_d_strain(1e-6)
#uncs, uncs2, bs1, bs2, times = uncertainty_plot(10,mP=10,a=-2,b=1,mode='m')
#A = np.power(uncs[:-2,:]*uncs[1:-1,:]*uncs[2:,:],1/3)[::3,:]
#one_year_degeneracy()
#deriv_check()
#correlation_mat()
#is_analytical_truly_better()
#plot_all_Tamanini()
#look_at_unc()
#strain_ig()
#hr_d_strain()
#sin_fit(ig_binary())
#A = mu_dependence_ig(20,True)
#A = XRayBinary(n=10)