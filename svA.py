from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import simps

import input_data as ip
from hamiltonian import Hmat as Hmat_tube
from hamiltonian import quantum_numbers
from hamiltonian_scroll import Hmat as Hmat_scroll
from fermi_derivative import fermi_derivative
from density import calc_fermi, calc_ndens
from spin_cond_111 import vzr as vzr_tube
from spin_cond_111 import vzp as vzp_tube
from spin_cond_111 import vzz as vzz_tube
from spin_cond_scroll import vzr as vzr_scroll
from spin_cond_scroll import vzp as vzp_scroll
from spin_cond_scroll import vzz as vzz_scroll

def calc_band_structure(k, alphaR=ip.alphaR, betaD=ip.betaD, system='tube', Nm=10):

    Nk = len(k)

    with ProcessPoolExecutor() as pool:
        if system == 'tube':
            H_fut = pool.map(Hmat_tube,
                k,
                repeat(ip.B),
                repeat(ip.theta),
                repeat(Nm),
                repeat(alphaR),
                repeat(betaD)
            )
        if system == 'scroll':
            H_fut = pool.map(Hmat_scroll,
                k,
                repeat(ip.B),
                repeat(ip.theta),
                repeat(Nm),
                repeat(alphaR),
                repeat(betaD)
            )
    H = np.array([fut for fut in H_fut])
    E = np.zeros(Nk, dtype=object)
    V = np.zeros(Nk, dtype=object)
    for ik in range(Nk):
        E[ik], V[ik] = eigh(H[ik], lower=True)
    E = np.stack(E)
    V = np.stack(V)

    norm = np.stack([np.real(np.sum(V[ik]@V[ik].conj().T)) for ik in range(Nk)])

    v = np.gradient(H, axis=0)
    v = np.stack([V[ik].conj().T@v[ik]@V[ik]/norm[ik] for ik in range(Nk)])
    v = np.stack([np.diagonal(v[ik]) for ik in range(Nk)])
    v = v/ip.hbar

    return E, V, v, norm

def calc_charge_conductivity(k, v, E, V, chem_potential, T=0.1):

    Nk = len(k)
    Nmu = len(chem_potential)

    fermi = np.stack([fermi_derivative(k, E, mu, T, order=1) for mu in chem_potential])

    cond = np.stack([[v[ik]*v[ik]*fermi[imu,ik] for ik in range(Nk)] for imu in range(Nmu)])
    cond = -np.sum([simps(cond[imu], k, axis=0) for imu in range(Nmu)],axis=1)

    coeff = ip.e**2*ip.tau_e/2/np.pi/ip.area

    return cond*coeff

def calc_spin_conductivity(k, v, E, V, norm, mu, T=0.1, system='tube', polarization='p'):

    Nk = len(k)

    with ProcessPoolExecutor() as pool:
        if system == 'tube' and polarization == 'p':
            v_fut = pool.map(vzp_tube, k)
        if system == 'tube' and polarization == 'z':
            return np.zeros(Nmu)
        if system == 'scroll' and polarization == 'r':
            v_fut = pool.map(vzr_scroll, k)
        if system == 'scroll' and polarization == 'p':
            v_fut = pool.map(vzp_scroll, k)
        if system == 'scroll' and polarization == 'z':
            v_fut = pool.map(vzz_scroll, k)
        
    vs = np.stack([fut for fut in v_fut])

    vs = np.stack([V[ik].conj().T@vs[ik]@V[ik]/norm[ik] for ik in range(Nk)])
    vs = np.stack([np.diagonal(vs[ik])*ip.hbar/2 for ik in range(Nk)])
    
    fermi = np.stack(fermi_derivative(k, E, mu, T, order=2))

    cond = np.stack([v[ik]*v[ik]*vs[ik]*fermi[ik] for ik in range(Nk)])
    cond = np.sum(simps(cond, k, axis=0),axis=0)

    coeff = ip.e**2*ip.tau_s**2/2/np.pi/ip.area*ip.hbar/4/ip.e/1e3

    return cond*coeff

if __name__=="__main__":

    k0 = 2*np.pi/ip.R
    Nk = 2000
    k = np.linspace(-k0, k0, Nk)
    NA = 20
    A = np.linspace(0, 50e-12*ip.e, NA)
    #A = np.linspace(-ip.alphaR, ip.alphaR, NA)
    scond = np.zeros(NA, dtype=object)
    for ia, a in enumerate(A):
        #E, V, v, norm = calc_band_structure(k, alphaR=ip.alphaR, betaD=a, system='scroll')
        E, V, v, norm = calc_band_structure(k, alphaR=a, betaD=ip.betaD, system='tube')
        mu = np.min(E[:,4])
        #mu = 0.0*ip.e/1e3
        scond[ia] = calc_spin_conductivity(k, v, E, V, norm, mu, 0.1, system='tube', polarization='p')

    plt.ylabel(r'$\sigma^{(2),\varphi}_z$')
    #plt.ylabel(r'$\sigma^{(2),z}_z$')
    #plt.xlabel(r'$\alpha/\beta$')
    #plt.xlabel(r'$\beta/\alpha$')
    #plt.plot(A/ip.alphaR, scond/1e9, 'k')
    plt.plot(A/ip.e*1e12, scond/1e9, 'k')
    plt.plot(A/ip.e*1e12, np.gradient(scond/1e9,axis=0), 'b', linestyle='dotted')
    plt.axvline(x=20,c='gray',linestyle='dotted')
    plt.axvline(x=26.3,c='gray',linestyle='dotted')
    plt.axvline(x=28.3,c='gray',linestyle='dotted')
    plt.show()
    
    
