from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import simps

import input_data as ip
from hamiltonian import Hmat
from fermi_derivative import fermi_derivative
from density import calc_fermi

def main():

    k0 = 2*np.pi/ip.R
    Nk = 300
    k = np.linspace(-k0, k0, Nk)

    NB = 50
    B0 = 1.0
    B = np.linspace(-B0, B0, NB)
    H = np.zeros(NB, dtype=object)
    E = np.zeros(NB, dtype=object)
    for ib, b in enumerate(B):
        with ProcessPoolExecutor() as pool:
            H_fut = pool.map(Hmat, k, repeat(b), repeat(np.pi/2))
        H[ib] = np.array([fut for fut in H_fut])

    H = np.stack(H, axis=0)

    for ib in range(NB):
        E[ib] = np.array([eigh(H[ib, ik], eigvals_only=True, lower=True) for ik in range(Nk)])

    E = np.stack(E, axis=0)

    #plt.plot(B, E[:,int(Nk/3)]/ip.e*1e3)
    #plt.show()

    T = 0.2
    #Nmu = 100
    #dmu = 12*ip.e/1e3
    #band_min = np.min(E)-0.2*ip.e/1e3
    #band_max = np.min(E)+dmu
    #chem_potential = np.linspace(band_min, band_max, Nmu)
    mu = 0.0*ip.e/1e3
    #mu = calc_fermi(T, 0, ip.theta)
    #mu = 3*ip.e/1e3

    #v = np.gradient(E, axis=0)
    v = np.gradient(H, axis=0)
    #v = np.stack([V[ik].conj().T@v[ik]@V[ik] for ik in range(Nk)])


    fermi = np.stack([fermi_derivative(k, E[ib], mu, T, order=1) for ib in range(NB)])
    print(fermi.shape)
    
    cond = np.stack([[v[ib, ik]*v[ib, ik]*fermi[ib,ik] for ik in range(Nk)] for ib in range(NB)])
    cond = -np.sum([simps(cond[ib], k, axis=0) for ib in range(NB)],axis=1)

    print(cond.shape)
    coeff = ip.e**2*ip.tau_e/2/np.pi/ip.area


    #fig, (ax1, ax2) = plt.subplots(2,1)
    fig, ax1 = plt.subplots()
    #ax1.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
    #ax2.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
    #ax2.set_ylim([-np.pi, np.pi])
    ax1.plot(B, cond*coeff, 'k')
    #ax2.plot(E/ip.e*1e3, k*ip.R, 'k')
    plt.show()

if __name__=="__main__":
    main()
