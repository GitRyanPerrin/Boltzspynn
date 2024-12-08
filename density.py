from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.linalg import eigh
from scipy.optimize import fsolve

import input_data as ip
from hamiltonian import Hmat
from fermi_derivative import FD, fermi_derivative

def calc_ndens(k, E, mu, T, B=ip.B, theta=ip.theta, k0=2*np.pi/ip.R):
        Nk = len(k)

        fermi = fermi_derivative(k, E, mu, T, order=0)

        ndens = np.stack([fermi[ik] for ik in range(Nk)])
       #ndens = np.sum(simps(ndens, k, axis=0),axis=0)/ip.R**2/np.pi**2
        ndens = np.sum(simps(ndens, k, axis=0),axis=0)

        return ndens

def calc_fermi(T, B, theta, k0=2*np.pi/ip.R):
    Nk = 1500
    k = np.linspace(-k0, k0, Nk)

    with ProcessPoolExecutor() as pool:
        H_fut = pool.map(Hmat, k, repeat(B), repeat(theta))
    H = np.array([fut for fut in H_fut])
    E = np.stack([eigh(H[ik], eigvals_only=True, lower=True) for ik in range(Nk)])

    def calc_ndens(mu):
        fermi = fermi_derivative(k, E, mu, T, order=0)

        ndens = np.stack([fermi[ik] for ik in range(Nk)])
        ndens = np.sum(simps(ndens, k, axis=0),axis=0)/ip.R**2/np.pi**2

        return ndens - 1.17e23
    
    EF = fsolve(calc_ndens, 12*ip.e/1e3)

    return EF/ip.e*1e3

def calc_fermi2D(T, B, theta, k0=2*np.pi/ip.R):
    Nk = 1500
    k = np.linspace(-k0, k0, Nk)

    with ProcessPoolExecutor() as pool:
        H_fut = pool.map(Hmat, k, repeat(B), repeat(theta))
    H = np.array([fut for fut in H_fut])
    E = np.stack([eigh(H[ik], eigvals_only=True, lower=True) for ik in range(Nk)])

    def calc_ndens(mu):
        fermi = fermi_derivative(k, E, mu, T, order=0)

        ndens = np.stack([fermi[ik] for ik in range(Nk)])
        ndens = np.sum(simps(ndens, k, axis=0),axis=0)/2/ip.R/np.pi**2

        return ndens - 1.17e15
    
    EF = fsolve(calc_ndens, 12*ip.e/1e3)

    return EF/ip.e*1e3


def calc_DOS(T, dmu, B, theta, k0=2*np.pi/ip.R):

    '''
    Returns the density of states dN/dE
    Energy in meV
    DOS in (meV)^-1(m^2)^-1
    '''

    Nk = 1200
    k = np.linspace(-k0, k0, Nk)

    with ProcessPoolExecutor() as pool:
        H_fut = pool.map(Hmat, k, repeat(B), repeat(theta))
    H = np.array([fut for fut in H_fut])
    E = np.stack([eigh(H[ik], eigvals_only=True, lower=True) for ik in range(Nk)])

    Nmu = 200
    dmu = dmu*ip.e/1e3
    band_min = np.min(E)-0.3*ip.e/1e3
    band_max = np.min(E)+dmu
    chem_potential = np.linspace(band_min, band_max, Nmu)
    
    fermi = np.stack([fermi_derivative(k, E, mu, T, order=0) for mu in chem_potential])

    dos = np.stack([[fermi[imu,ik] for ik in range(Nk)] for imu in range(Nmu)])
    dos = np.sum([simps(dos[imu], k, axis=0) for imu in range(Nmu)],axis=1)/ip.R**2/np.pi**2
    dos = np.gradient(dos)

    return chem_potential/ip.e*1e3, dos*1e6*ip.e

if __name__=="__main__":

    EF = calc_fermi(1.0, 0.0, 2*np.pi/ip.R)
    print(EF)
    energy, dos = calc_DOS(0.2, 20, 0.0, 2*np.pi/ip.R)

    plt.ylabel('E [meV]')
    plt.xlabel(r'DOS [meV$^{-1} \cdot$cm$^{-2}$]')
    plt.axhline(y=EF, color='gray', linestyle='dotted')
    plt.plot(dos, energy, 'k')
    plt.show()