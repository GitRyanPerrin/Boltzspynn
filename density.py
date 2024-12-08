from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.linalg import eigh
from scipy.optimize import fsolve

import input_data as ip
from hamiltonian import Hmat, quantum_numbers
from hamiltonian_scroll import Hmat as Hmat_scroll
from fermi_derivative import FD, fermi_derivative

def calc_ndens(k, E, mu, T, B=ip.B, theta=ip.theta, k0=2*np.pi/ip.R, system='tube'):
    Nk = len(k)

    fermi = fermi_derivative(k, E, mu, T, order=0)

    ndens = np.stack([fermi[ik] for ik in range(Nk)])
    #ndens = np.sum(simps(ndens, k, axis=0),axis=0)/ip.R**2/np.pi**2
    ndens = np.sum(simpson(ndens, x=k, axis=0),axis=0)

    return ndens

def calc_ndens_phi(phi, k, E, V, mu, T, Nm=10, B=ip.B, theta=ip.theta, k0=2*np.pi/ip.R, system='tube'):
    Nk = len(k)

    fermi = fermi_derivative(k, E, mu, T, order=0)
    
    #prob = np.stack([V[ik].conj().T@V[ik] for ik in range(Nk)])
    #print(prob)
    #ndens = np.stack([prob[ik] for ik in range(Nk)])
    #ndens = simpson(ndens, x=k, axis=0)
    #ndens = simpson(fermi, x=k, axis=0)
    #G = simpson([V[ik].conj().T@V[ik]*fermi[ik] for ik in range(Nk)], x=k, axis=0)

    Nstat = 2*(2*Nm+1)

    '''
    prob = np.zeros([Nk, Nstat], dtype=complex)
    for ik in range(Nk):
        qn = quantum_numbers(Nm)
        for a1 in range(Nstat):
            m1 = qn[a1, 0]
            s1 = qn[a1, 1]
            for a2 in range(Nstat):
                m2 = qn[a2, 0]
                s2 = qn[a2, 1]

                prob[ik, a1] += fermi[ik,a1]*abs(V[ik, a2, a1])**2

    G = simpson(prob*fermi, x=k, axis=0)
    '''

    out = 0
    #out = np.zeros(Nstat, dtype=complex)
    qn = quantum_numbers(Nm)
    for a1 in range(Nstat):
        m1 = qn[a1, 0]
        s1 = qn[a1, 1]
        for a2 in range(Nstat):
            m2 = qn[a2, 0]
            s2 = qn[a2, 1]


            if s1 == s2: 
                #out += simpson([np.exp(1.0j*m2*phi)*V[ik,a2,a2]*np.exp(-1.0j*m1*phi)*V[ik,a1,a1] for ik in range(Nk)], x=k, axis=0)
                #out += np.exp(1.0j*m2*phi)*np.exp(-1.0j*m1*phi)*G[a1]
                out += np.exp(1.0j*m2*phi)*np.exp(-1.0j*m1*phi)*fermi[ik,a1]*abs(V[ik,a1,a2])**2

    return out.real

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
        ndens = np.sum(simpson(ndens, x=k, axis=0),axis=0)/ip.R**2/np.pi**2

        return ndens - 1.17e23
    
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
    dos = np.sum([simpson(dos[imu], x=k, axis=0) for imu in range(Nmu)],axis=1)/ip.R**2/np.pi**2
    dos = np.gradient(dos)

    return chem_potential/ip.e*1e3, dos*1e6*ip.e

if __name__=="__main__":

    Nk = 1000
    k0 = 2*np.pi/ip.R
    k = np.linspace(-k0, k0, Nk)
    system = 'tube'
    Nm = 10

    with ProcessPoolExecutor() as pool:
        if system == 'tube':
            H_fut = pool.map(Hmat,
                k,
                repeat(ip.B),
                repeat(ip.theta),
                repeat(Nm),
                repeat(ip.alphaR)
            )
        if system == 'scroll':
            H_fut = pool.map(Hmat_scroll,
                k,
                repeat(ip.B),
                repeat(ip.theta),
                repeat(Nm),
                repeat(ip.alphaR)
            )
    H = np.array([fut for fut in H_fut])
    E = np.zeros(Nk, dtype=object)
    V = np.zeros(Nk, dtype=object)
    for ik in range(Nk):
        E[ik], V[ik] = eigh(H[ik], lower=True)
    E = np.stack(E)
    V = np.stack(V)
    norm = np.stack([np.real(np.sum(V[ik]@V[ik].conj().T)) for ik in range(Nk)])

    mu = 10*ip.e/1e3
    T = 1.0

    phi = np.linspace(0, 2*np.pi, 50)
    ndens = np.stack([calc_ndens_phi(p, k, E, V, mu, T, B=ip.B, theta=ip.theta, k0=2*np.pi/ip.R, system='tube', Nm=Nm) for p in phi])
    
    plt.plot(phi/np.pi, ndens/ip.R**2/np.pi**2)
    plt.xlim([0,2])
    #plt.plot(phi/np.pi, ndens.imag/ip.R**2/np.pi**2)
    plt.show()