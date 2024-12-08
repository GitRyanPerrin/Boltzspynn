from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import simpson

import input_data as ip
from hamiltonian import Hmat
from fermi_derivative import fermi_derivative
from density import calc_fermi
from calc_conductivities import calc_charge_conductivity
from spin_cond_111 import vzp as vzp_tube
from spin_cond_scroll import vzr as vzr_scroll
from spin_cond_scroll import vzp as vzp_scroll
from spin_cond_scroll import vzz as vzz_scroll

def calc_conds(k, v, vs, E, mu, T):

    Nk = len(k)
    NB = E.shape[0]

    fermi = np.stack([fermi_derivative(k, E[ib], mu, T, order=1) for ib in range(NB)])

    cond = np.stack([[v[ib, ik]*v[ib, ik]*fermi[ib, ik] for ik in range(Nk)] for ib in range(NB)])
    cond = -np.sum([simpson(cond[ib], x=k, axis=0) for ib in range(NB)],axis=1)
    coeff = ip.e**2*ip.tau_e/2/np.pi/ip.area

    fermi = np.stack([fermi_derivative(k, E[ib], mu, T, order=2) for ib in range(NB)])

    scond = np.stack([[v[ib,ik]*v[ib,ik]*vs[ib,ik]*fermi[ib,ik] for ik in range(Nk)] for ib in range(NB)])
    scond = np.sum([simpson(scond[ib], x=k, axis=0) for ib in range(NB)],axis=1)
    scoeff = ip.e**2*ip.tau_s**2/2/np.pi/ip.area*ip.hbar/4/ip.e/1e3

    #return np.array([cond*coeff, scond*scoeff], dtype=object)
    return cond*coeff, scond*scoeff

def svA(system, polarization):

    k0 = 2*np.pi/ip.R
    Nk = 2000
    k = np.linspace(-k0, k0, Nk)

    NA = 10
    A0 = ip.alphaR
    A = np.linspace(0, A0, NA)
    H = np.zeros(NA, dtype=object)
    E = np.zeros([NA, Nk], dtype=object)
    V = np.zeros([NA, Nk], dtype=object)
    for ia, a in enumerate(A):
        with ProcessPoolExecutor() as pool:
            H_fut = pool.map(Hmat,
                k, 
                repeat(0), 
                repeat(ip.theta),
                repeat(10),
                A
            )
        H[ia] = np.array([fut for fut in H_fut])

    H = np.stack(H, axis=0)

    for ia in range(NA):
        for ik in range(Nk):
            E[ia, ik], V[ia, ik] = eigh(H[ia, ik])
    Nstat = E[0,0].shape[0]
    E = np.stack(np.hstack(E)).reshape([NA, Nk, Nstat])
    V = np.stack(np.hstack(V)).reshape([NA, Nk, Nstat, Nstat])
    norm = np.stack([[np.real(np.sum(V[ia,ik]@V[ia,ik].conj().T)) for ik in range(Nk)] for ia in range(NA)])

    T = 0.1
    mu = [-0.2*ip.e/1e3, 0.0, 0.2*ip.e/1e3, 0.9*ip.e/1e3]
    c = ['b', 'r', 'k', 'g']

    v = np.gradient(H, axis=1)/ip.hbar
    v = np.stack([[np.diagonal(V[ib, ik].conj().T@v[ib, ik]@V[ib, ik]/norm[ib,ik]) for ik in range(Nk)] for ib in range(NA)])

    for ib, b in enumerate(A):
        with ProcessPoolExecutor() as pool:
            if system == 'tube' and polarization == 'p':
                v_fut = pool.map(vzp_tube, k)
            if system == 'tube' and polarization == 'z':
                return np.zeros(NA)
            if system == 'scroll' and polarization == 'r':
                v_fut = pool.map(vzr_scroll, k)
            if system == 'scroll' and polarization == 'p':
                v_fut = pool.map(vzp_scroll, k)
            if system == 'scroll' and polarization == 'z':
                v_fut = pool.map(vzz_scroll, k)
        
    vs = np.stack([fut for fut in v_fut])
    vs = np.stack([[np.diagonal(V[ia, ik].conj().T@vs[ik]@V[ia, ik])*ip.hbar/2/norm[ia,ik] for ik in range(Nk)] for ia in range(NA)])

    cond = np.zeros(4, dtype=object)
    scond = np.zeros(4, dtype=object)
    for imu, m in enumerate(mu):
        cond[imu], scond[imu] = calc_conds(k, v, vs, E, m, T)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    plt.autoscale()
 
    ax1.set_ylabel(r"$\sigma_e^{(1)}$")

    ax2.set_ylabel(r"$\sigma_\sigma^{\varphi(2)}$")
    ax2.set_xlabel(r"$\alpha$")

    [ax1.plot(A, cond[imu]/1e9, 'k') for imu in range(4)]
    [ax2.plot(A, scond[imu]/1e9, 'k') for imu in range(4)]
    plt.show()

if __name__=="__main__":
    svA(system='tube', polarization='p')
