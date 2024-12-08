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

def svB(system, polarization):

    k0 = 2*np.pi/ip.R
    Nk = 1000
    k = np.linspace(-k0, k0, Nk)

    NB = 30
    B0 = 0.50
    B = np.linspace(0, B0, NB)
    H = np.zeros(NB, dtype=object)
    E = np.zeros([NB, Nk], dtype=object)
    V = np.zeros([NB, Nk], dtype=object)
    for ib, b in enumerate(B):
        with ProcessPoolExecutor() as pool:
            H_fut = pool.map(Hmat, k, repeat(b), repeat(ip.theta))
        H[ib] = np.array([fut for fut in H_fut])

    H = np.stack(H, axis=0)

    for ib in range(NB):
        for ik in range(Nk):
            E[ib, ik], V[ib, ik] = eigh(H[ib, ik])
    Nstat = E[0,0].shape[0]
    E = np.stack(np.hstack(E)).reshape([NB, Nk, Nstat])
    V = np.stack(np.hstack(V)).reshape([NB, Nk, Nstat, Nstat])
    norm = np.stack([[np.real(np.sum(V[ib,ik]@V[ib,ik].conj().T)) for ik in range(Nk)] for ib in range(NB)])

    T = 0.1
    mu = [-0.2*ip.e/1e3, 0.0, 0.2*ip.e/1e3, 0.9*ip.e/1e3]
    c = ['b', 'r', 'k', 'g']

    v = np.gradient(H, axis=1)/ip.hbar
    v = np.stack([[np.diagonal(V[ib, ik].conj().T@v[ib, ik]@V[ib, ik]/norm[ib,ik]) for ik in range(Nk)] for ib in range(NB)])

    for ib, b in enumerate(B):
        with ProcessPoolExecutor() as pool:
            if system == 'tube' and polarization == 'p':
                v_fut = pool.map(vzp_tube, k)
            if system == 'tube' and polarization == 'z':
                return np.zeros(NB)
            if system == 'scroll' and polarization == 'r':
                v_fut = pool.map(vzr_scroll, k)
            if system == 'scroll' and polarization == 'p':
                v_fut = pool.map(vzp_scroll, k)
            if system == 'scroll' and polarization == 'z':
                v_fut = pool.map(vzz_scroll, k)
        
    vs = np.stack([fut for fut in v_fut])
    vs = np.stack([[np.diagonal(V[ib, ik].conj().T@vs[ik]@V[ib, ik])*ip.hbar/2/norm[ib,ik] for ik in range(Nk)] for ib in range(NB)])

    cond = np.zeros(4, dtype=object)
    scond = np.zeros(4, dtype=object)
    for imu, m in enumerate(mu):
        cond[imu], scond[imu] = calc_conds(k, v, vs, E, m, T)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    plt.autoscale()
 
    ax1.set_ylabel(r"$\sigma_e^{(1)}$")
    ax1.set_xlabel(r"$B$ [T]")

    ax2.set_ylabel(r"$\sigma_\sigma^{\varphi(2)}$")
    ax2.set_xlabel(r"$B$ [T]")

    [ax1.plot(B, cond[imu]/1e9, 'k') for imu in range(4)]
    [ax2.plot(B, scond[imu]/1e9, 'k') for imu in range(4)]
    plt.show()

def contour(system, polarization):

    k0 = 2*np.pi/ip.R
    Nk = 2000
    k = np.linspace(-k0, k0, Nk)

    NB = 80
    B0 = 0.50
    B = np.linspace(0, B0, NB)
    H = np.zeros(NB, dtype=object)
    E = np.zeros([NB, Nk], dtype=object)
    V = np.zeros([NB, Nk], dtype=object)
    for ib, b in enumerate(B):
        with ProcessPoolExecutor() as pool:
            H_fut = pool.map(Hmat, k, repeat(b), repeat(ip.theta))
        H[ib] = np.array([fut for fut in H_fut])

    H = np.stack(H, axis=0)

    for ib in range(NB):
        for ik in range(Nk):
            E[ib, ik], V[ib, ik] = eigh(H[ib, ik])
    Nstat = E[0,0].shape[0]
    E = np.stack(np.hstack(E)).reshape([NB, Nk, Nstat])
    V = np.stack(np.hstack(V)).reshape([NB, Nk, Nstat, Nstat])
    norm = np.stack([[np.real(np.sum(V[ib,ik]@V[ib,ik].conj().T)) for ik in range(Nk)] for ib in range(NB)])

    #V = np.stack(V, axis=0)

    T = 0.1
    #mu = calc_fermi(T, 0.0, ip.theta)
    #mu = 0.1*ip.e/1e3
    Nmu = 80
    #hem_potential = [-0.1*ip.e/1e3, 0.0, 0.1*ip.e/1e3, 0.2*ip.e/1e3]
    mu_max = 0.6
    mu_min = -0.4
    chem_potential = np.linspace(mu_min*ip.e/1e3, mu_max*ip.e/1e3, Nmu)
    colors = ['r', 'k', 'b', 'g']
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    plt.autoscale()
 
    v = np.gradient(H, axis=1)/ip.hbar
    v = np.stack([[np.diagonal(V[ib, ik].conj().T@v[ib, ik]@V[ib, ik]/norm[ib,ik]) for ik in range(Nk)] for ib in range(NB)])

    for ib, b in enumerate(B):
        with ProcessPoolExecutor() as pool:
            if system == 'tube' and polarization == 'p':
                v_fut = pool.map(vzp_tube, k)
            if system == 'tube' and polarization == 'z':
                return np.zeros(NB)
            if system == 'scroll' and polarization == 'r':
                v_fut = pool.map(vzr_scroll, k)
            if system == 'scroll' and polarization == 'p':
                v_fut = pool.map(vzp_scroll, k)
            if system == 'scroll' and polarization == 'z':
                v_fut = pool.map(vzz_scroll, k)
        
    vs = np.stack([fut for fut in v_fut])

    vs = np.stack([[np.diagonal(V[ib, ik].conj().T@vs[ik]@V[ib, ik])*ip.hbar/2/norm[ib,ik] for ik in range(Nk)] for ib in range(NB)])

    with ProcessPoolExecutor() as pool:
        conds_fut = pool.map(
            calc_conds,
            repeat(k),
            repeat(v),
            repeat(vs),
            repeat(E),
            chem_potential,
            repeat(T)
        )

    conds = np.stack([fut for fut in conds_fut])

    cond = conds[:,0]
    scond = conds[:,1]

    for ib in range(NB):
        for imu in range(Nmu):
            cond[imu, ib] = np.real(cond[imu, ib])
            scond[imu, ib] = np.real(scond[imu, ib])

    cond = np.array(cond, dtype=float)
    scond = np.array(scond, dtype=float)

    ax1.set_ylim([chem_potential[0]/ip.e*1e3, chem_potential[-1]/ip.e*1e3])
    ax1.set_ylabel(r"$\mu$ [meV]")
    ax1.set_xlabel(r"$B$ [T]")

    ax2.set_ylim([chem_potential[0]/ip.e*1e3, chem_potential[-1]/ip.e*1e3])
    ax2.set_ylabel(r"$\mu$ [meV]")
    ax2.set_xlabel(r"$B$ [T]")

    ax3.set_ylim([chem_potential[0]/ip.e*1e3, chem_potential[-1]/ip.e*1e3])
    ax3.set_ylabel(r"E [meV]")
    ax3.set_xlabel(r"$B$ [T]")

    #[ax1.plot(B, cond[imu]/1e9) for imu in range(Nmu)]
    #[ax2.plot(B, scond[imu]/1e9) for imu in range(Nmu)]
    e = ax1.contourf(B.ravel(), chem_potential.ravel()/ip.e*1e3, cond/1e9, levels=500, cmap='gray')
    s = ax2.contourf(B.ravel(), chem_potential.ravel()/ip.e*1e3, scond/1e9, levels=500, cmap='gray')
    ax3.plot(B, E[:, int(Nk/2)]/ip.e*1e3, 'k')
    ecb = fig.colorbar(e)
    scb = fig.colorbar(s)
    ecb.ax.set_title(r"$\sigma_e$ [S/mm]")
    scb.ax.set_title(r"$\sigma_\sigma^z$ [meV/m$^2$]")
    plt.show()

if __name__=="__main__":
    svB(system='tube', polarization='p')
