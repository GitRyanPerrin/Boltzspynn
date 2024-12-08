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

def calc_band_structure(k, system='tube'):

    Nk = len(k)

    with ProcessPoolExecutor() as pool:
        if system == 'tube':
            H_fut = pool.map(Hmat_tube, k, repeat(ip.B))
        if system == 'scroll':
            H_fut = pool.map(Hmat_scroll, k, repeat(ip.B))
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

def calc_charge_correction(k, v, E, V, chem_potential, T=0.1):

    Nk = len(k)
    Nmu = len(chem_potential)
    Na = V.shape[1]

    fermi = np.stack([fermi_derivative(k, E, mu, T, order=2) for mu in chem_potential])

    cond = np.stack([[v[ik]*v[ik]*fermi[imu,ik] for ik in range(Nk)] for imu in range(Nmu)])

    sz = np.zeros([Na, Na])
    qn = quantum_numbers()
    for a1 in range(Na):
        m1 = qn[a1,0]
        s1 = qn[a1,1]
        for a2 in range(Na):
            m2 = qn[a1,0]
            s2 = qn[a2,1]

            if m1 == m2:
                if s1 == 1 and s2 == s1:
                    sz[a1, a2] = 1
                if s1 == -1 and s2 == s1:
                    sz[a1, a2] = -1
    sz = np.stack([np.diagonal(V[ik].conj().T@sz@V[ik]) for ik in range(Nk)])

    print(sz.shape, cond.shape)
    cond = np.stack([-simps(sz*cond[imu], k, axis=0) for imu in range(Nmu)])
    print(cond.shape)
    cond = np.sum(cond, axis=1)

    coeff = ip.e**2*ip.tau_e/2/np.pi/ip.area

    return cond*coeff

def calc_spin_conductivity(k, v, E, V, norm, chem_potential, T=0.1, system='tube', polarization='p'):

    Nk = len(k)
    Nmu = len(chem_potential)

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
    
    fermi = np.stack([fermi_derivative(k, E, mu, T, order=2) for mu in chem_potential])

    cond = np.stack([[v[ik]*v[ik]*vs[ik]*fermi[imu,ik] for ik in range(Nk)] for imu in range(Nmu)])
    cond = np.sum([simps(cond[imu], k, axis=0) for imu in range(Nmu)],axis=1)

    coeff = ip.e**2*ip.tau_s**2/2/np.pi/ip.area*ip.hbar/4/ip.e/1e3

    return cond*coeff

def calc_spin_correction(k, v, E, V, norm, chem_potential, T=0.1, system='tube', polarization='p'):

    Nk = len(k)
    Nmu = len(chem_potential)
    Na = V.shape[1]

    with ProcessPoolExecutor() as pool:
        if system == 'tube' and polarization == 'r':
            v_fut = pool.map(vzr_tube, k)
        if system == 'tube' and polarization == 'p':
            v_fut = pool.map(vzp_tube, k)
        if system == 'tube' and polarization == 'z':
            #return np.zeros(Nmu)
            v_fut = pool.map(vzz_tube, k)
        if system == 'scroll' and polarization == 'r':
            v_fut = pool.map(vzr_scroll, k)
        if system == 'scroll' and polarization == 'p':
            v_fut = pool.map(vzp_scroll, k)
        if system == 'scroll' and polarization == 'z':
            v_fut = pool.map(vzz_scroll, k)
        
    vs = np.stack([fut for fut in v_fut])

    vs = np.stack([V[ik].conj().T@vs[ik]@V[ik]/norm[ik] for ik in range(Nk)])
    vs = np.stack([np.diagonal(vs[ik])*ip.hbar/2 for ik in range(Nk)])
    

    fermi = np.stack([fermi_derivative(k, E, mu, T, order=3) for mu in chem_potential])

    cond = np.stack([[vs[ik]*v[ik]*v[ik]*fermi[imu,ik] for ik in range(Nk)] for imu in range(Nmu)])

    sz = np.zeros([Na, Na])
    qn = quantum_numbers()
    for a1 in range(Na):
        m1 = qn[a1,0]
        s1 = qn[a1,1]
        for a2 in range(Na):
            m2 = qn[a1,0]
            s2 = qn[a2,1]

            if m1 == m2:
                if s1 == 1 and s2 == s1:
                    sz[a1, a2] = 1
                if s1 == -1 and s2 == s1:
                    sz[a1, a2] = -1
    sz = np.stack([np.diagonal(V[ik].conj().T@sz@V[ik]) for ik in range(Nk)])

    print(sz.shape, cond.shape)
    cond = np.stack([-simps(sz*cond[imu], k, axis=0) for imu in range(Nmu)])
    print(cond.shape)
    cond = np.sum(cond, axis=1)

    coeff = ip.e**2*ip.tau_e/2/np.pi/ip.area

    return cond*coeff

def electric_conds(system='tube'):

    if ip.B < 1.0:
        k0 = 2*np.pi/ip.R
    elif ip.B >= 1.0 and ip.B < 3.0:
        k0 = 4*np.pi/ip.R
    else:
        k0 = 7*np.pi/ip.R
    Nk = 3000
    k = np.linspace(-k0, k0, Nk)
    E, V, v, norm = calc_band_structure(k, system=system)

    T = 0.1
    Nmu = 150
    dmu = 4*ip.e/1e3
    band_min = np.min(E)-0.1*ip.e/1e3
    band_max = np.min(E)+dmu
    chem_potential = np.linspace(band_min, band_max, Nmu)

    #charge_cond = calc_charge_conductivity(k, v, E, V, chem_potential, T=T)
    #charge_cond = calc_charge_correction(k, v, E, V, chem_potential, T=T)
    #charge_cond = calc_spin_correction(k, v, E, V, norm, chem_potential, T=T, system=system, polarization='r')
    spin_cond_p = calc_spin_conductivity(k, v, E, V, norm, chem_potential, T=T, system=system, polarization='p')
    #spin_cond_z = calc_spin_conductivity(k, v, E, V, norm, chem_potential, T=T, system=system, polarization='z')
    ndens = lambda mu: calc_ndens(k, E, mu*ip.e/1e3, T=T)/2/np.pi**2/ip.R/1e15

    #fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize=(6,6))
    #fig, (ax1, ax2, ax4) = plt.subplots(3,1,figsize=(6,6))
    fig, ax1 = plt.subplots()
    ax1_2 = ax1.twiny()
    plt.autoscale()
    
    mu_ticks = [chem_potential[i]/ip.e*1e3 for i in np.arange(0,Nmu,10)]
    mu_labels = [round(chem_potential[i]/ip.e*1e3,1) for i in np.arange(0,Nmu,10)]

    ax1.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
    ax1_2.set_xlim(ndens(band_min/ip.e*1e3), ndens(band_max/ip.e*1e3))
    #ax2.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
    #ax3.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
    #ax4.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])

    ax1.set_ylabel(r"$\sigma_e$ [S/mm]")
    #ax1_2.set_xlabel(r'n$_{2D}$ [$10^{15}$ 1/m$^2$]')
    ax1_2.set_xlabel(r'n$_{2D}$ [$10^{11}$ 1/cm$^2$]')
    #ax2.set_ylabel(r"$\sigma_\sigma^\varphi$ [meV/m$^2$]")    
    #ax3.set_ylabel(r"$\sigma_\sigma^z$ [meV/m$^2$]")
    ax1.set_xlabel(r"$\mu$ [meV]")
    #ax4.set_ylabel(r"$k_zR$")
    #ax4.set_xlabel(r"E($k_z$) [meV]")
    
    ax1.set_xticks(mu_ticks, labels=mu_labels)
    #ax2.set_xticks(mu_ticks, labels=mu_labels)
    #ax3.set_xticks(mu_ticks, labels=mu_labels)
    #ax4.set_xticks(mu_ticks, labels=mu_labels)

    #ax1.plot(chem_potential/ip.e*1e3, charge_cond/1e9, 'k')
    #ax2.plot(chem_potential/ip.e*1e3, spin_cond_p/1e9, 'k')
    ax1.plot(chem_potential/ip.e*1e3, spin_cond_p/1e9, 'k')
    #ax3.plot(chem_potential/ip.e*1e3, spin_cond_z/1e9, 'k')
    #ax4.plot(E[int(Nk/2):]/ip.e*1e3, k[int(Nk/2):]*ip.R, 'k')

    plt.show()

if __name__=="__main__":
    
    electric_conds(system='tube')
