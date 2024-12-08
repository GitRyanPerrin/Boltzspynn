from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import simpson

import input_data as ip
from hamiltonian import Hmat as Hmat_tube
from hamiltonian_scroll import Hmat as Hmat_scroll
from fermi_derivative import fermi_derivative
from density import calc_fermi, calc_ndens
from spin_cond_111 import vzr as vzr_tube
from spin_cond_111 import vzp as vzp_tube
from spin_cond_111 import vzz as vzz_tube
from spin_cond_scroll import vzr as vzr_scroll
from spin_cond_scroll import vzp as vzp_scroll
from spin_cond_scroll import vzz as vzz_scroll
from calc_thermal import calc_charge_thermo, calc_spin_thermo, calc_ballistic_thermo

def calc_band_structure(k, system='tube', B=ip.B, theta=ip.theta, Nm=10, alphaR=ip.alphaR, betaD=ip.betaD):

    Nk = len(k)

    with ProcessPoolExecutor(max_workers=10) as pool:
        if system == 'tube':
            H_fut = pool.map(Hmat_tube,
                k,
                repeat(B),
                repeat(theta),
                repeat(Nm),
                repeat(alphaR)
            )
        if system == 'scroll':
            H_fut = pool.map(Hmat_scroll,
                k,
                repeat(B),
                repeat(theta),
                repeat(Nm),
                repeat(alphaR)
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

    #np.savez_compressed(f"./hamiltonians/{system}_{ip.material}_50_{ip.alphaR/ip.e*1e12}_{ip.betaD/ip.e*1e12}_{ip.B}_{ip.theta/np.pi}.npz", k, E, V, v, norm)

    return E, V, v, norm

def calc_charge_conductivity(k, v, E, V, chem_potential, T=0.1, system='tube'):

    Nk = len(k)
    Nmu = len(chem_potential)

    fermi = np.stack([fermi_derivative(k, E, mu, T, order=1) for mu in chem_potential])

    cond = np.stack([[v[ik]*v[ik]*fermi[imu,ik] for ik in range(Nk)] for imu in range(Nmu)])
    cond = -np.sum([simpson(cond[imu], x=k, axis=0) for imu in range(Nmu)],axis=1)

    coeff = ip.e**2*ip.tau_e/2/np.pi/ip.area

    #np.savez_compressed(f"./conductivities/elec_charge_{system}_{ip.material}_50_{ip.alphaR/ip.e*1e12}_{ip.betaD/ip.e*1e12}_{ip.B}_{ip.theta/np.pi}.npz", chem_potential, cond)


    return cond*coeff

def calc_ballistic_conductivity(k, v, E, V, chem_potential, T=0.1, system='tube'):

    Nk = len(k)
    Nmu = len(chem_potential)

    vp = np.where(v > 0, v, 0)
    vn = np.where(v < 0, v, 0)

    fermi = np.stack([fermi_derivative(k, E, mu, T, order=1) for mu in chem_potential])

    cond = np.stack([[(vp[ik]-vn[ik])*fermi[imu,ik] for ik in range(Nk)] for imu in range(Nmu)])
    cond = -np.sum([simpson(cond[imu], x=k, axis=0) for imu in range(Nmu)],axis=1)

    coeff = ip.e**2*ip.tau_e/2/np.pi/ip.area

    #np.savez_compressed(f"./conductivities/elec_charge_ball_{system}_{ip.material}_50_{ip.alphaR/ip.e*1e12}_{ip.betaD/ip.e*1e12}_{ip.B}_{ip.theta/np.pi}.npz", chem_potential, cond)


    return cond*coeff

def calc_spin_conductivity(k, v, E, V, norm, chem_potential, T=0.1, system='tube', polarization='p'):

    Nk = len(k)
    Nmu = len(chem_potential)

    with ProcessPoolExecutor(max_workers=10) as pool:
        if system == 'tube' and polarization == 'r':
            v_fut = pool.map(vzr_tube, k)
        if system == 'tube' and polarization == 'p':
            v_fut = pool.map(vzp_tube, k)
        if system == 'tube' and polarization == 'z':
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
    
    fermi = np.stack([fermi_derivative(k, E, mu, T, order=2) for mu in chem_potential])

    cond = np.stack([[v[ik]*v[ik]*vs[ik]*fermi[imu,ik] for ik in range(Nk)] for imu in range(Nmu)])
    cond = np.sum([simpson(cond[imu], x=k, axis=0) for imu in range(Nmu)],axis=1)

    coeff = ip.e**2*ip.tau_s**2/2/np.pi/ip.area*ip.hbar/4/ip.e/1e3
    #coeff = 1
    cond = cond*coeff

    #np.savez_compressed(f"./conductivities/elec_spin_{system}_{polarization}_{ip.material}_50_{ip.alphaR/ip.e*1e12}_{ip.betaD/ip.e*1e12}_{ip.B}_{ip.theta/np.pi}.npz", chem_potential, cond)

    return cond

def calc_spin_ballistic(k, v, E, V, norm, chem_potential, T=0.1, system='tube', polarization='p'):

    Nk = len(k)
    Nmu = len(chem_potential)

    vp = np.where(v > 0, v, 0)
    vn = np.where(v < 0, v, 0)

    with ProcessPoolExecutor(max_workers=10) as pool:
        if system == 'tube' and polarization == 'r':
            v_fut = pool.map(vzr_tube, k)
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

    cond = np.stack([[(vp[ik]-vn[ik])*vs[ik]*fermi[imu,ik] for ik in range(Nk)] for imu in range(Nmu)])
    cond = np.sum([simpson(cond[imu], x=k, axis=0) for imu in range(Nmu)],axis=1)

    coeff = ip.e**2*ip.tau_s**2/2/np.pi/ip.area*ip.hbar/4/ip.e/1e3
    cond = cond*coeff

    #np.savez_compressed(f"./conductivities/elec_spin_{system}_{polarization}_{ip.material}_50_{ip.alphaR/ip.e*1e12}_{ip.betaD/ip.e*1e12}_{ip.B}_{ip.theta/np.pi}.npz", chem_potential, cond)

    return cond

def first_order(system):
    
    if ip.B < 1.0:
        k0 = 2*np.pi/ip.R
    elif ip.B >= 1.0 and ip.B < 3.0:
        k0 = 4*np.pi/ip.R
    else:
        k0 = 7*np.pi/ip.R
    Nk = 4000
    k = np.linspace(-k0, k0, Nk)
    E, V, v, norm = calc_band_structure(k, system=system)

    T = 0.1
    Nmu = 150
    dmu = 0.3*ip.e/1e3
    band_min = np.min(E)-0.1*ip.e/1e3
    band_max = np.min(E)+dmu
    chem_potential = np.linspace(band_min, band_max, Nmu)

    charge_cond = calc_charge_conductivity(k, v, E, V, chem_potential, T=T, system=system)
    charge_thermo = calc_charge_thermo(k, v, E, V, chem_potential, T=T, system=system)
    #charge_thermo = calc_ballistic_thermo(k, v, E, V, chem_potential, T=T, system=system)
    ballistic_cond = calc_ballistic_conductivity(k, v, E, V, chem_potential, T=T, system=system)
    #3D: ndens = lambda mu: calc_ndens(k, E, mu*ip.e/1e3, T=T)/ip.area/1e23/2/np.pi
    ndens = lambda mu: calc_ndens(k, E, mu*ip.e/1e3, T=T)/2/np.pi**2/ip.R/1e15

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(6,8), sharex=True)
    ax1_G = ax1.twinx()
    #ax1_G.spines['right'].set_color('blue')
    #ax1_G.yaxis.label.set_color('blue')
    #ax1_G.tick_params(axis='y', colors='blue')
    ax1_n = ax1.twiny()
    plt.autoscale()
    
    mu_ticks = np.linspace(band_min/ip.e*1e3, band_max/ip.e*1e3, 6)
    mu_labels = [round(tick,1) for tick in mu_ticks]
    #mu_ticks = [chem_potential[i]/ip.e*1e3 for i in np.arange(0,Nmu,10)]
    #mu_labels = [round(chem_potential[i]/ip.e*1e3,1) for i in np.arange(0,Nmu,10)]

    ax1.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
    ax1_n.set_xlim(ndens(band_min/ip.e*1e3), ndens(band_max/ip.e*1e3))
    ax2.set_ylim([-25,45])
    n_ticks = np.linspace(ndens(band_min/ip.e*1e3), ndens(band_max/ip.e*1e3), 6)
    n_labels = [round(tick,2) for tick in n_ticks]
    
    ax1.set_ylabel(r"$\sigma_e$ [S/mm]", fontsize=22)
    ax1_G.set_ylabel(r"$G_e$ [$e/h$]", fontsize=22)
    ax1_n.set_xlabel(r'n$_{2D}$ [$10^{11}$ cm$^{-2}$]', fontsize=22)
    #ax2.set_ylabel(r"$S_e$ [$k_B/e$]", fontsize=22)
    ax2.set_ylabel(r"$S_e$ [$\mu V/K$]", fontsize=22)
    ax2.set_xlabel(r"$\mu$ [meV]", fontsize=22)
    
    ax1_n.set_xticks(n_ticks, labels=n_labels)
    ax2.set_xticks(mu_ticks, labels=mu_labels)

    ax1.plot(chem_potential/ip.e*1e3, charge_cond/1e9, 'k')
    #ax1_G.plot(chem_potential/ip.e*1e3, ballistic_cond/ballistic_cond[int(Nmu/3)], 'b')
    ax1_G.plot(chem_potential/ip.e*1e3, ballistic_cond/18.5, 'k', linestyle='dashed')

    ax2.plot(chem_potential/ip.e*1e3, charge_thermo/charge_cond*1e6, 'k')
    #ax2.plot(chem_potential/ip.e*1e3, charge_thermo/charge_cond/ip.lorseeb, 'k')

    b0 = E[int(Nk/2), 0]/ip.e*1e3
    b1 = E[int(Nk/2), 1]/ip.e*1e3
    b2 = E[int(Nk/2), 2]/ip.e*1e3
    b3 = E[int(Nk/2), 3]/ip.e*1e3
    b4 = E[int(Nk/2), 4]/ip.e*1e3
    b5 = E[int(Nk/2), 5]/ip.e*1e3
    b6 = E[int(Nk/2), 6]/ip.e*1e3
    b7 = E[int(Nk/2), 7]/ip.e*1e3

    m0 = np.min(E[:, 0])/ip.e*1e3
    m1 = np.min(E[:, 1])/ip.e*1e3
    m5 = np.min(E[:, 5])/ip.e*1e3
    m6 = np.min(E[:, 6])/ip.e*1e3

    ax1.axvspan(m0, b1, ymin=0, ymax=1, color='lightgray', alpha=0.3)
    ax1.axvspan(m5, b6, ymin=0, ymax=1, color='lightgray', alpha=0.3)
    ax1.axvspan(b1, b2, ymin=0, ymax=1, color='lightgray', alpha=0.8)
    ax1.axvspan(b5, b7, ymin=0, ymax=1, color='lightgray', alpha=0.8)
    #ax1.axvspan(m1, b1, ymin=0, ymax=1, color='lightgray', alpha=0.3)
    #ax1.axvspan(m1, b1, ymin=0, ymax=1, color='lightgray', alpha=0.3)
    #ax1.axvspan(b0, b3, ymin=0, ymax=1, color='lightgray', alpha=0.5)

    ax2.text(x=m0-0.08, y=47, s=r'$\Delta E_{min}^{1/2}$')
    ax2.text(x=m5-0.07, y=47, s=r'$\Delta E_{min}^{3/2}$')
    ax2.text(x=b1+0.04, y=47, s=r'$\Delta E_{SO}^{1/2}$')
    ax2.text(x=b5+0.06, y=47, s=r'$\Delta E_{SO}^{3/2}$')
    ax2.axvspan(m0, b1, ymin=0, ymax=1, color='lightgray', alpha=0.3)
    ax2.axvspan(m5, b6, ymin=0, ymax=1, color='lightgray', alpha=0.3)
    ax2.axvspan(b1, b2, ymin=0, ymax=1, color='lightgray', alpha=0.8)
    ax2.axvspan(b5, b7, ymin=0, ymax=1, color='lightgray', alpha=0.8)
    #ax2.axvspan(m1, b1, ymin=0, ymax=1, color='lightgray', alpha=0.3)
    #ax2.axvspan(b0, b3, ymin=0, ymax=1, color='lightgray', alpha=0.5)

    ax1.axvline(x=b0, color='gray', linewidth=0.5)
    ax1.axvline(x=b1, color='gray', linewidth=0.5)
    ax1.axvline(x=b2, color='gray', linewidth=0.5)
    ax1.axvline(x=b3, color='gray', linewidth=0.5)
    ax1.axvline(x=b4, color='gray', linewidth=0.5)
    ax1.axvline(x=b5, color='gray', linewidth=0.5)
    ax1.axvline(x=b6, color='gray', linewidth=0.5)
    ax1.axvline(x=m0, color='gray', linewidth=0.5)
    ax1.axvline(x=m5, color='gray', linewidth=0.5)
    
    ax2.axvline(x=b0, color='gray', linewidth=0.5)
    ax2.axvline(x=b1, color='gray', linewidth=0.5)
    ax2.axvline(x=b2, color='gray', linewidth=0.5)
    ax2.axvline(x=b3, color='gray', linewidth=0.5)
    ax2.axvline(x=b4, color='gray', linewidth=0.5)
    ax2.axvline(x=b5, color='gray', linewidth=0.5)
    ax2.axvline(x=b6, color='gray', linewidth=0.5)
    ax2.axvline(x=m0, color='gray', linewidth=0.5)
    ax2.axvline(x=m5, color='gray', linewidth=0.5)

    fig.align_labels()
    plt.subplots_adjust(left=0.132, hspace=0.09, wspace=0.515, right=0.88, top=0.92, bottom=0.077)
    plt.tight_layout()
    plt.show()

def second_order(system):
    
    if ip.B < 1.0:
        k0 = 2*np.pi/ip.R
    elif ip.B >= 1.0 and ip.B < 3.0:
        k0 = 4*np.pi/ip.R
    else:
        k0 = 7*np.pi/ip.R
    Nk = 8000
    k = np.linspace(-k0, k0, Nk)
    E, V, v, norm = calc_band_structure(k, system=system)

    T = 0.1
    Nmu = 200
    dmu = 2.0*ip.e/1e3
    band_min = np.min(E)-0.1*ip.e/1e3
    band_max = np.min(E)+dmu
    chem_potential = np.linspace(band_min, band_max, Nmu)

    spin_cond = calc_spin_conductivity(k, v, E, V, norm, chem_potential, T=T, system=system, polarization='p')
    spin_thermo = calc_spin_thermo(k, v, E, V, norm, chem_potential, T=T, system=system, polarization='p')
    ndens = lambda mu: calc_ndens(k, E, mu*ip.e/1e3, T=T)/2/np.pi**2/ip.R/1e15

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(6,8), sharex=True)
    #fig, (ax1, ax3) = plt.subplots(2,1, sharex=True)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=20)
    #fig, ax1 = plt.subplots()
    ax1_n = ax1.twiny()
    plt.autoscale()

    mu_ticks = np.linspace(band_min/ip.e*1e3, band_max/ip.e*1e3, 6)
    mu_labels = [round(tick,1) for tick in mu_ticks]
    #mu_ticks = [chem_potential[i]/ip.e*1e3 for i in np.arange(0,Nmu,5)]
    #mu_labels = [round(chem_potential[i]/ip.e*1e3,1) for i in np.arange(0,Nmu,5)]

    ax1.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
    ax1_n.set_xlim(ndens(band_min/ip.e*1e3), ndens(band_max/ip.e*1e3))
    n_ticks = np.linspace(ndens(band_min/ip.e*1e3), ndens(band_max/ip.e*1e3), 6)
    n_labels = [round(tick,2) for tick in n_ticks]

    ax1.set_ylabel(r"$\sigma_\sigma^\varphi$ [meV/V$^2$]", fontsize=22)
    ax1_n.set_xlabel(r'n$_{2D}$ [$10^{11}$ cm$^{-2}$]', fontsize=22)
    ax2.set_ylabel(r"$\eta_\sigma^\varphi$ [meV/K$^2$]", fontsize=22)
    ax3.set_ylabel(r"$\eta_\sigma^\varphi/\sigma_\sigma^\varphi$ [$mV^2/K^2$]", fontsize=22)
    ax3.set_xlabel(r"$\mu$ [meV]", fontsize=22)

    ax1_n.set_xticks(n_ticks, labels=n_labels)
    #ax2.set_xticks(mu_ticks, labels=mu_labels)

    ax1.plot(chem_potential/ip.e*1e3, spin_cond/1e9, 'k')

    ax2.plot(chem_potential/ip.e*1e3, spin_thermo, 'k')
    
    ax3.plot(chem_potential/ip.e*1e3, spin_thermo/spin_cond/ip.lorentz, 'k')
    ax3.set_ylim([0.3,2.5])

    b0 = E[int(Nk/2), 0]/ip.e*1e3
    b1 = E[int(Nk/2), 1]/ip.e*1e3
    b2 = E[int(Nk/2), 2]/ip.e*1e3
    b3 = E[int(Nk/2), 3]/ip.e*1e3
    b4 = E[int(Nk/2), 4]/ip.e*1e3
    b5 = E[int(Nk/2), 5]/ip.e*1e3
    b6 = E[int(Nk/2), 6]/ip.e*1e3
    b7 = E[int(Nk/2), 7]/ip.e*1e3

    m0 = np.min(E[:, 0])/ip.e*1e3
    m1 = np.min(E[:, 1])/ip.e*1e3
    m5 = np.min(E[:, 5])/ip.e*1e3
    m6 = np.min(E[:, 6])/ip.e*1e3

    ax1.axvspan(m0, b1, ymin=0, ymax=1, color='lightgray', alpha=0.3)
    ax1.axvspan(m5, b6, ymin=0, ymax=1, color='lightgray', alpha=0.3)
    ax1.axvspan(b1, b2, ymin=0, ymax=1, color='lightgray', alpha=0.8)
    ax1.axvspan(b5, b7, ymin=0, ymax=1, color='lightgray', alpha=0.8)
    #ax1.axvspan(m0, b0, ymin=0, ymax=1, color='lightgray', alpha=0.3)
    #ax1.axvspan(m1, b1, ymin=0, ymax=1, color='lightgray', alpha=0.3)
    #ax1.axvspan(b1, b2, ymin=0, ymax=1, color='lightgray', alpha=0.5)

    ax2.axvspan(m0, b1, ymin=0, ymax=1, color='lightgray', alpha=0.3)
    ax2.axvspan(m5, b6, ymin=0, ymax=1, color='lightgray', alpha=0.3)
    ax2.axvspan(b1, b2, ymin=0, ymax=1, color='lightgray', alpha=0.8)
    ax2.axvspan(b5, b7, ymin=0, ymax=1, color='lightgray', alpha=0.8)
    
    ax2.text(x=m0-0.08, y=3.125, s=r'$\Delta E_{min}^{1/2}$')
    ax2.text(x=m5-0.07, y=3.125, s=r'$\Delta E_{min}^{3/2}$')
    ax2.text(x=b1+0.04, y=3.125, s=r'$\Delta E_{SO}^{1/2}$')
    ax2.text(x=b5+0.06, y=3.125, s=r'$\Delta E_{SO}^{3/2}$')
    ax3.axvspan(m0, b1, ymin=0, ymax=1, color='lightgray', alpha=0.3)
    ax3.axvspan(m5, b6, ymin=0, ymax=1, color='lightgray', alpha=0.3)
    ax3.axvspan(b1, b2, ymin=0, ymax=1, color='lightgray', alpha=0.8)
    ax3.axvspan(b5, b7, ymin=0, ymax=1, color='lightgray', alpha=0.8)

    ax1.axvline(x=b0, color='gray', linewidth=0.5)
    ax1.axvline(x=b1, color='gray', linewidth=0.5)
    ax1.axvline(x=b2, color='gray', linewidth=0.5)
    ax1.axvline(x=b3, color='gray', linewidth=0.5)
    ax1.axvline(x=b4, color='gray', linewidth=0.5)
    ax1.axvline(x=b5, color='gray', linewidth=0.5)
    ax1.axvline(x=b6, color='gray', linewidth=0.5)
    ax1.axvline(x=m0, color='gray', linewidth=0.5)
    ax1.axvline(x=m5, color='gray', linewidth=0.5)
    
    ax2.axvline(x=b0, color='gray', linewidth=0.5)
    ax2.axvline(x=b1, color='gray', linewidth=0.5)
    ax2.axvline(x=b2, color='gray', linewidth=0.5)
    ax2.axvline(x=b3, color='gray', linewidth=0.5)
    ax2.axvline(x=b4, color='gray', linewidth=0.5)
    ax2.axvline(x=b5, color='gray', linewidth=0.5)
    ax2.axvline(x=b6, color='gray', linewidth=0.5)
    ax2.axvline(x=m0, color='gray', linewidth=0.5)
    ax2.axvline(x=m5, color='gray', linewidth=0.5)

    ax3.axvline(x=b0, color='gray', linewidth=0.5)
    ax3.axvline(x=b1, color='gray', linewidth=0.5)
    ax3.axvline(x=b2, color='gray', linewidth=0.5)
    ax3.axvline(x=b3, color='gray', linewidth=0.5)
    ax3.axvline(x=b4, color='gray', linewidth=0.5)
    ax3.axvline(x=b5, color='gray', linewidth=0.5)
    ax3.axvline(x=b6, color='gray', linewidth=0.5)
    ax3.axvline(x=m0, color='gray', linewidth=0.5)
    ax3.axvline(x=m5, color='gray', linewidth=0.5)

    #ax1.set_yticks(fontsize=20)    
    #ax3.set_xticks(fontsize=20)
    #ax3.set_yticks(fontsize=20)
    fig.align_labels()
    plt.tight_layout()
    plt.subplots_adjust(left=0.17, hspace=0.15, wspace=0.5, right=0.96, top=0.92, bottom=0.1)

    plt.show()

if __name__=="__main__":
    #first_order(system='tube')
    second_order(system='tube')