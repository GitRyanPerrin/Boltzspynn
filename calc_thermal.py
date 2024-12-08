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
from spin_cond_111 import vzp as vzp_tube
from spin_cond_scroll import vzr as vzr_scroll
from spin_cond_scroll import vzp as vzp_scroll
from spin_cond_scroll import vzz as vzz_scroll

def calc_band_structure(k, system='tube'):

    Nk = len(k)

    with ProcessPoolExecutor(max_workers=10) as pool:
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

    #np.savez_compressed(f"./hamiltonians/{system}_{ip.material}_50_{ip.alphaR/ip.e*1e12}_{ip.betaD/ip.e*1e12}_{ip.B}_{ip.theta/np.pi}.npz", k, E, V, v, norm)

    return E, V, v, norm

def calc_charge_conductivity(k, v, E, V, chem_potential, T=0.1, system='tube'):

    Nk = len(k)
    Nmu = len(chem_potential)

    fermi = np.stack([fermi_derivative(k, E, mu, T, order=1) for mu in chem_potential])

    cond = np.stack([[v[ik]*v[ik]*fermi[imu,ik] for ik in range(Nk)] for imu in range(Nmu)])
    cond = -np.sum([simpson(cond[imu], x=k, axis=0) for imu in range(Nmu)],axis=1)

    coeff = ip.e**2*ip.tau_e/2/np.pi/ip.area
    cond = cond*coeff

    #np.savez_compressed(f"./conductivities/elec_charge_{system}_{ip.material}_50_{ip.alphaR/ip.e*1e12}_{ip.betaD/ip.e*1e12}_{ip.B}_{ip.theta/np.pi}.npz", chem_potential, cond)

    return cond

def calc_charge_thermo(k, v, E, V, chem_potential, T=0.1, system='tube'):

    Nk = len(k)
    Nmu = len(chem_potential)
    kBT = ip.kB*T

    fermi = np.stack([fermi_derivative(k, E, mu, T, order=1) for mu in chem_potential])

    energy_coeff = np.stack([[(E[ik]-mu)/T for ik in range(Nk)] for mu in chem_potential])

    cond = np.stack([[energy_coeff[imu, ik]*v[ik]*v[ik]*fermi[imu,ik] for ik in range(Nk)] for imu in range(Nmu)])
    cond = -np.sum([simpson(cond[imu], x=k, axis=0) for imu in range(Nmu)],axis=1)

    coeff = ip.e*ip.tau_e/2/np.pi/ip.area
    cond = cond*coeff

    #np.savez_compressed(f"./conductivities/thermal_charge_{system}_{ip.material}_50_{ip.alphaR/ip.e*1e12}_{ip.betaD/ip.e*1e12}_{ip.B}_{ip.theta/np.pi}.npz", chem_potential, cond)

    return cond

def calc_ballistic_thermo(k, v, E, V, chem_potential, T=0.1, system='tube'):

    Nk = len(k)
    Nmu = len(chem_potential)
    kBT = ip.kB*T

    vp = np.where(v > 0, v, 0)
    vn = np.where(v < 0, v, 0)
    
    fermi = np.stack([fermi_derivative(k, E, mu, T, order=1) for mu in chem_potential])

    energy_coeff = np.stack([[(E[ik]-mu)/T for ik in range(Nk)] for mu in chem_potential])

    cond = np.stack([[energy_coeff[imu, ik]*(vp[ik]-vn[ik])*fermi[imu,ik] for ik in range(Nk)] for imu in range(Nmu)])
    cond = -np.sum([simpson(cond[imu], x=k, axis=0) for imu in range(Nmu)],axis=1)

    coeff = ip.e*ip.tau_e/2/np.pi/ip.area
    cond = cond*coeff

    #np.savez_compressed(f"./conductivities/ballistic_thermal_charge_{system}_{ip.material}_50_{ip.alphaR/ip.e*1e12}_{ip.betaD/ip.e*1e12}_{ip.B}_{ip.theta/np.pi}.npz", chem_potential, cond)

    return cond

def calc_spin_thermo(k, v, E, V, norm, chem_potential, T=0.1, system='tube', polarization='p'):

    Nk = len(k)
    Nmu = len(chem_potential)

    with ProcessPoolExecutor(max_workers=10) as pool:
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
    
    kBT = ip.kB*T
    #energy_coeff1 = np.stack([[2*(E[ik]-mu)/kBT**2 for ik in range(Nk)] for mu in chem_potential])
    energy_coeff1 = np.stack([[2*(E[ik]-mu)/T**2 for ik in range(Nk)] for mu in chem_potential])
    fermi1 = np.stack([fermi_derivative(k, E, mu, T, order=1) for mu in chem_potential])
    #energy_coeff2 = np.stack([[(E[ik]-mu)**2/kBT**2 for ik in range(Nk)] for mu in chem_potential])
    energy_coeff2 = np.stack([[(E[ik]-mu)**2/T**2 for ik in range(Nk)] for mu in chem_potential])
    fermi2 = np.stack([fermi_derivative(k, E, mu, T, order=2) for mu in chem_potential])

    energy_coeff = np.stack([[energy_coeff1[imu,ik]*fermi1[imu,ik] + energy_coeff2[imu,ik]*fermi2[imu,ik] for ik in range(Nk)] for imu in range(Nmu)])

    cond = np.stack([[energy_coeff[imu,ik]*v[ik]*v[ik]*vs[ik] for ik in range(Nk)] for imu in range(Nmu)])
    cond = np.sum([simpson(cond[imu], x=k, axis=0) for imu in range(Nmu)],axis=1)

    coeff = ip.tau_s**2/2/np.pi/ip.area*ip.hbar/4/ip.e/1e3
    #coeff = 1
    cond = cond*coeff

    #np.savez_compressed(f"./conductivities/thermal_spin_{system}_{polarization}_{ip.material}_{ip.R/1e-9}_{ip.alphaR/ip.e*1e12}_{ip.betaD/ip.e*1e12}_{ip.B}_{ip.theta/np.pi}.npz", chem_potential, cond)

    return cond

def thermal_conds(system='tube'):

    if ip.B < 1.0:
        k0 = 2*np.pi/ip.R
    elif ip.B >= 1.0 and ip.B < 3.0:
        k0 = 4*np.pi/ip.R
    else:
        k0 = 7*np.pi/ip.R
    Nk = 3500
    k = np.linspace(-k0, k0, Nk)
    E, V, v, norm = calc_band_structure(k, system=system)

    T = 0.1
    Nmu = 300
    dmu = 1.0*ip.e/1e3
    band_min = np.min(E)-0.1*ip.e/1e3
    band_max = np.min(E)+dmu
    chem_potential = np.linspace(band_min, band_max, Nmu)

    charge_thermo = calc_charge_thermo(k, v, E, V, chem_potential, T, system=system)
    #charge_cond = calc_charge_conductivity(k, v, E, V, chem_potential, T=T, system=system)
    spin_thermo_p = calc_spin_thermo(k, v, E, V, norm, chem_potential, T=T, system=system, polarization='p')
    spin_thermo_z = calc_spin_thermo(k, v, E, V, norm, chem_potential, T=T, system=system, polarization='z')
    ndens = lambda mu: calc_ndens(k, E, mu*ip.e/1e3, T=T)/ip.area/1e23/2/np.pi

    #fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize=(6,6),dpi=120)
    plt.rcParams.update({'font.size': 18})
    fig, (ax1, ax2, ax4) = plt.subplots(3,1,figsize=(3.375,2.5),dpi=300)
    ax1_2 = ax1.twiny()
    #plt.tight_layout()
    plt.autoscale()
    
    mu_ticks = [chem_potential[i]/ip.e*1e3 for i in np.arange(0,Nmu,5)]
    mu_labels = [round(chem_potential[i]/ip.e*1e3,1) for i in np.arange(0,Nmu,5)]

    ax1.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
    ax1_2.set_xlim(ndens(band_min/ip.e*1e3), ndens(band_max/ip.e*1e3))
    ax1.set_ylim([-30,40])
    ax2.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
    #ax3.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
    ax4.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
    ax4.set_ylim([0, 1.5])

    ax1.set_ylabel(r"$S_e$ [$\mu$V/K]", fontsize=22)
    ax1_2.set_xlabel(r'n$_{3D}$ [$10^{17}$ 1/cm$^3$]', fontsize=22)
    ax2.set_ylabel(r"$\eta_\sigma^\varphi$ [mJ/(mK)$^2$]", fontsize=22)    
    #ax3.set_ylabel(r"$\eta_\sigma^z$ [mJ/(mK)$^2$]")
    #ax3.set_xlabel(r"$\mu$ [meV]")
    ax4.set_ylabel(r"$k_zR$", fontsize=22)
    ax4.set_xlabel(r"E($k_z$) [meV]", fontsize=22)
    
    ax1.set_xticks(mu_ticks, labels=mu_labels)
    ax2.set_xticks(mu_ticks, labels=mu_labels)
    #ax3.set_xticks(mu_ticks, labels=mu_labels)
    ax4.set_xticks(mu_ticks, labels=mu_labels)

    plt.subplots_adjust(left=0.2, hspace=0.15)

    ax1.plot(chem_potential/ip.e*1e3, charge_thermo/charge_cond*1e6, 'k')
    ax2.plot(chem_potential/ip.e*1e3, spin_thermo_p, 'k')
    #ax3.plot(chem_potential/ip.e*1e3, spin_thermo_z/1e12, 'k')
    ax4.plot(E[int(Nk/2):]/ip.e*1e3, k[int(Nk/2):]*ip.R, 'k')

    plt.show()

if __name__=="__main__":
    
    thermal_conds(system='scroll')
