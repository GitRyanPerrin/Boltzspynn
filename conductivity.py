from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import simpson

import input_data as ip
#from hamiltonian import Hmat, quantum_numbers
from hamiltonian_scroll import Hmat, quantum_numbers
from fermi_derivative import fermi_derivative
from density import calc_ndens

def main():

    if ip.B < 1.0:
        k0 = 2*np.pi/ip.R
    elif ip.B >= 1.0 and ip.B < 3.0:
        k0 = 4*np.pi/ip.R
    else:
        k0 = 7*np.pi/ip.R
    Nk = 3500
    k = np.linspace(-k0, k0, Nk)

    with ProcessPoolExecutor() as pool:
        H_fut = pool.map(Hmat, k, repeat(ip.B))
    H = np.array([fut for fut in H_fut])
    #E = np.stack([eigh(H[ik], eigvals_only=True, lower=True) for ik in range(Nk)])
    E = np.zeros(Nk, dtype=object)
    V = np.zeros(Nk, dtype=object)
    for ik in range(Nk):
        E[ik], V[ik] = eigh(H[ik], lower=True)
    E = np.stack(E)
    V = np.stack(V)
    norm = np.stack([np.real(np.sum(V[ik]@V[ik].conj().T)) for ik in range(Nk)])

    T = 0.1
    Nmu = 130
    dmu = 4*ip.e/1e3
    band_min = np.min(E)-0.1*ip.e/1e3
    band_max = np.min(E)+dmu
    chem_potential = np.linspace(band_min, band_max, Nmu)

    v = np.gradient(H, axis=0)
    v = np.stack([V[ik].conj().T@v[ik]@V[ik]/norm[ik] for ik in range(Nk)])
    v = v/ip.hbar

    fermi = np.stack([fermi_derivative(k, E, mu, T, order=1) for mu in chem_potential])

    energy_coeff = np.stack([[(E[ik] - mu)/ip.kB/T for ik in range(Nk)] for mu in chem_potential])
    econd = np.stack([[np.diagonal(v[ik])*np.diagonal(v[ik])*fermi[imu,ik] for ik in range(Nk)] for imu in range(Nmu)])
    tcond = np.stack([[energy_coeff[imu, ik]*np.diagonal(v[ik])*np.diagonal(v[ik])*fermi[imu,ik] for ik in range(Nk)] for imu in range(Nmu)])
    econd = -np.sum([simpson(econd[imu], x=k, axis=0) for imu in range(Nmu)],axis=1)
    tcond = -np.sum([simpson(tcond[imu], x=k, axis=0) for imu in range(Nmu)],axis=1)
    #plt.plot(chem_potential, ndens/1e23)
    #plt.show()

    #coeff = ip.e**2*ip.tau_e/2/np.pi/ip.area/ip.hbar**2
    # For some reason there is a missing 1e9 in the denominator...
    ecoeff = ip.e**2*ip.tau_e/2/np.pi/ip.area
    tcoeff = ip.e*ip.tau_e/ip.area

    mu_ticks = [chem_potential[i]/ip.e*1e3 for i in np.arange(0,Nmu,10)]
    mu_labels = [round(chem_potential[i]/ip.e*1e3,1) for i in np.arange(0,Nmu,10)]

    #ndens = np.stack([calc_ndens(k, E, tick*ip.e/1e3, T)/ip.area/1e23/2/np.pi for tick in mu_ticks])
    #n_ticks = [ndens[i]/ip.area/1e23 for i in np.arange(0,Nmu,10)]
    #n_labels = [round(ndens[i]/ip.area/1e23,1) for i in np.arange(0,Nmu,10)]    
    #n_ticks = ndens
    #n_labels = np.around(ndens,1)
    ndens = lambda mu: calc_ndens(k, E, mu*ip.e/1e3, T)/ip.area/1e23/2/np.pi

    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
    ax1_2 = ax1.twiny()
    mu_min, mu_max = ax1.get_ylim()
    ax1_2.set_xlim(ndens(mu_min), ndens(mu_max))
    ax1_2.set_xlabel(r'n$_{3D}$ [$10^{17}$ 1/cm$^3$]')
    ax1.set_ylabel(r"$\sigma_e$ [S/mm]")
    ax1.set_xlabel(r"$\mu$ [meV]")
    ax2.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
    ax2.set_ylabel(r"$k_zR$")
    ax2.set_xlabel(r"E($k_z$) [meV]")
    ax2.set_ylim([0, k0*ip.R])
    ax1.set_xticks(mu_ticks, labels=mu_labels)
    ax2.set_xticks(mu_ticks, labels=mu_labels)
    #ax1_2.set_xticks(n_ticks, labels=n_labels)
    ax1.plot(chem_potential/ip.e*1e3, tcond*tcoeff/econd/ecoeff/1e9, 'k')
    ax2.plot(E[int(Nk/2):]/ip.e*1e3, k[int(Nk/2):]*ip.R, 'k')
    plt.show()

if __name__=="__main__":
    main()
