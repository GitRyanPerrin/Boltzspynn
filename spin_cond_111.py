from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
from scipy.linalg import eigh
from scipy.integrate import simps
import matplotlib.pyplot as plt

import input_data as ip
from hamiltonian import Hmat, quantum_numbers
from fermi_derivative import fermi_derivative
from density import calc_ndens

def vzr(k, B=ip.B, Nm=10, alphaR=ip.alphaR, betaD=ip.betaD, R=ip.R):

    hbar = ip.hbar
    meff = ip.meff

    qn = quantum_numbers(Nm)
    Nstat = 2*(2*Nm+1)

    v = np.zeros([Nstat, Nstat], dtype=complex)

    for a1 in range(Nstat):
        m1 = int(qn[a1, 0])
        s1 = int(qn[a1, 1])

        v[a1, a1] += 0

        for a2 in range(Nstat):
            m2 = int(qn[a2, 0])
            s2 = int(qn[a2, 1])

            if m1 == m2+1 and s1 == -s2:
                v[a1, a2] += hbar*k/meff

            if abs(m1-m2) == 3 and s1 == s2:
                v[a1, a2] += betaD/np.sqrt(6)/hbar/2

            v[a2,a1] += np.conjugate(v[a1,a2])
    
    return v

def vzp(k, B=ip.B, Nm=10, alphaR=ip.alphaR, betaD=ip.betaD, R=ip.R):

    hbar = ip.hbar
    meff = ip.meff

    qn = quantum_numbers(Nm)
    Nstat = 2*(2*Nm+1)

    v = np.zeros([Nstat, Nstat], dtype=complex)

    for a1 in range(Nstat):
        m1 = int(qn[a1, 0])
        s1 = int(qn[a1, 1])

        v[a1, a1] += alphaR/hbar
        #v[a1, a1] += alphaR/2

        for a2 in range(Nstat):
            m2 = int(qn[a2, 0])
            s2 = int(qn[a2, 1])

            if m1 == m2+1 and s1 == -s2:
                #v[a1, a2] += 1.0j*hbar*k/meff
                v[a1, a2] += -1.0j*k/meff
                #v[a1, a2] += k/meff

            if abs(m1-m2) == 3 and s1 == s2:
                v[a1, a2] -= s1*1.0j*betaD/np.sqrt(6)/hbar/2

            v[a2,a1] += np.conjugate(v[a1,a2])
    
    return v

def vzz(k, B=ip.B, Nm=10, alphaR=ip.alphaR, betaD=ip.betaD, R=ip.R):

    hbar = ip.hbar
    meff = ip.meff

    qn = quantum_numbers(Nm)
    Nstat = 2*(2*Nm+1)

    v = np.zeros([Nstat, Nstat], dtype=complex)

    for a1 in range(Nstat):
        m1 = int(qn[a1, 0])
        s1 = int(qn[a1, 1])

        v[a1, a1] += s1*hbar*k/meff
    
    return v

if __name__=="__main__":
    
    if ip.B < 1.0:
        k0 = 2*np.pi/ip.R
    elif ip.B >= 1.0 and ip.B < 3.0:
        k0 = 4*np.pi/ip.R
    else:
        k0 = 7*np.pi/ip.R
    Nk = 3500
    k = np.linspace(-k0, k0, Nk)

    with ProcessPoolExecutor() as pool:
        v_fut = pool.map(vzp, k)
        H_fut = pool.map(Hmat, k, repeat(ip.B))
    vs = np.stack([fut for fut in v_fut])
    H = np.stack([fut for fut in H_fut])
    
    E = np.zeros(Nk, dtype=object)
    V = np.zeros(Nk, dtype=object)
    for ik in range(Nk):
        E[ik], V[ik] = eigh(H[ik])
    E = np.stack(E)
    V = np.stack(V)
    norm = np.stack([np.real(np.sum(V[ik]@V[ik].conj().T)) for ik in range(Nk)])

    v = np.gradient(H, axis=0)/ip.hbar
    v = np.stack([V[ik].conj().T@v[ik]@V[ik]/norm[ik] for ik in range(Nk)])
    v = np.stack([np.diagonal(v[ik]) for ik in range(Nk)])
    vs = np.stack([V[ik].conj().T@vs[ik]@V[ik]/norm[ik] for ik in range(Nk)])
    vs = np.stack([np.diagonal(vs[ik])*ip.hbar/2 for ik in range(Nk)])


    T = 0.2
    Nmu = 150
    dmu = 2.5*ip.e/1e3
    band_min = np.min(E)-0.2*ip.e/1e3
    band_max = np.min(E)+dmu
    chem_potential = np.linspace(band_min, band_max, Nmu)

    fermi = np.stack([fermi_derivative(k, E, mu, T, order=2) for mu in chem_potential])

    cond = np.stack([[v[ik]*v[ik]*vs[ik]*fermi[imu,ik] for ik in range(Nk)] for imu in range(Nmu)])
    cond = np.sum([simps(cond[imu], k, axis=0) for imu in range(Nmu)],axis=1)
    #print(cond.shape)

    #coeff = ip.e**2*ip.tau_e/2/np.pi/ip.area/ip.hbar**2
    # For some reason there is a missing 1e9 in the denominator...
    coeff = ip.e**2*ip.tau_s**2/2/np.pi/ip.area*ip.hbar/4/ip.e/1e3

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
    ax1.set_ylabel(r"$\sigma_\sigma$ [meV/m$^2$]")
    ax1.set_xlabel(r"$\mu$ [meV]")
    ax2.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
    ax2.set_ylabel(r"$k_zR$")
    ax2.set_xlabel(r"E($k_z$) [meV]")
    ax2.set_ylim([0, k0*ip.R])
    ax1.set_xticks(mu_ticks, labels=mu_labels)
    ax2.set_xticks(mu_ticks, labels=mu_labels)
    ax1.plot(chem_potential/ip.e*1e3, cond*coeff/1e9, 'k')
    ax2.plot(E[int(Nk/2):]/ip.e*1e3, k[int(Nk/2):]*ip.R, 'k')
    plt.show()