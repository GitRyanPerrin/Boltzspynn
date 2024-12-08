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
from spin_cond_scroll import vzr as vzr_scroll
from spin_cond_scroll import vzp as vzp_scroll
from spin_cond_scroll import vzz as vzz_scroll
from calc_conductivities import calc_ballistic_conductivity, calc_band_structure, calc_charge_conductivity, calc_spin_conductivity
from calc_thermal import calc_charge_thermo, calc_ballistic_thermo

def svA():

    if ip.B < 1.0:
        k0 = 2*np.pi/ip.R
    elif ip.B >= 1.0 and ip.B < 3.0:
        k0 = 4*np.pi/ip.R
    else:
        k0 = 7*np.pi/ip.R
    Nk = 2000
    k = np.linspace(-k0, k0, Nk)
    system='tube'

    NA = 2
    E = np.zeros(NA, dtype=object)
    V = np.zeros(NA, dtype=object)
    v = np.zeros(NA, dtype=object)
    norm = np.zeros(NA, dtype=object)
    ballistic_cond = np.zeros(NA, dtype=object)
    spin_cond = np.zeros(NA, dtype=object)
    A = np.linspace(ip.alphaR/2, ip.alphaR, NA)
    for ia, a in enumerate(A):
        E[ia], V[ia], v[ia], norm[ia] = calc_band_structure(k, system=system, B=0.0, alphaR=a)

    T = 0.1
    Nmu = 150
    dmu = 2*ip.e/1e3
    band_min = np.min(E[0])-0.5*ip.e/1e3
    band_max = np.min(E[0])+dmu
    chem_potential = np.linspace(band_min, band_max, Nmu)

    #for ib in range(NB):
    #    ballistic_cond[ib] = calc_ballistic_conductivity(k, v[ib], E[ib], V[ib], chem_potential, T=T, system=system)
    #ballistic_cond = np.stack(ballistic_cond)

    with ProcessPoolExecutor() as pool:
        cond_fut = pool.map(calc_ballistic_conductivity, repeat(k), v, E, V, repeat(chem_potential), repeat(T), repeat(system))
    ballistic_cond = [fut for fut in cond_fut]
    ballistic_cond = np.stack(ballistic_cond)

    with ProcessPoolExecutor() as pool:
        cond_fut = pool.map(calc_charge_conductivity, repeat(k), v, E, V, repeat(chem_potential), repeat(T), repeat(system))
    diff_cond = [fut for fut in cond_fut]
    diff_cond = np.stack(diff_cond)

    with ProcessPoolExecutor() as pool:
        cond_fut = pool.map(calc_charge_thermo, repeat(k), v, E, V, repeat(chem_potential), repeat(T), repeat(system))
    seeb = [fut for fut in cond_fut]
    seeb = np.stack(seeb)

    with ProcessPoolExecutor() as pool:
        cond_fut = pool.map(calc_charge_thermo, repeat(k), v, E, V, repeat(chem_potential), repeat(T), repeat(system))
    ball_seeb = [fut for fut in cond_fut]
    ball_seeb = np.stack(ball_seeb)


    ndens = lambda mu: calc_ndens(k, E[0], mu*ip.e/1e3, T=T)/ip.area/1e23/2/np.pi

    #fig, ax1 = plt.subplots()
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, figsize=(6,6))
    ax1_n = ax1.twiny()
    ax1_n.set_xlim(ndens(band_min/ip.e*1e3), ndens(band_max/ip.e*1e3))

    ax1.set_ylabel(r"$\sigma_e^{(1)}$ [S/mm]")
    ax1_n.set_xlabel(r'n$_{3D}$ [$10^{17}$ 1/cm$^3$]')
    ax2.set_ylabel(r"$G_e^{(1)}$ [$2e/h$]")
    ax3.set_ylabel(r"$S^{(1)}$ [$\mu$V/K]")
    ax3.set_xlabel(r"$\mu$ [meV]")

    ax3.set_ylim([-30, 30])
    #ax4.set_ylim([-30, 30])

    l = ['solid', 'dotted']
    label = [r'$\alpha$ = 20 meV$\cdot$nm', r'$\alpha$ = 40 meV$\cdot$nm']
    [ax1.plot(chem_potential/ip.e*1e3, diff_cond[ib]/1e9, 'k', linestyle=l[ib]) for ib in range(NA)]
    [ax2.plot(chem_potential/ip.e*1e3, ballistic_cond[ib]/295.4605793707242*2, 'k', linestyle=l[ib]) for ib in range(NA)]
    [ax3.plot(chem_potential/ip.e*1e3, seeb[ib]/diff_cond[ib]*1e6, 'k', linestyle=l[ib]) for ib in range(NA)]
    ax1.legend()
    plt.show()

def spin_currents(system, polarization, betaD=ip.betaD):

    if ip.B < 1.0:
        k0 = 2*np.pi/ip.R
    elif ip.B >= 1.0 and ip.B < 3.0:
        k0 = 4*np.pi/ip.R
    else:
        k0 = 7*np.pi/ip.R
    Nk = 2000
    k = np.linspace(-k0, k0, Nk)

    NA = 20
    E = np.zeros(NA, dtype=object)
    V = np.zeros(NA, dtype=object)
    v = np.zeros(NA, dtype=object)
    norm = np.zeros(NA, dtype=object)
    ballistic_cond = np.zeros(NA, dtype=object)
    spin_cond = np.zeros(NA, dtype=object)
    #A = np.linspace(0e-12*ip.e, 40e-12*ip.e, NA)
    A = np.linspace(0e-12*ip.e, 60e-12*ip.e, NA)
    #A = np.linspace(9e-12*ip.e, 11e-12*ip.e, NA)
    T = 0.1
    #mu = 2*ip.e/1e3 
    #mu = calc_fermi(T, B=ip.B, theta=ip.theta)*ip.e/1e3
    vs = np.zeros(NA, dtype=object)
    fermi = np.zeros(NA, dtype=object)
    cond = np.zeros(NA, dtype=object)
    therm = np.zeros(NA, dtype=object)
    for ia, a in enumerate(A):
        E[ia], V[ia], v[ia], norm[ia] = calc_band_structure(k, system=system, B=0.0, alphaR=a, betaD=betaD)
        mu = np.min(E[ia][:,5])
        #mu = -0.0*ip.e/1e3

        with ProcessPoolExecutor() as pool:
            if system == 'tube' and polarization == 'r':
                v_fut = pool.map(vzr_tube, k)
            if system == 'tube' and polarization == 'p':
                v_fut = pool.map(vzp_tube,
                    k,
                    repeat(ip.B),
                    repeat(10),
                    repeat(a),
                    repeat(betaD)
                )
            if system == 'scroll' and polarization == 'r':
                v_fut = pool.map(vzr_scroll, k)
            if system == 'scroll' and polarization == 'p':
                v_fut = pool.map(vzp_scroll, k)
            if system == 'scroll' and polarization == 'z':
                v_fut = pool.map(vzz_scroll, k)
        
        vs[ia] = np.stack([fut for fut in v_fut])

        vs[ia] = np.stack([V[ia][ik].conj().T@vs[ia][ik]@V[ia][ik]/norm[ia][ik] for ik in range(Nk)])
        vs[ia] = np.stack([np.diagonal(vs[ia][ik])*ip.hbar/2 for ik in range(Nk)])
        
        #fermi[ia] = fermi_derivative(k, E[ia], mu, T, order=2)
        fermi[ia] = fermi_derivative(k, E[ia], mu, T, order=2)
        cond[ia] = np.stack([v[ia][ik]*v[ia][ik]*vs[ia][ik]*fermi[ia][ik] for ik in range(Nk)])
        cond[ia] = np.sum(simpson(cond[ia], x=k, axis=0),axis=0)
        '''
        energy_coeff = np.stack([(E[ia][ik]-mu)/T for ik in range(Nk)])
        
        fermi[ia] = fermi_derivative(k, E[ia], mu, T, order=1)

        cond[ia] = np.stack([v[ia][ik]*v[ia][ik]*fermi[ia][ik] for ik in range(Nk)])
        cond[ia] = -np.sum(simpson(cond[ia], x=k, axis=0), axis=0)

        therm[ia] = np.stack([energy_coeff[ik]*v[ia][ik]*v[ia][ik]*fermi[ia][ik] for ik in range(Nk)])
        therm[ia] = -np.sum(simpson(therm[ia], x=k, axis=0), axis=0)
        '''
    #coeff = ip.e**2*ip.tau_e/2/np.pi/ip.area
    #cond = cond*coeff 

    #t_coeff = ip.e*ip.tau_e/2/np.pi/ip.area       
    #therm = therm*t_coeff

    coeff = ip.e**2*ip.tau_s**2/2/np.pi/ip.area*ip.hbar/4/ip.e/1e3
    s_cond = cond*coeff

    fig, ax1 = plt.subplots()

    #ax1.set_ylabel(r"$\sigma_\sigma^{(2)}$ [meV/m$^2$]")
    ax1.set_ylabel(r"$\sigma_z^{(2)\varphi}$ [meV/V$^2$]", fontsize=16)
    ax1.set_xlabel(r"$\alpha$ [meV$\cdot$nm]", fontsize=16)
    ax1.text(x=32, y=19, s=r'$\alpha_c^{1/2}$')
    ax1.axvline(x=34, color='gray', linestyle='dotted')
    ax1.axvline(x=43, color='gray', linestyle='dotted')
    ax1.text(x=41, y=19, s=r'$\alpha_c^{3/2}$')
    #ax1.text(x=45.6, y=19, s=r'$\alpha_c^{1/2}$')
    ax1.axvline(x=45, color='blue', linestyle='dotted')
    ax1.axvline(x=55, color='blue', linestyle='dotted')
    #ax1.text(x=55.5, y=19, s=r'$\alpha_c^{3/2}$')
    #plt.arrow(34,0.5,10,0,width=0.01, color='k')
    #plt.arrow(43,0.7,11,0,width=0.01, color='k')
    #ax1.text(x=20, y=11.5, s=r'$\alpha_c^{1/2}$')
    #ax1.axvline(x=20.5, color='gray', linestyle='dotted')
    #ax1.axvline(x=26.3, color='gray', linestyle='dotted')
    #ax1.text(x=26, y=11.5, s=r'$\alpha_c^{3/2}$')
    
    #ax1.text(x=28, y=1.35, s=r'$\alpha_c^{7/2}$')
    #ax1.axvline(x=29, color='gray', linestyle='dotted')
    #ax1.text(x=33, y=1.35, s=r'$\Delta E_{SO,min}$')
    #ax1.axvline(x=33, color='gray', linestyle='dotted')
    #ax1.plot(A/ip.e/1e-12, therm/cond*1e6, 'k')
    #ax1.plot(A/ip.e/1e-12, cond/1e9, 'k')
    ax1.plot(A/ip.e/1e-12, s_cond/1e9, 'k')
    #ax1.plot(A/ip.e/1e-12, 2*np.gradient(s_cond)/1e9, 'b', linestyle='dashed')
    #ax1.axvline(x=ip.betaD*np.sqrt(6)/ip.e/1e-12, color='gray', linestyle='dotted')
    #ax1.axvline(x=ip.betaD*np.sqrt(12)/ip.e/1e-12, color='gray', linestyle='dotted')
    plt.show()

    return A/ip.e/1e-12, s_cond/1e9

if __name__=="__main__":

    betaD = ip.betaD
    A, s_cond = spin_currents('tube', 'p', betaD=ip.betaD)
    #betaD = betaD/ip.e*1e12
    np.savez_compressed(f'./InAs_alpha_conds_E4.npz', A, s_cond)

    #spin_currents(system='tube',polarization='p')
    #spin_currents(system='tube',polarization='p', betaD=0)
    #plt.show()
    '''
    A, cond1 = spin_currents(system='tube', polarization='p', betaD=ip.betaD)
    A, cond2 = spin_currents(system='tube', polarization='p', betaD=0)

    fig, ax1 = plt.subplots()
    ax1.set_ylabel(r"$\sigma_z^{(2)\varphi}$ [meV/V$^2$]", fontsize=16)
    ax1.set_xlabel(r"$\alpha$ [meV$\cdot$nm]", fontsize=16)

    ax1.text(x=32, y=19, s=r'$\alpha_c^{1/2}$')
    ax1.axvline(x=34, color='gray', linestyle='dotted')
    ax1.axvline(x=43, color='gray', linestyle='dotted')
    ax1.text(x=41, y=19, s=r'$\alpha_c^{3/2}$')
    #ax1.text(x=45.6, y=19, s=r'$\alpha_c^{1/2}$')
    ax1.axvline(x=45, color='blue', linestyle='dotted')
    ax1.axvline(x=55, color='blue', linestyle='dotted')
    #ax1.text(x=55.5, y=19, s=r'$\alpha_c^{3/2}$')
    plt.arrow(34,5,10,0,width=0.2, color='k')
    plt.arrow(43,7,11,0,width=0.2, color='k')
    
    ax1.plot(A, cond1, 'k')
    ax1.plot(A, cond2, 'gray', linestyle='dashed')

    plt.show()
    '''