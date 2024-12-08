from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

import input_data as ip
from hamiltonian import Hmat as Hmat
from hamiltonian_scroll import Hmat as Hmat_scroll
#from hamiltonian_scroll import Hmat as Hmat
from density import calc_DOS, calc_fermi

def k_plot(Nm=5, B=ip.B, alphaR=ip.alphaR, betaD=ip.betaD, R=ip.R, theta=ip.theta, k0=np.pi/ip.R, Emax=20):
    Nk = 300
    k = np.linspace(-k0, k0, Nk)

    with ProcessPoolExecutor() as pool:
        H_fut = pool.map(Hmat,
            k,
            repeat(B),
            repeat(theta),
            repeat(Nm),
            repeat(alphaR),
            repeat(betaD),
            repeat(R)
        )

    '''
    with ProcessPoolExecutor() as pool:
        H_fut = pool.map(Hmat,
            k,
            repeat(B),
            repeat(Nm),
            repeat(alphaR),
            repeat(betaD),
            repeat(R)
        )
    '''
    H = np.array([fut for fut in H_fut])
    E = np.zeros(Nk, dtype=object)
    for ik in range(Nk):
        E[ik] = eigh(H[ik], eigvals_only=True)
    E = np.stack(E)

    T = 0.1
    #EF = calc_fermi(T, B, theta)
    #energy, dos = calc_DOS(T, E[int(Nk/2), 20]/ip.e*1e3)
    #energy, dos = calc_DOS(T, Emax, B, theta)

    fig, ax1 = plt.subplots()
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #ax1.set_ylim([-0.25, E[int(Nk/2), 20]/ip.e*1e3])
    ax1.set_ylim([-0.5, Emax])
    ax1.set_ylabel('E [meV]')
    ax1.set_xlabel('kR')
    ax1.margins(0,tight=True)
    #ax2.set_ylim([-0.25, E[int(Nk/2), 20]/ip.e*1e3])
    #ax2.set_ylim([-0.5, Emax])
    #ax2.set_xlabel(r'DOS [meV$^{-1}\cdot$cm$^{-2}$]')
    ax1.plot(k*ip.R, E[:,:20]/ip.e*1e3, 'k')
    ax1.set_title(rf"InAs: $\alpha$ = {round(alphaR*1e12/ip.e)} meV$\cdot$nm, $\beta$ = {round(betaD*1e12/ip.e)} meV$\cdot$nm")
    #ax2.plot(dos, energy, 'k')
    #ax1.axhline(y=EF, color='gray', linestyle='dotted')
    #ax2.axhline(y=EF, color='gray', linestyle='dotted')
    plt.show()

def mag_plot(Nm=5, k=0.0, alphaR=ip.alphaR, betaD=ip.betaD, R=ip.R, B0=1.0):
    NB = 100
    Nb = 60
    B = np.linspace(0, B0, NB)
    Bb = np.linspace(0, B0, Nb)

    with ProcessPoolExecutor() as pool:
        H_fut = pool.map(
            Hmat,
            repeat(k),
            B,
            repeat(ip.theta),
            repeat(Nm),
            repeat(alphaR),
            repeat(betaD),
            repeat(R)
        )

        fermi_fut = pool.map(
            calc_fermi,
            repeat(1.0),
            Bb,
            repeat(2*np.pi/R)
        )

    H = np.array([fut for fut in H_fut])
    EF = np.array([fut for fut in fermi_fut])
    E = np.zeros(NB, dtype=object)
    for iB in range(NB):
        E[iB] = eigh(H[iB], eigvals_only=True)
    E = np.stack(E)

    plt.ylabel('E [meV]')
    plt.xlabel('B [T]')
    plt.plot(B, E[:,:20]/ip.e*1e3, 'k')
    plt.plot(Bb, EF[:,0], 'gray', linestyle='dotted')
    plt.show()

def angle_plot(Nm=5, k=0.0, alphaR=ip.alphaR, betaD=ip.betaD, R=ip.R, B0=1.0):
    #NB = 100
    #Nb = 60
    #B = np.linspace(0, B0, NB)
    #Bb = np.linspace(0, B0, Nb)
    NT = 200
    theta = np.linspace(0, 2*np.pi, NT)

    with ProcessPoolExecutor() as pool:
        H_fut = pool.map(
            Hmat,
            repeat(k),
            repeat(B0),
            theta,
            repeat(Nm),
            repeat(alphaR),
            repeat(betaD),
            repeat(R)
        )

        '''
        fermi_fut = pool.map(
            calc_fermi,
            repeat(1.0),
            Bb,
            repeat(2*np.pi/R)
        )
        '''

    H = np.array([fut for fut in H_fut])
    #EF = np.array([fut for fut in fermi_fut])
    E = np.zeros(NT, dtype=object)
    for iT in range(NT):
        E[iT] = eigh(H[iT], eigvals_only=True)
    E = np.stack(E)

    plt.ylabel('E [meV]')
    plt.xlabel(r'$\theta$ [rad]')
    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', '$\pi$/2', '$\pi$', '3$\pi$/2', '2$\pi$'])
    plt.plot(theta, E[:,:20]/ip.e*1e3, 'k')
    #plt.plot(Bb, EF[:,0], 'gray', linestyle='dotted')
    plt.show()

def radius_plot(Nm=5, k=0.0, B=ip.B, alphaR=ip.alphaR, betaD=ip.betaD, Ri=2e-9, Rf=100e-9):
    NR = 120
    R = np.linspace(Ri, Rf, NR)

    with ProcessPoolExecutor() as pool:
        H_fut = pool.map(
            Hmat,
            repeat(k),
            repeat(B),
            repeat(ip.theta),
            repeat(Nm),
            repeat(alphaR),
            repeat(betaD),
            R
        )

    H = np.array([fut for fut in H_fut])
    E = np.zeros(NR, dtype=object)
    for iR in range(NR):
        E[iR] = eigh(H[iR], eigvals_only=True)
    E = np.stack(E)

    plt.ylabel('E [meV]')
    plt.xlabel('R [nm]')
    plt.plot(R*1e9, E[:,:20]/ip.e*1e3, 'k')
    plt.show()

def alpha_plot(Nm=5, k=0.0, B=ip.B, alphaR=ip.alphaR, betaD=ip.betaD, R=ip.R, Ai=0.0, Af=100e-12*ip.e):
    NA = 120
    A = np.linspace(Ai, Af, NA)

    with ProcessPoolExecutor() as pool:
        H_fut = pool.map(
            Hmat,
            repeat(k),
            repeat(B),
            repeat(ip.theta),
            repeat(Nm),
            A,
            repeat(betaD),
            repeat(R)
        )

    H = np.array([fut for fut in H_fut])
    E = np.zeros(NA, dtype=object)
    for iA in range(NA):
        E[iA] = eigh(H[iA], eigvals_only=True)
    E = np.stack(E)

    plt.ylabel('E [meV]')
    plt.xlabel(r'$\alpha$ [meV$\cdot$nm]')
    plt.plot(A/ip.e*1e12, E[:,:20]/ip.e*1e3, 'k')
    #plt.plot(A/ip.R/ip.E0, E[:,:20]/ip.e*1e3, 'k')
    plt.show()

def beta_plot(Nm=5, k=0.0, B=ip.B, alphaR=ip.alphaR, betaD=ip.betaD, R=ip.R, Bi=0.0, Bf=30e-12*ip.e):
    NB = 120
    Beta = np.linspace(Bi, Bf, NB)

    with ProcessPoolExecutor() as pool:
        H_fut = pool.map(
            Hmat,
            repeat(k),
            repeat(B),
            repeat(Nm),
            repeat(alphaR),
            Beta
        )

    H = np.array([fut for fut in H_fut])
    E = np.zeros(NB, dtype=object)
    for iB in range(NB):
        E[iB] = eigh(H[iB], eigvals_only=True)
    E = np.stack(E)

    plt.ylabel('E [meV]')
    plt.xlabel(r'$\beta$ [meV$\cdot$nm]')
    plt.plot(Beta/ip.e*1e12, E[:,:20]/ip.e*1e3, 'k')
    plt.show()

def energy_compare(Nm=5, B=ip.B, alphaR=ip.alphaR, betaD=ip.betaD, R=ip.R, theta=ip.theta, k0=np.pi/ip.R, Emax=20):

    Nk = 120
    k = np.linspace(-k0, k0, Nk)

    '''
    with ProcessPoolExecutor() as pool:
        H_fut_base = pool.map(Hmat,
            k,
            repeat(B),
            repeat(theta),
            repeat(Nm),
            repeat(alphaR),
            repeat(0),
            repeat(R)
        )
    '''

    with ProcessPoolExecutor() as pool:
        H_fut_tube = pool.map(Hmat,
            k,
            repeat(B),
            repeat(theta),
            repeat(Nm),
            repeat(alphaR),
            repeat(betaD),
            repeat(R)
        )
    
        H_fut_scroll = pool.map(Hmat_scroll,
            k,
            repeat(B),
            repeat(theta),
            repeat(Nm),
            repeat(alphaR),
            repeat(betaD),
            repeat(R)
        )

    #H_base = np.array([fut for fut in H_fut_base])
    H_tube = np.array([fut for fut in H_fut_tube])
    H_scroll = np.array([fut for fut in H_fut_scroll])
    #E_base = np.zeros(Nk, dtype=object)
    E_tube = np.zeros(Nk, dtype=object)
    E_scroll = np.zeros(Nk, dtype=object)
    for ik in range(Nk):
        #E_base[ik] = eigh(H_base[ik], eigvals_only=True)
        E_tube[ik] = eigh(H_tube[ik], eigvals_only=True)
        E_scroll[ik] = eigh(H_scroll[ik], eigvals_only=True)
    #E_base = np.stack(E_tube)
    E_tube = np.stack(E_tube)
    E_scroll = np.stack(E_scroll)

    T = 0.1
    #EF = calc_fermi(T, B, theta)
    #energy, dos = calc_DOS(T, Emax, B, theta)

    #fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    #ax0.set_title("No DSOI")
    #ax0.set_ylim([-0.5, Emax])
    #ax0.set_ylabel('E [meV]')
    ax1.set_xlabel(r'$k_zR$')
    ax1.set_title("Nanotube")
    ax1.set_ylim([-0.5, Emax])
    ax1.set_xlabel(r'$k_zR$')
    ax2.set_title("Nanoscroll")
    ax2.set_ylim([-0.5, Emax])
    ax2.set_xlabel(r'$k_zR$')
    ax1.set_ylabel(r'$E(k_z)$ [meV]')

    #ax0.plot(k*ip.R, E_base[:,:20]/ip.e*1e3, 'k')
    ax1.plot(k*ip.R, E_tube[:,:20]/ip.e*1e3, 'k')
    ax2.plot(k*ip.R, E_scroll[:,:20]/ip.e*1e3, 'k')
    #ax0.axhline(y=EF, color='gray', linestyle='dotted')
    #ax1.axhline(y=EF, color='gray', linestyle='dotted')
    #ax2.axhline(y=EF, color='gray', linestyle='dotted')
    plt.show()

if __name__=='__main__':
    if ip.material == 'InAs':
        Emax = 18
    if ip.material == 'InSb':
        Emax = 26
    
    #radius_plot(Rf=100e-9) # Has an odd quadratic-like behavior. Would anticipate asymptotic.
    #R = 35e-9
    k_plot(R=ip.R, B=ip.B, theta=ip.theta, Emax=6, k0=2.75/ip.R) # Works as expected
    #alpha_plot(R=ip.R) # One diamond occurs between alpha = {0.0, 2*R}
    #mag_plot(R=ip.R, B0=3) # Works as expected, pretty sure I've seen this in refs
    #angle_plot(B0=0.1)
    #energy_compare(R=ip.R, B=0.0, theta=np.pi/2, Emax=6, k0=np.pi/ip.R/2)


    '''
    It could be interesting to calculate
    DOS, or something similar, to add to 
    these graphs. It would be nice to have
    a way to show spin-polarization or 
    something else. Add "density of info"
    '''