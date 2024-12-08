from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

import input_data as ip
from hamiltonian import Hmat as Hmat
from hamiltonian_scroll import Hmat as Hmat_scroll
from density import calc_DOS, calc_fermi

def k_plot(Nm=5, B=ip.B, alphaR=ip.alphaR, betaD=ip.betaD, R=ip.R, theta=ip.theta, k0=np.pi/ip.R, Emax=20):
    Nk = 250
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

    b0 = E[int(Nk/2), 0]/ip.e*1e3
    b1 = E[int(Nk/2), 1]/ip.e*1e3
    b2 = E[int(Nk/2), 2]/ip.e*1e3
    b3 = E[int(Nk/2), 3]/ip.e*1e3
    b4 = E[int(Nk/2), 4]/ip.e*1e3
    b5 = E[int(Nk/2), 5]/ip.e*1e3
    b6 = E[int(Nk/2), 6]/ip.e*1e3
    b7 = E[int(Nk/2), 7]/ip.e*1e3
    b8 = E[int(Nk/2), 8]/ip.e*1e3

    m0 = np.min(E[:, 0])/ip.e*1e3
    m1 = np.min(E[:, 1])/ip.e*1e3
    m4 = np.min(E[:, 4])/ip.e*1e3
    m5 = np.min(E[:, 5])/ip.e*1e3
    m6 = np.min(E[:, 6])/ip.e*1e3

    #T = 0.1
    #EF = calc_fermi(T, B, theta)
    #energy, dos = calc_DOS(T, E[int(Nk/2), 20]/ip.e*1e3)
    #energy, dos = calc_DOS(T, Emax, B, theta)

    #fig, (ax1, ax2) = plt.subplots(1, 2)
    fig, ax1 = plt.subplots(figsize=(6,8))
    #ax1.set_aspect(aspect=1.2)
    #ax1.set_ylim([-0.25, E[int(Nk/2), 20]/ip.e*1e3])
    #ax1.set_title(rf'B = {B} T, $\alpha$ = {round(alphaR/1e-12/ip.e)} meV$\cdot$nm, $\beta$ = {betaD/1e-12/ip.e} meV$\cdot$nm')
    #ax1.set_title(rf'$\alpha$ = {round(alphaR/1e-12/ip.e)} meV$\cdot$nm, $\beta$ = {round(betaD/1e-12/ip.e)} meV$\cdot$nm, $B$ = {B} T')
    ax1.set_title(rf'InAs: $\alpha$ = {round(alphaR/1e-12/ip.e)} meV$\cdot$nm, $\beta$ = {round(betaD/1e-12/ip.e)} meV$\cdot$nm', fontsize=18)
    ax1.set_ylim([-0.15, Emax])
    #ax1.set_ylim([-0.5, Emax])
    #ax1.set_xlim([-np.pi, np.pi])
    #ax1.set_xlim([-2.25, 2.25])
    ax1.set_xlim([-2, 2])
    ax1.set_ylabel(r'$E$ [meV]', fontsize=22)
    ax1.set_xlabel(r'$k_zR$', fontsize=22)
    plt.xticks([-2,-1,0,1,2], [-2, -1, 0, 1, 2], fontsize=20)
    plt.yticks(fontsize=20)
    ax1.axhline(y=m0, color='gray', linewidth=0.5)
    #ax1.axhline(y=b0, color='gray', linewidth=0.5)
    #ax1.axhspan(xmin=-2, xmax=2, ymin=m0, ymax=b0, color='lightgray', alpha=0.3)
    #ax1.axhline(y=m1, color='gray', linewidth=0.5)
    ax1.axhspan(xmin=-2, xmax=2, ymin=m0, ymax=b1, color='lightgray', alpha=0.3)
    ax1.axhline(y=b1, color='gray', linewidth=0.5)
    ax1.axhspan(xmin=-2, xmax=2, ymin=b1, ymax=b2, color='lightgray', alpha=0.9)
    ax1.axhline(y=b2, color='gray', linewidth=0.5)
    #ax1.axhspan(xmin=-2, xmax=2, ymin=b1, ymax=b2, color='lightgray', alpha=0.9)
    #ax1.axhline(y=b3, color='gray', linewidth=0.5)
    #ax1.axhspan(xmin=-2, xmax=2, ymin=b2, ymax=b3, color='lightgray', alpha=0.5)

    ax1.axhline(y=m4, color='gray', linewidth=0.5)
    #ax1.axhline(y=b4, color='gray', linewidth=0.5)
    ax1.axhspan(xmin=-2, xmax=2, ymin=m4, ymax=b5, color='lightgray', alpha=0.3)
    #ax1.axhline(y=m5, color='gray', linewidth=0.5)
    ax1.axhline(y=b5, color='gray', linewidth=0.5)
    #ax1.axhspan(xmin=-2, xmax=2, ymin=m5, ymax=b5, color='lightgray', alpha=0.3)
    ax1.axhline(y=b6, color='gray', linewidth=0.5)
    ax1.axhspan(xmin=-2, xmax=2, ymin=b5, ymax=b6, color='lightgray', alpha=0.9)
    #ax1.axhline(y=b7, color='gray', linewidth=0.5)
    #ax1.axhspan(xmin=-2, xmax=2, ymin=b5, ymax=b6, color='lightgray', alpha=0.9)
    #ax1.axhline(y=m5, color='gray', linewidth=0.5)
    #ax1.axhline(y=b4, color='gray', linewidth=0.5)
    #ax1.axhspan(xmin=-2, xmax=2, ymin=b6, ymax=b7, color='lightgray', alpha=0.5)

    #ax2.set_ylim([-0.25, E[int(Nk/2), 20]/ip.e*1e3])
    #ax2.set_ylim([-0.5, Emax])
    #ax2.set_xlabel(r'DOS [meV$^{-1}\cdot$cm$^{-2}$]')
    ax1.plot(k*ip.R, E[:,:20]/ip.e*1e3, 'k')
    #ax2.plot(dos, energy, 'k')
    #ax1.axhline(y=EF, color='gray', linestyle='dotted')
    #ax2.axhline(y=EF, color='gray', linestyle='dotted')
    plt.tight_layout()
    plt.show()

def label_plot(Nm=5, B=ip.B, alphaR=ip.alphaR, betaD=ip.betaD, R=ip.R, theta=ip.theta, k0=np.pi/ip.R, Emax=20):
    Nk = 250
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

    H = np.array([fut for fut in H_fut])
    E = np.zeros(Nk, dtype=object)
    for ik in range(Nk):
        E[ik] = eigh(H[ik], eigvals_only=True)
    E = np.stack(E)

    fig, ax1 = plt.subplots()
    ax1.set_ylim([-0.15, Emax])
    ax1.set_xlim([-1,1])
    ax1.set_ylabel('E [meV]', fontsize=22)
    ax1.set_xlabel('$k_zR$', fontsize=22)
    ax1.set_yticks([-0.1, 0.0, 0.1, 0.2])
    ax1.set_xticks([-1, -0.5, 0.0, 0.5, 1])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    ax1.plot(k*ip.R, E[:,:20]/ip.e*1e3, 'k')
    plt.tight_layout()
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

def alpha_plot(Nm=5, k=0.0, B=ip.B, betaD=ip.betaD, R=ip.R, Ai=0.0, Af=50e-12*ip.e):
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

    plt.ylim([-0.5,4])
    plt.ylabel('E [meV]')
    plt.xlabel(r'$\alpha$ [meV$\cdot$nm]')
    plt.plot(A/ip.e*1e12, E[:,:20]/ip.e*1e3, 'k')
    plt.text(x=16, y=8.1, s=r'j=$\pm$1/2')
    plt.axvline(x=20, ymin=0.055, color='gray', linestyle='dotted')
    plt.text(x=23, y=8.1, s=r'$\pm$3/2')
    plt.axvline(x=26, ymin=0.2, color='gray', linestyle='dotted')
    plt.text(x=25, y=8.5, s=r'$\pm$5/2')
    plt.axvline(x=27, ymin=0.5, color='gray', linestyle='dotted')
    plt.text(x=27, y=9, s=r'$\pm$7/2')
    plt.axvline(x=29, ymin=0.97, color='gray', linestyle='dotted')
    #plt.plot(A/ip.R/ip.E0, E[:,:20]/ip.e*1e3, 'k')
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
    EF = calc_fermi(T, B, theta)
    energy, dos = calc_DOS(T, Emax, B, theta)

    #fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, dpi=200)

    #ax0.set_title("No DSOI")
    #ax0.set_ylim([-0.5, Emax])
    #ax0.set_ylabel('E [meV]')
    ax1.set_ylabel(r'$E(k_z)$ [meV]')
    ax1.set_xlabel(r'$k_zR$')
    ax1.set_title("Nanotube")
    ax1.set_ylim([-0.5, Emax])
    ax1.set_xlabel(r'$k_zR$')
    ax2.set_title("Nanoscroll")
    ax2.set_ylim([-0.5, Emax])
    ax2.set_xlabel(r'$k_zR$')

    #ax0.plot(k*ip.R, E_base[:,:20]/ip.e*1e3, 'k')
    ax1.plot(k*ip.R, E_tube[:,:20]/ip.e*1e3, 'k')
    ax2.plot(k*ip.R, E_scroll[:,:20]/ip.e*1e3, 'k')
    #ax0.axhline(y=EF, color='gray', linestyle='dotted')
    ax1.axhline(y=EF, color='gray', linestyle='dotted')
    ax1.text(x=3.5, y=EF, s=r'$E_F$')
    ax2.axhline(y=EF, color='gray', linestyle='dotted')
    ax2.text(x=3.5, y=EF, s=r'$E_F$')
    plt.show()

if __name__=='__main__':
    #radius_plot(Rf=100e-9)
    R = ip.R
    #label_plot(Nm=10,R=R, B=ip.B, theta=ip.theta, Emax=0.25, k0=1/ip.R)
    k_plot(Nm=10,R=R, B=ip.B, theta=ip.theta, Emax=2, k0=1*np.pi/ip.R) # Works as expected
    #alpha_plot(k=0, R=50e-9) # One diamond occurs between alpha = {0.0, 2*R}
    #mag_plot(R=ip.R, B0=0.5) # Works as expected, pretty sure I've seen this in refs
    #angle_plot(B0=0.1)
    #energy_compare(R=ip.R, B=0.0, theta=np.pi/2, Emax=5, k0=np.pi/ip.R/2)