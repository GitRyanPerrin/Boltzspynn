from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

import input_data as ip

def E(m, k):

    return ip.E0*(m**2 + (k*ip.R)**2)

def quantum_numbers(Nm=10):

    Nstat = 2*(2*Nm+1)

    a = 0
    qn = np.zeros([Nstat,2])
    for m in np.arange(-Nm, Nm+1, 1):
        for s in np.arange(-1, 2, 2):
            qn[a, 0] = m
            qn[a, 1] = s
            a+=1
    return qn

def Hmat(k, B, theta=ip.theta, Nm=10, alphaR=ip.alphaR, betaD=ip.betaD, R=ip.R):
    
    hbar = ip.hbar
    e = ip.e
    meff = ip.meff
    geff = ip.geff
    muB = ip.muB
    Bx = B*np.sin(theta)
    Bz = B*np.cos(theta)

    if Bx != 0.0:
        lBx = np.sqrt(hbar/e/abs(Bx))
    else:
        lBx = np.inf    
        
    if Bz != 0.0:
        lBz = np.sqrt(hbar/e/abs(Bz))
    else:
        lBz = np.inf

    omegax = e*Bx/meff
    omegaz = e*Bz/meff   

    Nstat = 2*(2*Nm+1)

    H = np.zeros([Nstat, Nstat], dtype=complex)

    qn = quantum_numbers(Nm)
    for a1 in range(Nstat):
        m1 = int(qn[a1, 0])
        s1 = int(qn[a1, 1])

        H[a1, a1] = E(m1, k)/2 #1

        for a2 in range(Nstat):
            m2 = int(qn[a2, 0])
            s2 = int(qn[a2, 1])

            # SOI (Non-magnetic terms)
            if m1 == m2 and s1 == s2:
                H[a1, a2] += alphaR/2/R*m1*s1 #6
                H[a1, a2] -= betaD*k*s1/2 #10

            if m1 == m2+1 and s1 == -s2:
                H[a1, a2] += 1.0j*alphaR*k*(1-s1)/2 #7
                H[a1, a2] += 1.0j*betaD/4/R*(1-s1)*(2*m1-1) #11

            
            # Parallel Magnetic Terms
            if s1 == s2:
                if m1 == m2:
                    H[a1, a2] += hbar*omegaz*R**2/8/lBz**2
                    H[a1, a2] -= hbar*omegaz*m1/4
                    H[a1, a2] += s1*geff*muB*Bz/4
                    # I'm not sure why, but Kokurin does not seem to have this term
                    #H[a1, a2] -= alphaR*s1*R/lBz**2/2
            
            if s1 == -s2:
                if m1 == m2:
                    H[a1, a2] += -1.0j*betaD*s1*R/lBz**2/2

            # Perpendicular Magnetic Terms
            if s1 == s2:
                if m1 == m2+1:
                    H[a1, a2] -= 1.0j*hbar*omegax*k*R/2 #2
                    H[a1, a2] += 1.0j*betaD*R/2/lBx**2*s1 #12
                if m1 == m2:
                    H[a1, a2] += hbar*omegax*R**2/8/lBx**2 #3
                if m1 == m2+2:
                    H[a1, a2] -= hbar*omegax*R**2/8/lBx**2 #4

            if s1 == -s2:
                if m1 == m2:
                    H[a1, a2] -= alphaR*R/4/lBx**2 #9
                if m1 == m2+2:
                    H[a1, a2] += (1-s1)*alphaR*R/4/lBx**2 #8
                if m1 == m2:
                    H[a1, a2] -= geff*muB*Bx/4 #5

            H[a2, a1] += np.conjugate(H[a1,a2])

    return H

if __name__=="__main__":
    Nm = 5
    Nk = 120
    k0 = np.pi/ip.R
    #k0 = 7*np.pi/ip.R
    k = np.linspace(-k0, k0, Nk)

    with ProcessPoolExecutor() as pool:
        H_fut = pool.map(Hmat, k, repeat(ip.B))

    H = np.array([fut for fut in H_fut])
    E = np.zeros(Nk, dtype=object)
    for ik in range(Nk):
        E[ik] = eigh(H[ik], eigvals_only=True)
    E = np.stack(E)

    #plt.title(rf'B=7 T, $\alpha$=20 meV$\cdot$nm, $\beta$=3 meV$\cdot$nm')
    plt.ylabel('E [meV]')
    plt.xlabel('k [1/nm]')
    plt.ylim([-1,4])
    #plt.ylim([-1,50])
    plt.plot(k/1e9, E[:,:20]/ip.e*1e3, 'k')
    plt.show()
