from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import simps

import input_data as ip
from hamiltonian import Hmat, quantum_numbers
from fermi_derivative import fermi_derivative

def main():

    k0 = 2*np.pi/ip.R
    Nk = 2000
    Nm = 40
    k = np.linspace(-k0, k0, Nk)

    with ProcessPoolExecutor() as pool:
        H_fut = pool.map(Hmat, k, repeat(ip.B), repeat(ip.theta), repeat(Nm))
    H = np.array([fut for fut in H_fut])
    E = np.zeros(Nk, dtype=object)
    V = np.zeros(Nk, dtype=object)
    for ik in range(Nk):
        E[ik], V[ik] = eigh(H[ik], eigvals_only=False, lower=True)
    E = np.stack(E)
    V = np.stack(V)
    Nstat = E.shape[1]

    T = 0.1
    mu = 12*ip.e/1e3

    fermi = fermi_derivative(k, E, mu, T, order=0)
    G = simps(np.stack([V[ik].conj().T@V[ik]*fermi[ik] for ik in range(Nk)]), k, axis=0)
    qn = quantum_numbers(Nm)

    Np = 30
    s = np.zeros(Np,dtype=complex)
    phi = np.linspace(0, 2*np.pi, Np)
    for p, pp in enumerate(phi):
        for a1 in range(Nstat):
            m1 = qn[a1, 0]
            s1 = qn[a1, 1]
            for a2 in range(Nstat):
                m2 = qn[a2, 0]
                s2 = qn[a2, 1]

                #if s1 == -s2:

                    #s[p] += s1*ip.hbar/2*np.real(np.exp(1.0j*(m2-m1)*pp)*G[a1,a2])/2/np.pi/ip.R
                s[p] += G[a1,a2]*4/np.pi/ip.R*np.exp(1.0j*(m2-m1)*pp)

    plt.plot(phi, np.real(s))
    plt.plot(phi, np.imag(s))
    plt.show()
if __name__=="__main__":
    main()
