from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import input_data as ip
from hamiltonian import Hmat as Hmat
#from hamiltonian_scroll import Hmat as Hmat
from density import calc_DOS, calc_fermi

def alpha_plot(Nm=8, B=ip.B, betaD=ip.betaD, R=ip.R, Ai=0.0, Af=100e-12*ip.e):

    Nk = 250
    k0 = 2*np.pi/ip.R
    k = np.linspace(-k0, k0, Nk)
    NA = 120
    A = np.linspace(Ai, Af, NA)

    min = np.zeros(NA, dtype=object)
    E = np.zeros(NA, dtype=object)
    for ia, a in enumerate(A):
        with ProcessPoolExecutor() as pool:
            H_fut = pool.map(
                Hmat,
                k,
                repeat(B),
                repeat(ip.theta),
                repeat(Nm),
                repeat(a),
                repeat(betaD),
                repeat(R)
            )

        H = np.array([fut for fut in H_fut])
        E[ia] = np.stack([eigh(H[ik], eigvals_only=True) for ik in range(Nk)])
        min[ia] = np.stack([np.min(E[ia][:,n]) for n in range(E[0].shape[1])])
    min = np.stack(min)
    E = np.stack(E)

    #plt.ylim([-0.5,8])
    plt.ylabel('E [meV]', fontsize=20)
    plt.xlabel(r'$\alpha$ [meV$\cdot$nm]', fontsize=20)
    plt.plot(A/ip.e*1e12, E[:,int(Nk/2),:22]/ip.e*1e3, 'k', linestyle='dashed', linewidth=3)
    plt.plot(A/ip.e*1e12, min[:,:22]/ip.e*1e3, 'k')

    #plt.text(x=16, y=2.05, s=r'j=$\pm$1/2')
    #plt.text(x=19, y=2.05, s=r'$\alpha_c^{1/2}$', fontsize=18)
    #plt.axvline(x=20, ymin=0.055, color='gray', linestyle='dotted')
    #plt.axvline(x=20, color='gray', linestyle='dotted')
    #plt.text(x=25, y=2.05, s=r'$\alpha_c^{3/2}$', fontsize=18)
    #plt.text(x=24, y=2.05, s=r'$\pm$3/2')
    #plt.axvline(x=26, color='gray', linestyle='dotted')

    plt.text(x=26, y=4.1, s=r'$\alpha_c^{1/2}$ = $\alpha_c^1$')
    plt.text(x=44, y=4.1, s= r'$\alpha_c^0$')
    plt.text(x=50, y=4.1, s=r'$\alpha_c^6$')
    plt.axvline(x=34, color='k', linestyle='dotted')
    plt.axvline(x=45, color='k', linestyle='dotted')
    plt.axvline(x=51, color='k', linestyle='dotted')

    #plt.text(x=25, y=8.5, s=r'$\pm$5/2')
    #plt.axvline(x=27, ymin=0.5, color='gray', linestyle='dotted')
    #plt.text(x=27, y=9, s=r'$\pm$7/2')
    #plt.axvline(x=29, ymin=0.97, color='gray', linestyle='dotted')
    #plt.axvline(x=29, color='gray', linestyle='dotted')
    plt.ylim([-0.5,4])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    gap = Line2D([0],[0], color='black', linewidth=3, linestyle='dashed', label=r'$k_z=0$')
    min = Line2D([0],[0], color='black', label='Minima')
    plt.legend(handles=[gap, min])
    plt.show()

def gap_plot(Nm=8, B=ip.B, betaD=ip.betaD, R=ip.R, Ai=0.0, Af=100e-12*ip.e):

    Nk = 250
    k0 = 2*np.pi/ip.R
    k = np.linspace(-k0, k0, Nk)
    NA = 50
    A = np.linspace(Ai, Af, NA)

    min = np.zeros(NA, dtype=object)
    E = np.zeros(NA, dtype=object)
    for ia, a in enumerate(A):
        with ProcessPoolExecutor() as pool:
            H_fut = pool.map(
                Hmat,
                k,
                repeat(B),
                repeat(ip.theta),
                repeat(Nm),
                repeat(a),
                repeat(betaD),
                repeat(R)
            )

        H = np.array([fut for fut in H_fut])
        E[ia] = np.stack([eigh(H[ik], eigvals_only=True) for ik in range(Nk)])
        min[ia] = np.stack([np.min(E[ia][:,n]) for n in range(E[0].shape[1])])
    min = np.stack(min)
    E = np.stack(E)

    SOGap1 = E[:,int(Nk/2),0]-E[:,int(Nk/2),2]
    SOGap2 = E[:,int(Nk/2),1]-E[:,int(Nk/2),2]
    for ia in range(NA):
        if min[ia,0] == E[ia,int(Nk/2),0]:
            SOGap1[ia] = None
        else: 
            print(A[ia])
        if min[ia,1] == E[ia,int(Nk/2),1]:
            SOGap2[ia] = None


    #inflection1 = -np.min(SOGap1[int(NA/2):])
    #inflection2 = -np.min(SOGap2[int(NA/2):])
    #print(inflection/ip.e*1e3)

    #plt.ylim([-0.5,8])
    plt.ylabel('$\Delta E$ [meV]', fontsize=20)
    plt.xlabel(r'$\alpha$ [meV$\cdot$nm]', fontsize=20)
    #plt.plot(A/ip.e*1e12, -(E[:,int(Nk/2),0]-E[:,int(Nk/2),2])/ip.e*1e3, 'k', linestyle='dotted', label='SO Gap')
    plt.plot(A/ip.e*1e12, -SOGap1/ip.e*1e3, 'k', linestyle='dashed', marker='s', markevery=10, label=r'SO Gap $\Delta E_{0\rightarrow 2}$')
    plt.plot(A/ip.e*1e12, -SOGap2/ip.e*1e3, 'k', linestyle='dashed', marker='^', markevery=10, label=r'SO Gap $\Delta E_{1\rightarrow 2}$')
    plt.plot(A/ip.e*1e12, (E[:,int(Nk/2),0]-min[:,0])/ip.e*1e3, 'k', linestyle='solid', marker='s', markevery=10, label=r'SO Minimum Depth $E_0$')
    plt.plot(A/ip.e*1e12, (E[:,int(Nk/2),1]-min[:,1])/ip.e*1e3, 'k', linestyle='solid', marker='^', markevery=10, label=r'SO Minimum Depth $E_1$')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    #plt.ylim([-0.06, 1.0])

    #plt.scatter(34, -0.06, color='k', marker='^', clip_on=False)
    #plt.scatter(45, -0.06, color='k', marker='s', clip_on=False)

    plt.text(x=33, y=1.1, s=r'$\alpha_c^{0}$')
    plt.text(x=44, y=1.1, s=r'$\alpha_c^{1}$')
    plt.text(x=37, y=0.5, s=r'$\Delta \alpha_c$')
    plt.arrow(34.45,0.46,9.3,0,width=0.01, head_width=0.04, head_length=1, color='k')
    plt.arrow(34.25+9.5,0.46,-8.55,0,width=0.01, head_width=0.04, head_length=1, color='k')

    plt.axvline(x=34.7, color='k', linestyle='dotted', marker='^', markevery=10, clip_on=False)
    plt.axvline(x=44.9, color='k', linestyle='dotted', marker='s', markevery=10, clip_on=False)

    #plt.text(x=19, y=0.38, s=r'$\alpha_c^{1/2}$', fontsize=18)
    #plt.axvline(x=20, ymin=0.055, color='gray', linestyle='dotted')
    #plt.axvline(x=20.5, color='gray', linestyle='dotted')
    #plt.text(x=29, y=0.38, s=r'$\alpha_c^{7/2}$')
    #plt.axvline(x=29, ymin=0.97, color='gray', linestyle='dotted')
    #plt.axvline(x=29.5, color='gray', linestyle='dotted')
    #plt.text(x=29.5, y=0.38, s=r'$\alpha_c^{9/2}$')
    #plt.axvline(x=30.2, color='gray', linestyle='dotted')

    #plt.text(x=33, y=0.38, s=r'$\Delta E_{SO,min}$')
    #plt.axvline(x=33, color='gray', linestyle='dotted')

    plt.tight_layout()
    plt.legend(fontsize=10)
    plt.show()

if __name__=="__main__":
    alpha_plot()
    #gap_plot()