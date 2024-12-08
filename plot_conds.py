import numpy as np
import matplotlib.pyplot as plt

import input_data as ip
from density import calc_ndens

def load_band_structure(system):

    with np.load(f"./hamiltonians/{system}_InAs_50_{ip.alphaR/ip.e*1e12}_{ip.betaD/ip.e*1e12}_{ip.B}_0.5.npz") as file:
        k = file['arr_0']
        E = file['arr_1']
        V = file['arr_2']
        v = file['arr_3']
        norm = file['arr_4']

    return k, E, V, v, norm

def load_elec_charge_cond(system):
    with np.load(f"./conductivities/elec_charge_{system}_InAs_50_{ip.alphaR/ip.e*1e12}_{ip.betaD/ip.e*1e12}_{ip.B}_0.5.npz") as file:
        chem_potential = file['arr_0']
        econd = file['arr_1']
    return chem_potential, econd

def load_elec_ballistic_cond(system):
    with np.load(f"./conductivities/elec_charge_ball_{system}_InAs_50_{ip.alphaR/ip.e*1e12}_{ip.betaD/ip.e*1e12}_{ip.B}_0.5.npz") as file:
        chem_potential = file['arr_0']
        econd = file['arr_1']
    return chem_potential, econd

def load_elec_spin_cond(system, polarization='p'):
    with np.load(f"./conductivities/elec_spin_{system}_{polarization}_InAs_50_{ip.alphaR/ip.e*1e12}_{ip.betaD/ip.e*1e12}_{ip.B}_0.5.npz") as file:
        chem_potential = file['arr_0']
        scond = file['arr_1']
    return chem_potential, scond

def load_thermal_charge(system):
    with np.load(f"./conductivities/thermal_charge_{system}_InAs_50_{ip.alphaR/ip.e*1e12}_{ip.betaD/ip.e*1e12}_{ip.B}_0.5.npz") as file:
        chem_potential = file['arr_0']
        tecond = file['arr_1']
    return chem_potential, tecond

def load_thermal_spin(system, polarization='p'):
    with np.load(f"./conductivities/thermal_spin_{system}_{polarization}_InAs_50_{ip.alphaR/ip.e*1e12}_{ip.betaD/ip.e*1e12}_{ip.B}_0.5.npz") as file:
        chem_potential = file['arr_0']
        tscond = file['arr_1']
    return chem_potential, tscond

if __name__=='__main__':

    driving_field = 'electric'
    system = 'tube'
    k, E, V, v, norm = load_band_structure(system)

    chem_potential, econd = load_elec_charge_cond(system)
    chem_potential, ball_cond = load_elec_ballistic_cond(system)
    if driving_field == 'electric':
        chem_potential, spcond = load_elec_spin_cond(system, 'p')
        if system == 'scroll':
            fig, axs = plt.subplots(3,2,figsize=(7, 4),dpi=300)
            ax1 = axs[0,0]
            ax2 = axs[1,0]
            ax3 = axs[2,0]
            gridspec = axs[0,0].get_subplotspec().get_gridspec()
            for ax in axs[:,1]: ax.remove()
            subfig = fig.add_subfigure(gridspec[:,1])
            ax4 = subfig.subplots(1,1)
            chem_potential, szcond = load_elec_spin_cond(system, 'z')
        else:
            fig, axs = plt.subplots(2,2,figsize=(7, 4),dpi=300)
            ax1 = axs[0,0]
            ax2 = axs[1,0]
            gridspec = axs[0,0].get_subplotspec().get_gridspec()
            for ax in axs[:,1]: ax.remove()
            subfig = fig.add_subfigure(gridspec[:,1])
            ax4 = subfig.subplots(1,1)
        
    if driving_field == 'thermal':
        chem_potential, tecond = load_thermal_charge(system)
        chem_potential, tspcond = load_thermal_spin(system, 'p')
        if system == 'scroll':
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize=(3.375, 4),dpi=300)
            chem_potential, tszcond = load_thermal_spin(system, 'z')
        else:
            fig, (ax1, ax2, ax4) = plt.subplots(3,1,figsize=(3.375, 4),dpi=300)

    Nk = len(k)

    T = 0.1
    Nmu = 200
    dmu = 0.5*ip.e/1e3
    band_min = np.min(E)-0.1*ip.e/1e3
    band_max = np.min(E)+dmu
    chem_potential = np.linspace(band_min, band_max, Nmu)

    plt.rcParams.update({'font.size': 8})
    title_size = 10
    plt.autoscale()

    #mu_ticks = [chem_potential[i]/ip.e*1e3 for i in np.arange(0,Nmu,40)]
    #mu_labels = [round(chem_potential[i]/ip.e*1e3,1) for i in np.arange(0,Nmu,40)]
    mu_ticks = np.linspace(band_min/ip.e*1e3, band_max/ip.e*1e3, 6)
    #mu_labels = np.linspace(round(band_min/ip.e*1e3,1), round(band_max/ip.e*1e3,1), 6)
    mu_labels = [-0.4, 0.2, 0.8, 1.4, 2.0, 2.6]

    ndens = lambda mu: calc_ndens(k, E, mu*ip.e/1e3, T=T)/ip.area/1e23/2/np.pi


    if driving_field == 'thermal':
        ax1_n = ax1.twiny()

        ax1.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
        ax1.set_ylim([-15,30])
        ax1_n.set_xlim(ndens(band_min/ip.e*1e3), ndens(band_max/ip.e*1e3))
        ax2.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
        ax4.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
        ax4.set_ylim([0, 5.0])

        ax1.set_ylabel(r"$S_e$ [$\mu$V/K]", fontsize=title_size)
        ax1_n.set_xlabel(r'n$_{3D}$ [$10^{17}$ 1/cm$^3$]')
        ax2.set_ylabel(r"$\eta_\sigma^\varphi$ [mJ/(mK)$^2$]", fontsize=8)    
        ax4.set_ylabel(r"$k_zR$", fontsize=title_size)
        ax4.set_xlabel(r"E($k_z$) [meV]", fontsize=title_size)

        ax1.tick_params(axis='x', direction='in')
        ax2.tick_params(axis='x', direction='in')
        ax1.set_xticks(mu_ticks, labels=[])
        ax1.set_yticks([-10, 0, 10, 20], labels=['', 0, '', 20])
        ax2.set_xticks(mu_ticks, labels=[])
        ax2.set_yticks([-80, 0, 80, 160, 240, 320], labels=['', 0, '', 160, '', 320])
        ax4.set_xticks(mu_ticks, labels=mu_labels)
        ax4.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], labels=[0, 1.0, 2.0, 3.0, 4.0, ''])
        ylabel_pos = -0.125
        ax1.yaxis.set_label_coords(ylabel_pos, 0.5)
        ax2.yaxis.set_label_coords(ylabel_pos, 0.5)
        ax4.yaxis.set_label_coords(ylabel_pos, 0.5)

        plt.subplots_adjust(left=0.175, hspace=0, right=0.99, top=0.98)
        coeff = ip.e**2*ip.tau_e/2/np.pi/ip.area
        econd = coeff*econd
        ax1.plot(chem_potential/ip.e*1e3, tecond/econd*1e6, 'k')
        ax2.plot(chem_potential/ip.e*1e3, tspcond, 'k')
        if system == 'scroll':
            ax3.yaxis.set_label_coords(ylabel_pos, 0.5)
            ax3.tick_params(axis='x', direction='in')
            ax3.set_xticks(mu_ticks, labels=[])
        
            ax3.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
            ax3.set_ylabel(r"$\eta_\sigma^z$ [mJ/(mK)$^2$]", fontsize=8)
            ax3.set_xlabel(r"$\mu$ [meV]")
            ax3.set_ylim([-40, 160])
            ax3.set_yticks([-40, 0, 40, 80, 120, 160], labels=['', 0, '', 80, '', ''])
        
            ax3.plot(chem_potential/ip.e*1e3, tszcond, 'k')

    if driving_field == 'electric':
        ax1_G = ax1.twinx()
        ax1_n = ax1.twiny()

        ax1.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
        ax1.set_ylim([0, 50])
        #ax1_G.set_ylim([0, 2.5])
        ax2.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
 
        ax4.set_ylim([band_min/ip.e*1e3, band_max/ip.e*1e3])
        ax4.set_xlim([-1.5, 1.5])

        ax1.set_ylabel(r"$\sigma_e$ [S/mm]", fontsize=title_size)
        ax1_G.set_ylabel(r"$G_e$ [$2e^2/h$]", fontsize=title_size)
        ax1_n.set_xlabel(r'n$_{3D}$ [$10^{17}$ 1/cm$^3$]', fontsize=title_size)
        ax2.set_ylabel(r"$\sigma_\sigma^\varphi$ [meV/m$^2$]", fontsize=title_size)
        ax2.set_xlabel(r"$\mu$ [meV]")    
        ax4.set_xlabel(r"$k_zR$", fontsize=title_size)
        ax4.set_ylabel(r"E($k_z$) [meV]", fontsize=title_size)

        ax1.tick_params(axis='x', direction='in')
        if system=='scroll': ax2.tick_params(axis='x', direction='in')
        ax1.set_xticks(mu_ticks, labels=[])
        #ax1.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40], labels=[0, '', 10, '', 20, '', 30, '' , 40])
        #ax1.set_yticks([0, 12.5, 25, 37.5, 50, 62.5, 75], labels=[0, '', 25, '', 50, '', ''])
        ax1.set_yticks([0, 12.5, 25, 37.5, 50], labels=[0, '', 25, '', 50])
        #ax1_G.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5], labels=[0, '', 1, '', 2, ''])
        if system=='scroll':
            ax2.set_xticks(mu_ticks, labels=[])
            ax2.set_ylim([0, 1.25])
            ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25], labels=[0, '', 0.5,'', 1.0, ''])
        else:
            ax2.set_xticks(mu_ticks, labels=mu_labels)
            #ax2.set_ylim([0, 1])
            #ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0], labels=[0, '', 0.5,'',''])
            ax2.set_ylim([-0.1, 2.5])
            ax2.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5], labels=[0, '', 1.0, '', 2.0, ''])
        ax4.set_yticks(mu_ticks, labels=mu_labels)
        ax4.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], labels=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])

        ylabel_pos = -0.175
        ax1.yaxis.set_label_coords(ylabel_pos, 0.5)
        ax2.yaxis.set_label_coords(ylabel_pos, 0.5)
        #ax4.yaxis.set_label_coords(ylabel_pos, 0.5)

        plt.subplots_adjust(left=0.08, hspace=0, wspace=0.5, right=0.85, top=0.9, bottom=0.12)
        
        coeff = ip.e**2*ip.tau_e/2/np.pi/ip.area
        econd = econd*coeff
        ax1.plot(chem_potential/ip.e*1e3, econd/1e9, 'k')
        ax1_G.plot(chem_potential/ip.e*1e3, ball_cond*ip.hbar/ip.e/1e23, 'gray', linestyle='dashed')
        #ax1_G.axhline(y=0.7)
        ax1_G.scatter([-.3125],[0.7])
        ax2.scatter([-0.3125], [2.08])
        ax2.plot(chem_potential/ip.e*1e3, spcond/1e9, 'k')
        if system == 'scroll':
            #ax3.tick_params(axis='x', direction='in')
            ax3.set_xticks(mu_ticks, labels=mu_labels)
            
            ax3.yaxis.set_label_coords(ylabel_pos, 0.5)
            ax3.set_xlim([band_min/ip.e*1e3, band_max/ip.e*1e3])
            ax3.set_ylabel(r"$\sigma_\sigma^z$ [meV/m$^2$]", fontsize=title_size)
            ax3.set_xlabel(r"$\mu$ [meV]")
            ax3.set_ylim([-0.2,0.2])
            ax3.set_yticks([-0.2, -0.1, 0, 0.1, 0.2], labels=['', -0.1, 0, 0.1, ''])
    
            ax3.plot(chem_potential/ip.e*1e3, szcond/1e9, 'k')
    #ax4.plot(E[int(Nk/2):]/ip.e*1e3, k[int(Nk/2):]*ip.R, 'k')
    ax4.plot(k*ip.R, E/ip.e*1e3, 'k')

    plt.show()