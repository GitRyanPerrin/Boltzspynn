import numpy as np
import matplotlib.pyplot as plt

import input_data as ip

with np.load("InAs_alpha_conds_E0.npz", allow_pickle=True) as file:
    A0 = file['arr_0']
    s_cond0 = file['arr_1']

with np.load("InAs_alpha_conds_E1.npz", allow_pickle=True) as file:
    A1 = file['arr_0']
    s_cond1 = file['arr_1']

with np.load("InAs_alpha_conds_E2.npz", allow_pickle=True) as file:
    A2 = file['arr_0']
    s_cond2 = file['arr_1']

with np.load("InAs_alpha_conds_E3.npz", allow_pickle=True) as file:
    A3 = file['arr_0']
    s_cond3 = file['arr_1']

with np.load("InAs_alpha_conds_E4.npz", allow_pickle=True) as file:
    A4 = file['arr_0']
    s_cond4 = file['arr_1']

fig, ax1 = plt.subplots()

ax1.set_ylabel(r"$\sigma_z^{(2)\varphi}$ [meV/V$^2$]", fontsize=16)
ax1.set_xlabel(r"$\alpha$ [meV$\cdot$nm]", fontsize=16)
#ax1.text(x=33, y=1.55, s=r'$\alpha_c^{0}$')
#ax1.text(x=44, y=1.55, s=r'$\alpha_c^{1}$')
#ax1.text(x=41, y=1.55, s=r'$\alpha_c^{3/2}$')

#ax1.axvline(x=34.7, color='k', linestyle='dotted', marker='^', markevery=10, clip_on=False)
#ax1.axvline(x=44.9, color='k', linestyle='dotted', marker='s', markevery=10, clip_on=False)

#ax1.axvline(x=43, color='gray', linestyle='dotted')
#ax1.axvline(x=55, color='skyblue', linestyle='dotted')

#ax1.text(x=38, y=1.3, s=r'$\Delta \alpha_c$')
#plt.arrow(34.75,1.25,9.,0,width=0.01, head_width=0.04, head_length=1, color='k')
#plt.arrow(34.25+9.5,1.25,-8.55,0,width=0.01, head_width=0.04, head_length=1, color='k')
#plt.arrow(43.25,1.5,10.5,0,width=0.02, head_width=0.06, head_length=1, color='k')

ax1.set_ylim([-0.005,0.8])
ax1.set_xlim([0,60])
plt.tight_layout()

#ax1.plot(A1,s_cond1, 'k', marker='s', markevery=6, label=r'$\mu = 0.0$')
#ax1.plot(A2,s_cond2, 'k', marker='^', markevery=6, label=r'$\mu = -0.1$')
#ax1.plot(A3,s_cond3, 'k', linestyle='dashed', label=r'$\mu = -0.25$')

#ax1.plot(A1, np.gradient(s_cond1), 'b', marker='^', markevery=6, label=r'$E_0$ $\beta = 30$ meV$\cdot$nm')
#ax1.plot(A2, np.gradient(s_cond2), 'b', marker='s', markevery=6, label=r'$E_1$ $\beta = 30$ meV$\cdot$nm')
#ax1.plot(A3, np.gradient(s_cond3), 'b', linestyle='dashed', label=r'$\beta = 0$ meV$\cdot$nm')

ax1.plot(A0, s_cond0, 'k')
ax1.plot(A1, s_cond1, 'k')
ax1.plot(A2, s_cond2, 'k')
ax1.plot(A3, s_cond3, 'k')
ax1.plot(A4, s_cond4, 'k')

plt.legend()
plt.show()