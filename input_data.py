from numpy import pi, sqrt, inf, sin, cos, log
import numpy as np
import matplotlib.pyplot as plt

material = 'InAs'

R = 50e-9
e = 1.60217663e-19
hbar = 1.05471817e-34
m0 = 9.1093837e-31
Phi0 = 2*pi*hbar/e

if material == 'InAs':
    meff = 0.023*m0
    alphaR = 20e-12*e
    betaD = 4e-12*e
    #alphaR = 0
    #betaD = 0e-12*e
    geff = -14.9
if material == 'InSb':
    meff = 0.014*m0
    alphaR = 50e-12*e
    betaD = 30e-12*e
    #betaD = 0
    #alphaR = -5*betaD/2/sqrt(3)
    geff = -51.6
if material == 'InX':
    meff = (0.023+0.014)*m0/2
    alphaR = (50+20)/2e12*e
    betaD = (30+4)/2e12*e
    geff = (-14.9-51.6)/2
muB = e*hbar/m0/2
B = 0.0
theta = pi/2
Bx = B*sin(theta)
Bz = B*cos(theta)
omega = e*B/meff
kB = 1.380649e-23
if B != 0.0:
    lB = sqrt(hbar/e/B)
else:
    lB = inf
#mag_direction = 'x'
E0 = hbar**2/2/meff/R**2
Phi = pi*R**2*B
tau_e = 0.5e-12
tau_s = 1e-10
#area = pi*R**2
Rout = R + 0.1*R
Rin = R - 0.1*R
area = pi*(Rout**2-Rin**2)
lorentz = (pi*kB/e)**2/3
lorseeb = kB/e

if __name__=="__main__":
    #alphaR = 0.9*R*E0
    #betaD = 0.5*R*E0
    #print(alphaR/R/E0)
    #print((alphaR/R/E0)**2 > abs(1-alphaR/R/E0)*9)
    #print((alphaR/R/E0)**2 > abs(1-alphaR/R/E0)*(1+2*Phi/Phi0))
    #print(betaD/R/E0)
    #print(Phi/Phi0)
    #print(2*e**2/hbar/2/pi)
    #print(0.4*lorseeb*1e6)
    #print(betaD/R, betaD*R/lB**2)
    #print(7.34024216e-23/e*1e3)
    #print(6.35327396e-23/e*1e3)
    #print(5.558571981632653e-30/e*1e12)
    #print(7.193446093877551e-30/e*1e12)
    print(E0/e*1e3)
    print(alphaR*1e3/e/R)
    print(((alphaR/R)**2/E0 - E0 + alphaR/R)/e*1e3)
    print((2*meff/hbar**2*alphaR**2-hbar**2/2/meff/R**2+alphaR/R)/e*1e3)
    print(2*meff/hbar**2*alphaR**2/2/e*1e3)

    print(alphaR/2 - sqrt((1-alphaR/R)**2 + betaD**2/12))

    A = np.linspace(0, 3*alphaR, 30)
    def f(A):
        return -(2*meff/hbar**2*A**2/2-hbar**2/2/meff/R**2+A/R)/e*1e3
    def g(A):
        return A/R/E0 - sqrt((1-A/R/E0)**2*9/4 + betaD**2/R/E0**2/12*9/4)
    #plt.plot(A*1e12/e, f(A)/e*1e3)
    plt.plot(A*1e12/e, g(A))
    plt.show()