from numpy import pi, sqrt, inf, sin, cos

material = 'InAs'

R = 50e-9
e = 1.60217663e-19
hbar = 1.05471817e-34
m0 = 9.1093837e-31
if material == 'InAs':
    meff = 0.023*m0
    alphaR = 28.3e-12*e
    betaD = 4e-12*e
    #alphaR = 0
    #betaD = 0
    geff = -14.9
if material == 'InSb':
    meff = 0.014*m0
    alphaR = 50e-12*e
    betaD = 30e-12*e
    #betaD = 0
    geff = -51.6
if material == 'InX':
    meff = 0.017*m0
    alphaR = 35e-12*e
    betaD = 17e-12*e
    geff = -51.6
muB = e*hbar/m0/2
B = 0.0
theta = 0
Bx = B*sin(theta)
Bz = B*cos(theta)
omega = e*B/meff
kB = 1.380649e-23
#kB = 1.0
if B != 0.0:
    lB = sqrt(hbar/e/B)
else:
    lB = inf
#mag_direction = 'x'
E0 = hbar**2/2/meff/R**2
Phi0 = 2*pi*hbar/e
Phi = pi*R**2*B
tau_e = 0.5e-12
tau_s = 1e-10
#area = pi*R**2
Rout = R + 0.1*R
Rin = R - 0.1*R
area = pi*(Rout**2-Rin**2)
L = (pi*kB/e)**2/3

if __name__=="__main__":
    #alphaR = 0.9*R*E0
    #betaD = 0.5*R*E0
    #print(alphaR/R/E0)
    print((alphaR/R/E0)**2 > abs(1-alphaR/R/E0)*5)
    #print(betaD/R/E0)
    #print(Phi/Phi0)
    #print((pi*kB/e)**2/3)
    #print(5e6*2*(pi*kB/e)**2/3)


