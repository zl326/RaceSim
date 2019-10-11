# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:35:28 2019

@author: tom_m
"""

import matplotlib.pyplot as plt

# USER INPUTS
CdA = 0.28
Crr = 0.006
dt = 0.01
mCar = 860
rho = 1.2

vCar_init = 80

tol = 1


# CONSTANTS
kph2ms = 1/3.6
ms2kph = 3.6
g = 9.81

# Initialise
vCar = [vCar_init]
time = [0]

while vCar[-1] > tol :
    FAero = CdA * 0.5 * rho * (vCar[-1] * kph2ms)**2
    FTyre = mCar*g*Crr
    
    dvCar = dt * -(FAero + FTyre) / mCar

    vCar.append(vCar[-1] + dvCar)
    time.append(time[-1]+dt)


plt.plot(time, vCar)
plt.grid('on')


