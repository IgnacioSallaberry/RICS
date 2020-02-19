# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:25:06 2019
@author: Ignacio Sallaberry
"""

import numpy as np
import matplotlib.pyplot as plt


#==============================================================================
#                                Tipografía de los gráficos
#==============================================================================    
SMALL_SIZE = 28
MEDIUM_SIZE = 36
BIGGER_SIZE = 39
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=22)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#==============================================================================

plt.ioff()


tau = np.geomspace(1e-6,10,num=100)

i=0
ruido1 = []
ruido2 = []

#ruido = np.arange()
for i in tau:
    ruido1.append(np.random.uniform(9,21))#-np.random.uniform(14,16))
    ruido2.append(np.random.uniform(14,16))#-np.random.uniform(14,16))


s=1000
plt.figure()
plt.hlines(15,0,len(ruido1))  #linea que marca el valor del t_diff
plt.plot(ruido1,'limegreen')
plt.xlabel('Tiempo(s)')
plt.ylabel('Intensidad \n (Kcpms)')
plt.ylim(0,25)
plt.tick_params(which='minor', length=2, width=1.5)
plt.tick_params(which='major', length=4, width=2)
plt.show()

plt.figure()
plt.hlines(15,0,len(ruido2))  #linea que marca el valor del t_diff
plt.plot(ruido2,'limegreen')
plt.xlabel('Tiempo(s)')
plt.ylabel('Intensidad \n (Kcpms)')
plt.ylim(0,25)
plt.tick_params(which='minor', length=2, width=1.5)
plt.tick_params(which='major', length=4, width=2)
plt.show()



mu1 = 15 #np.mean(ruido) # media
sigma1 = np.std(ruido1) #desviación estándar     std = sqrt(mean(abs(x - x.mean())**2))

mu2 = 15#np.mean(ruido) # media
sigma2 = np.std(ruido2) #desviación estándar     std = sqrt(mean(abs(x - x.mean())**2))


def Gaussiana(mu,sigma):
     
    x_inicial = 0
    x_final = 30
    x_gaussiana=np.linspace(x_inicial,x_final,num=1000) # armo una lista de puntos donde quiero graficar la distribución de ajuste
   
    gaussiana3=(1/np.sqrt(2*np.pi*(sigma**2)))*np.exp((-.5)*((x_gaussiana-mu)/(sigma))**2)
#    gaussiana3= for g in gaussiana3:g/max(gaussiana3)

    return (x_gaussiana,gaussiana3)

x=np.linspace(0,30,1000)
Gausiana1 = Gaussiana(mu1,sigma1)[1]
Gausiana2 = Gaussiana(mu2,sigma2)[1]
Gausiana1=Gausiana1/max(Gausiana1)
Gausiana2=Gausiana2/max(Gausiana2)

plt.figure()
plt.plot(x,Gausiana1, color='tomato')
plt.axvline(x=mu1,ymin=0.05,ymax=0.95)
plt.show()

plt.figure()
plt.plot(x,Gausiana2, color='tomato')

plt.axvline(x=mu1,ymin=0.05,ymax=0.95)
plt.show()
