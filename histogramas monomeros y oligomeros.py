# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:25:06 2019
@author: Ignacio Sallaberry
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import re

guardar_imagenes = True
#==============================================================================
#                                Tipografía de los gráficos
#==============================================================================  
plt.close('all') # amtes de graficar, cierro todos las figuras que estén abiertas  
SMALL_SIZE = 54
MEDIUM_SIZE = 70
BIGGER_SIZE = 75
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=22)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#mpl.rcParams['axes.linewidth'] = 1  ## maneja el ancho de las lineas del recuadro de la figura
#==============================================================================
#==============================================================================





#23hs_cell4_rics_cyto_DIFFERENCE_DIFUSION
with open('C:\\Users\\ETCasa\\Desktop\\monomeros_NandB_histograma.txt') as fobj:
    monomeros = fobj.read()
monomeros = re.split('\t|\n', monomeros)
monomeros.remove('X')
monomeros.remove('BarSeries1')
monomeros.remove('')
monomeros.remove('')
    


with open('C:\\Users\\ETCasa\\Desktop\\oligomeros_NandB_histograma.txt') as fobj:
    oligomeros = fobj.read()
oligomeros = re.split('\t|\n', oligomeros)
oligomeros.remove('X')
oligomeros.remove('BarSeries1')
oligomeros.remove('')
oligomeros.remove('')



MONOMEROS_brillo=[]
MONOMEROS_pixel=[]

OLIGOMEROS_brillo=[]
OLIGOMEROS_pixel=[]


i=0
while i< len(monomeros):
    MONOMEROS_brillo.append(float(monomeros [i]))
    MONOMEROS_pixel.append(float(monomeros [i+1]))
    
    OLIGOMEROS_brillo.append(float(oligomeros [i]))
    OLIGOMEROS_pixel.append(float(oligomeros [i+1]))
    
    i+=2

#==============================================================================
#==============================================================================
# CALCULAR LA MEDIA, LA DESVIACIÓN ESTÁNDAR DE LOS DATOS y EL NÚMERO TOTAL DE CUENTAS
#==============================================================================

#
#mu_ajuste_MONOM=np.mean(MONOMEROS_brillo) # media
#sigma_ajuste_MONOM=np.std(MONOMEROS_brillo) #desviación estándar     std = sqrt(mean(abs(x - x.mean())**2))
#N_ajuste_MONOM=len(MONOMEROS_pixel) # número de mediciones
#std_err_ajuste_MONOM= sigma_ajuste_MONOM / N_ajuste_MONOM # error estándar
#
#
#
#def Gaussiana(mu,sigma):
#     
#    x_inicial = 0
#    x_final = mu+2*sigma
#    x_gaussiana=np.linspace(x_inicial,x_final,num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste
#   
##    gaussiana3=(1/np.sqrt(2*np.pi*(sigma**2)))*np.exp((-.5)*((x_gaussiana-mu)/(sigma))**2)
#    gaussiana3=np.exp((-.5)*((x_gaussiana-mu)/(sigma))**2)
#
#    return (x_gaussiana,gaussiana3)


###==============================================================================    
###                               Diferencias ajuste por Difusion
###==============================================================================    
#
#plt.plot(Gaussiana(2.7,0.48)[0],
#         Gaussiana(2.7,0.48)[1],
#         '--', color='tomato', label='Ajuste: Difusión \n $\mu$= {:10.3E} $\\sigma$ = {:10.3E}'.format(mu_ajuste_MONOM, sigma_ajuste_MONOM)
#
#         )





plt.figure(figsize=(19,12))
plt.bar(MONOMEROS_brillo, MONOMEROS_pixel,linewidth=3,log=True, color='darkcyan')
plt.xlim([0,max(OLIGOMEROS_brillo)+1])
#plt.xlim([0,max(MONOMEROS_brillo)+1])
plt.ylim([0,max(MONOMEROS_pixel)])
plt.xlabel('Brillo')
plt.ylabel('Número\n de píxeles')
plt.tick_params(which='minor', length=8, width=3.5)
plt.tick_params(which='major', length=10, width=5)
figManager = plt.get_current_fig_manager()  ####   esto y la linea de abajo me maximiza la ventana de la figura
figManager.window.showMaximized()           ####   esto y la linea de arriba me maximiza la ventana de la figura
plt.show()
if guardar_imagenes:
    plt.savefig('C:\\Users\\ETCasa\\Desktop\\histograma_monomeros.svg', format='svg')






plt.figure(figsize=(19,12))
plt.bar(OLIGOMEROS_brillo, OLIGOMEROS_pixel,linewidth=3,log=True, color='darkcyan')
plt.xlabel('Brillo')
plt.ylabel('Número\n de píxeles')
plt.xlim([0,max(OLIGOMEROS_brillo)+1])
plt.ylim([0,max(MONOMEROS_pixel)])
plt.tick_params(which='minor', length=8, width=3.5)
plt.tick_params(which='major', length=10, width=5)
figManager = plt.get_current_fig_manager()  ####   esto y la linea de abajo me maximiza la ventana de la figura
figManager.window.showMaximized()           ####   esto y la linea de arriba me maximiza la ventana de la figura
plt.show()
if guardar_imagenes:
    plt.savefig('C:\\Users\\ETCasa\\Desktop\\histograma_oligomeros.svg', format='svg')




