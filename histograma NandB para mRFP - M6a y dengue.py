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
#==============================================================================
# CARGO LOS DATOS 
#==============================================================================

### mRFP_citoplasmatica
#with open('C:\\Users\\ETCasa\\Desktop\\valores_histograma_brillo_mRFP_citplasmatica.txt') as fobj:


#### M6A_mRFP
#with open('C:\\Users\\ETCasa\\Desktop\\histograma_brillo_M6a.txt') as fobj:


### DENGUE
with open('C:\\Users\\ETCasa\\Desktop\\histogramas brillo dengue.txt') as fobj:


    monomeros = fobj.read()
monomeros = re.split('\t|\n', monomeros)
monomeros.remove('X')
monomeros.remove('BarSeries1')
monomeros.remove('')
monomeros.remove('')
    


MONOMEROS_brillo=[]
MONOMEROS_pixel=[]



i=0
while i< len(monomeros):
    MONOMEROS_brillo.append(float(monomeros [i]))
    MONOMEROS_pixel.append(float(monomeros [i+1]))
    
    
    i+=2



plt.bar(MONOMEROS_brillo, MONOMEROS_pixel,linewidth=3,log=True, color='darkcyan',width=0.1)
plt.xlim([0,5.5])
plt.ylim([0,max(MONOMEROS_pixel)])
plt.xlabel('Brillo')
plt.ylabel('Número\n de píxeles')
plt.tick_params(which='minor', length=8, width=3.5)
plt.tick_params(which='major', length=10, width=5)
figManager = plt.get_current_fig_manager()  ####   esto y la linea de abajo me maximiza la ventana de la figura
figManager.window.showMaximized()           ####   esto y la linea de arriba me maximiza la ventana de la figura
plt.show()
if guardar_imagenes:
    plt.savefig('C:\\Users\\ETCasa\\Desktop\\histogramas_brillo_dengue.svg', format='svg') 




#==============================================================================
# PARAMETROS DE TAMAÑAO DE LETRA DE LOS GRAFICOS
#==============================================================================


#SMALL_SIZE = 34
#MEDIUM_SIZE = 42
#BIGGER_SIZE = 45
#
##font = {'weight' : 'normal'}
#
#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title    
#plt.rc('lines', linewidth=3)
##plt.rcParams['axes.labelweight'] = 'normal'
#plt.tick_params(labelsize=MEDIUM_SIZE)



#plt.figure(2)
#plt.bar(OLIGOMEROS_brillo, OLIGOMEROS_píxel,linewidth=3,log=True, color='darkcyan')
#plt.xlabel('Brillo')
#plt.ylabel('Número\n de píxeles')
#plt.xlim([0,max(OLIGOMEROS_brillo)+1])
#plt.ylim([0,max(MONOMEROS_píxel)])
#plt.tick_params(which='minor', length=7, width=1.5)
#plt.tick_params(which='major', length=10, width=3.5)
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0) #hace que no me corte los márgenes
#plt.show()   
#
#
#









