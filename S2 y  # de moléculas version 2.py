# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:25:06 2019
@author: Ignacio Sallaberry
"""

import numpy as np
import matplotlib.pyplot as plt

SMALL_SIZE = 30
MEDIUM_SIZE = 36
BIGGER_SIZE = 39

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#==============================================================================
#                                  Parametros globales iniciales
#==============================================================================   
box_size = 256
roi = 128
tp = 1e-6             #seg    
tl = box_size * tp    #seg
dr = 0.05             # delta r = pixel size =  micrometros
w0 = 0.25              #radio de la PSF  = micrometros    
wz = 1.5              #alto de la PSF desde el centro  = micrometros


vol_caja = (dr*box_size)**3 

vol_PSF = (w0**2)*wz*((np.pi)**(1.5))

partic_en_caja_dif = [4000,3000,2000,1600,1000,750,500,300,250,150,100,50]

concentracion_en_PSF_dif = []

i=0
while i<len(partic_en_caja_dif):
    partic_en_PSF_dif = vol_PSF * partic_en_caja_dif[i] / vol_caja    #en femtolitros
    concentracion_en_PSF_dif.append(partic_en_PSF_dif/vol_PSF * 1e15)   # cantidad de moleculas en PSF/ litro)

    i+=1

un_mol=6.023e23  ## moleculas en un mol
concentracion_molar_en_PSF_dif = []
for i in concentracion_en_PSF_dif:
    concentracion_molar_en_PSF_dif.append(i/un_mol)
    

partic_en_caja_asoc_disoc = [2000,1600,1000,750,500,300,250,150,100,50]  
concentracion_en_PSF_asoc_disoc = []

i=0
while i<len(partic_en_caja_asoc_disoc):
    partic_en_PSF_asoc_disoc = vol_PSF * partic_en_caja_asoc_disoc[i] / vol_caja    #en femtolitros
    concentracion_en_PSF_asoc_disoc.append(partic_en_PSF_asoc_disoc/vol_PSF * 1e15)   # cantidad de moleculas en PSF/ litro)

    i+=1

un_mol=6.023e23  ## moleculas en un mol
concentracion_molar_en_PSF_asoc_disoc = []
for i in concentracion_en_PSF_asoc_disoc:
    concentracion_molar_en_PSF_asoc_disoc.append(i/un_mol)  
    

S2_asoc_disoc = [0.0000893636,0.0000775362,0.0002823328,0.0004867565,0.0015149203,0.0021687680,0.0052024122,0.0142299307,0.0413098291,0.1375668058]  
S2_difusion = [0.0000060492,0.0000102254,0.0000286686,0.0000502642,0.0001507467,0.0001554366,0.0003864523,0.0011281431,0.0022476617,0.0068247176,0.0320095827,0.1020694733]

S2_manu_difusion = 4.314e-7
concentracion_manu_difusion = 0.87e-6

S2_manu_asoc_disoc = 4.337e-7
concentracion_manu_asoc_disoc = 1.76e-6


plt.figure()
#fig.suptitle('S2 vs # de molec en PSF')
#fig, axs = plt.subplots(2)
#plt.semilogx(concentracion_molar_en_PSF,S2_difusion, 'b-*', label='S2 difusion')
plt.loglog(concentracion_molar_en_PSF_dif,S2_difusion, 'b--s', label='S2 difusion',linewidth=3,markersize=10)
plt.loglog(concentracion_molar_en_PSF_asoc_disoc,S2_asoc_disoc, 'r--s', label='S2 asoc-disoc',linewidth=3,markersize=10)
plt.loglog(concentracion_manu_difusion,S2_manu_difusion,'*g',markersize=10, label='S2 manu difu')
plt.loglog(concentracion_manu_asoc_disoc,S2_manu_asoc_disoc,'*',color='orange',markersize=10,label='S2 manu asoc-disoc')

plt.xlabel('Concentracion en PSF nM')
plt.ylabel('S2')
plt.tick_params(which='minor', length=6, width=2.5)
plt.tick_params(which='major', length=8, width=3.5)
figManager = plt.get_current_fig_manager()  ####   esto y la linea de abajo me maximiza la ventana de la figura
figManager.window.showMaximized()           ####   esto y la linea de arriba me maximiza la ventana de la figura
    
plt.legend()
plt.show()
