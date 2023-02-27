# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:25:06 2019
@author: Ignacio Sallaberry
"""

import numpy as np
import matplotlib.pyplot as plt


#==============================================================================
#                                  Parametros globales iniciales
#==============================================================================   
box_size = 256
roi = 128
tp = 1e-6             #seg    
tl = box_size * tp    #seg
dr = 0.05             # delta r = pixel size =  micrometros
w0 = 0.75              #radio de la PSF  = micrometros    
wz = 1.5              #alto de la PSF desde el centro  = micrometros

#%%
#==============================================================================
#                                  Para simulaciones en 2-Dimensiones
#==============================================================================   
vol_caja = (dr*box_size)**2 # OK

vol_PSF = (w0**2)*wz*1.04719 # OK esto es el valor reportado por el simFCS. No se de donde sale el factor 1.04719

vol_PSF_2Dimensiones = (w0**2)  # OK esto es en 2D

partic_en_caja = 100

partic_en_PSF = vol_PSF_2Dimensiones * partic_en_caja / vol_caja    #en femtolitros

concentracion_en_PSF = partic_en_PSF/vol_PSF_2Dimensiones # cantidad de moleculas en PSF = #de molec por um^2
#concentración en PSF está OK

un_mol=6.023e23  ## moleculas en un mol

concentracion_molar_en_PSF = concentracion_en_PSF/un_mol

#%%
#==============================================================================
#                                  Para simulaciones en 3-Dimensiones
#==============================================================================
vol_caja = (dr*box_size)**3

vol_PSF = ((w0*0.5)**2)*wz*((np.pi)**(1.5))

partic_en_caja = 4000

partic_en_PSF = vol_PSF * partic_en_caja / vol_caja    #en femtolitros

concentracion_en_PSF = partic_en_PSF/vol_PSF * 1e15   # cantidad de moleculas en PSF/ litro


un_mol=6.023e23  ## moleculas en un mol

concentracion_molar_en_PSF = concentracion_en_PSF/un_mol
