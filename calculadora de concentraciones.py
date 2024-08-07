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
box_size = 128        # OBS: El valor que se introduce en el simFCS es el semiancho de la caja.
                      # Acá "box_size"= ancho total de la caja. Por lo tanto, usar el doble del valor introducido en el simFCS
roi = 128
tp = 1e-6             #seg    
tl = box_size * tp    #seg
dr = 0.05             # delta r = pixel size =  micrometros
w0 = 0.15              # radio de la PSF  = micrometros    
                       # OBS!! : El simFCS indica "w0 = Radial Waist". # Esto significa que es el diámetro de la PSF.
                       # Por lo tanto acá utilizar w0 = (w0_simfcs)/2.
w0_simFCS = 2*w0

wz = 1.8              #alto de la PSF desde el centro  = micrometros

partic_en_caja = 100

#%%
#==============================================================================
#                                  Valores informados por el simFCS
#==============================================================================   
vol_caja_simFCS = (dr*box_size)**2 # OK  En micrones^2
vol_PSF_2Dimensiones_simFCS = np.pi*(w0_simFCS**2)  # OK esto es en 2D

## El problema de esto es que si se utiliza este volumen, la concentración de mol[eculas en la PSF no dá el mismo resultado que el reportado por el simFCS
partic_en_PSF_simFCS = vol_PSF_2Dimensiones_simFCS * partic_en_caja / vol_caja_simFCS    #en femtolitros

concentracion_en_PSF_utilizando_el_vol_reportado_por_el_simFCS = partic_en_PSF_simFCS/vol_PSF_2Dimensiones_simFCS # cantidad de moleculas en PSF = #de molec por um^2

## Se cancela el Volumen de la PSF en 2D. 
## Entonces la concentración de la PSF en 2D termina siendo directamente el numero de moléculas en la caja / el vol_caja en 2D que sería la superficie de la caja


#%%
#==============================================================================
#                                  Para simulaciones en 2-Dimensiones
#==============================================================================   
vol_caja = (dr*box_size)**3 # OK  En micrones^2

wz_2D = 1

vol_PSF_2Dimensiones = np.pi*(w0**2)  # OK esto es en 2D

partic_en_PSF = vol_PSF_2Dimensiones * partic_en_caja / vol_caja    #en femtolitros

concentracion_en_PSF = partic_en_PSF/vol_PSF_2Dimensiones # cantidad de moleculas en PSF = #de molec por um^2
## Se cancela el Volumen de la PSF en 2D. 
## Entonces la concentración de la PSF en 2D termina siendo directamente el numero de moléculas en la caja / el vol_caja en 2D que sería la superficie de la caja
## Por eso no importa que el simFCS utilice el ancho de la PSF y no el radio en este caso.

un_mol=6.023e23  ## moleculas en un mol

concentracion_molar_en_PSF = concentracion_en_PSF/un_mol * 1e15   ## Esta no la informa el simFCS

#%%
#==============================================================================
#                                  Para simulaciones en 3-Dimensiones
#==============================================================================
vol_caja_3D = (dr*box_size)**3

vol_PSF_3D = ((w0)**2)*wz*((np.pi)**(1.5))*1.4134614062867903  ## No sé de donde sale el factor 1.413... pero sino no da el mismo resultado que en el simFCS


partic_en_PSF = vol_PSF * partic_en_caja / vol_caja_3D    #en femtolitros

concentracion_en_PSF = partic_en_PSF/vol_PSF * 1e15   # cantidad de moleculas en PSF/ litro


un_mol=6.023e23  ## moleculas en un mol

concentracion_molar_en_PSF = concentracion_en_PSF/un_mol
