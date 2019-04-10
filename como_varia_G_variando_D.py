# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:25:06 2019

@author: ETCasa
"""

import numpy as np
import matplotlib.pyplot as plt
import array


box_size = 256
roi = 64
tp = 5e-6    #seg
tl = box_size * tp    #seg
dr = 0.05   # delta r = pixel size =  micrometros
w0 = 0.25   #radio de la PSF  = micrometros
wz = 1.5   #alto de la PSF desde el centro  = micrometros
gamma = 0.3536   #gamma factor de la 3DG
N = 0.0176   #Numero total de particulas en la PSF


D=90   #micrones^2/seg
x0 = (box_size)/2  #seda inicial
xf = (box_size + roi)/2  #seda final
x=np.arange(0,xf-x0)

y = 0  #miro solo la componente H orizontal

###termino de scanning                                            
S = np.exp(-0.5*((2*x*dr/w0)**2+(2*y*dr/w0)**2)/(1 + 4*D*(tp*x+tl*y)/(w0**2)))
###termino de correlación espacial
G = (gamma/N)*( 1 + 4*D*(tp*x+tl*y) / w0**2 )**(-1) * ( 1 + 4*D*(tp*x+tl*y) / wz**2 )**(-1/2)


Gtotal = S * G
Gtotal_normalizada = Gtotal/max(Gtotal)
### Plot the surface.
plt.plot(np.log10(x),Gtotal_normalizada,'-*')
plt.show()







##RASTER Scanning term (x and y direction) for isotropic difussion            
#w0, dr, tp, tl = params
#fitting = lambda D: np.exp(-0.5*((2*x*dr/w0)**2+(2*y*dr/w0)**2)/(1 + 4*D*self.f*(tp*x+tl*y)/(w0**2)))
#
#
##Spacial Correlation Function for isotropic difussion            
#w0, dr, tp, tl, gamma = params
#fitting = lambda G0, D: gamma*G0*( 1 + 4*D*self.f*(tp*x+tl*y) / w0**2 )**(-1) * ( 1 + 4*D*self.f*(tp*x+tl*y) / wz**2 )**(-1/2)
#            # gamma = gamma factor = 0.35 for 3D gaussian or 0.076 for Gaussian Lorentzian
