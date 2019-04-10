# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:25:06 2019

@author: ETCasa
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, ticker
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import re
import mpl_toolkits


box_size = 256
roi = 64
tp = 5e-6    #seg
tl = box_size * tp    #seg
dr = 0.05   # delta r = pixel size =  micrometros
w0 = 0.25   #radio de la PSF  = micrometros
wz = 1.5   #alto de la PSF desde el centro  = micrometros
gamma = 0.3536   #gamma factor de la 3DG
N = 0.0176   #Numero total de particulas en la PSF


D=np.arange(0,120,25)   #micrones^2/seg
x0 = (box_size)/2  #seda inicial
xf = (box_size + roi)/2  #seda final
x=np.arange(0,xf-x0)

y = 0  #miro solo la componente H orizontal
###------------------------------------------------GRAFICO EN 2D
###termino de scanning
fig = plt.figure()
#ax = plt.axes(projection='3d')

    

for d in D:
    S = np.exp(-0.5*((2*x*dr/w0)**2+(2*y*dr/w0)**2)/(1 + 4*d*(tp*x+tl*y)/(w0**2)))
###termino de correlación espacial
    G = (gamma/N)*( 1 + 4*d*(tp*x+tl*y) / w0**2 )**(-1) * ( 1 + 4*d*(tp*x+tl*y) / wz**2 )**(-1/2)


    Gtotal = S * G
    Gtotal_normalizada = Gtotal/max(Gtotal)


    plt.semilogx((dr*x), Gtotal_normalizada,'-',label=f'D={d}')
    
    plt.show()
plt.legend()
plt.xlabel(r'pixel shift x - micrones - pix size = {} $\mu$- box size = {} $\mu$ m'.format(dr,box_size),fontsize=14)
plt.ylabel('G(x)',fontsize=14)
plt.title(' H-line'.format(box_size),fontsize=18)





##
###----------------------------------------------------------------GRAFICO EN 3D
####termino de scanning
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#
#    
#
#for d in D:
#    S = np.exp(-0.5*((2*x*dr/w0)**2+(2*y*dr/w0)**2)/(1 + 4*d*(tp*x+tl*y)/(w0**2)))
####termino de correlación espacial
#    G = (gamma/N)*( 1 + 4*d*(tp*x+tl*y) / w0**2 )**(-1) * ( 1 + 4*d*(tp*x+tl*y) / wz**2 )**(-1/2)
#
#
#    Gtotal = S * G
#    Gtotal_normalizada = Gtotal/max(Gtotal)
#
#
##--------------------------grafico  2D  de Gtot vs seda para distintos D
#
#    ax.plot(d*np.ones_like(x), (x), Gtotal_normalizada)
#
#    plt.show()
#    
#
#ax.set_xlabel('D')
#ax.set_ylabel('x')
#ax.set_zlabel('G(x))')
#
#
