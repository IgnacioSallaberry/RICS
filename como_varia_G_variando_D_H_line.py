# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:25:06 2019

@author: ETCasa
"""

import numpy as np
import matplotlib.pyplot as plt



box_size = 1024
roi = 1024
tp = 5e-6    #seg    
tl = box_size * tp    #seg
dr = 0.025   # delta r = pixel size =  micrometros
w0 = 0.25   #radio de la PSF  = micrometros    
wz = 1.5   #alto de la PSF desde el centro  = micrometros
gamma = 0.3536   #gamma factor de la 3DG
N = 0.0176   #Numero total de particulas en la PSF




D=np.arange(1,1050,200)   #micrones^2/seg
x0 = (box_size)/2  #seda inicial
xf = (box_size + roi)/2  #seda final
x=np.arange(1,xf-x0)

y = 0  #miro solo la componente H orizontal

### -- #si quiero mirar la V line  hago tp=tl, esto puedo hacerlo que poner "x=0" y activar "y=np.arange(1,xf-x0)" es lo mismo que dejar x=0 y poner tp=tl
#tp=tl


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
    S_norm = S/max(S)
    G_norm = G/max(G)
    
    
    ###  Grafico solo termino difussivo
    plt.figure(1)
    plt.semilogx((dr*x), G_norm,'-.',label=f'D={d}')
    plt.legend()
    plt.xlabel(r'pixel shift $\psi$ - $\mu m$',fontsize=14)
    plt.ylabel(r'Gdiff($\psi$)',fontsize=14)
    plt.title('H-line G diff  \n tl = {}$ms$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e3,dr,box_size),fontsize=18)
    plt.show()
    
    
    ###  Grafico solo termino Scanning
    plt.figure(2)
    plt.semilogx((dr*x), S_norm,'-.',label=f'D={d}')
    plt.legend()
    plt.xlabel(r'pixel shift $\psi$ - $\mu m$',fontsize=14)
    plt.ylabel(r'Scanning($\psi$)',fontsize=14)
    plt.title('H-line  SCANNING  \n tl = {}$ms$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e3,dr,box_size),fontsize=18)
    plt.show()

    ###  Grafico funcion de correlación  total
    plt.figure(3)
    plt.semilogx((dr*x), Gtotal_normalizada,'-.',label=f'D={d}')
    plt.legend()
    plt.xlabel(r'pixel shift $\psi$ - $\mu m$',fontsize=14)
    plt.ylabel(r'Gtot($\psi$)',fontsize=14)
    plt.title('H-line G total \n tl = {}$ms$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e3,dr,box_size),fontsize=18)

    
#    i=0
#    while Gtotal_normalizada[i]>0.60:
#        i+=1
#    plt.semilogx(dr*x[i],Gtotal_normalizada[i], '*', label=f'dist para Gtotal/2 = {round(dr*x[i],5)}$\mu m$')      ###---> para graficar punto donde cae a la mitad la curva de correlación
    plt.show()








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
