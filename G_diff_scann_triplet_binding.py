# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:25:06 2019

@author: ETCasa
"""

import numpy as np
import matplotlib.pyplot as plt



box_size = 256
roi = 128
tp = 5e-6    #seg    
tl = box_size * tp    #seg
dr = 0.025   # delta r = pixel size =  micrometros
w0 = 0.25   #radio de la PSF  = micrometros    
wz = 1.5   #alto de la PSF desde el centro  = micrometros
gamma = 0.3536   #gamma factor de la 3DG
N = 0.0176   #Numero total de particulas en la PSF

#Parametros de Gtriplete
A = 0.5 #s a constant that depends on the fraction of molecules blinking and on the difference in fluorescence intensity between the two states (which is 1 for pure blinking)
tau =[1e-6,1e-5,1e-4,1e-3,0.16]  #contains both the time the fluorescence is on and off      unit = seg


D=[0]
#D=np.arange(1,1050,200)   #micrones^2/seg
x0 = (box_size)/2  #seda inicial
xf = (box_size + roi)/2  #seda final
x=np.arange(1,xf-x0)



##########            H - Line         #######################33

y = 0  
fig = plt.figure()

t_H_Line = [1e-6,1e-5,1e-4,1e-3,0.16]  #contains both the time the fluorescence is on and off      unit = seg

    
D=0
d=D
for t in tau:
    
#    tp = 5e-6    #seg    
    tl = box_size * tp    #seg
    
###termino de correlación espacial
    S = np.exp(-0.5*((2*x*dr/w0)**2+(2*y*dr/w0)**2)/(1 + 4*d*(tp*x+tl*y)/(w0**2)))

###termino de correlación difusivo
    G = (gamma/N)*( 1 + 4*d*(tp*x+tl*y) / w0**2 )**(-1) * ( 1 + 4*d*(tp*x+tl*y) / wz**2 )**(-1/2)

####termino de correlación blinking o triplete
#    Gt = 1 + A * np.exp(-(tp*x+tl*y)/t)
    
###termino de correlación blinking o triplete
    GB = A * np.exp(-((x*dr/w0)**2+(y*dr/w0)**2)-(tp*x+tl*y)/t)


    Gtotal = S * G + GB #NO estoy viendo Gt
    Gtotal_normalizada = Gtotal/max(Gtotal)  #G total normalizada
    S_norm = S/max(S)  #scanning normalizado
    G_norm = G/max(G)  #difusivo normalizado
#    Gt_norm = Gt/max(Gt)  #triplete normalizado    
    GB_norm = GB/max(GB)  #Binding normalizado


#    ###  Grafico solo termino binding
#    plt.figure(1)
#    plt.semilogx((dr*x), GB_norm,'-.',label=r'$\tau$={} seg'.format(t))
#    plt.legend()
#    plt.xlabel(r'pixel shift $\xi$ - $\mu m$',fontsize=14)
#    plt.ylabel(r'GBind($\xi$)',fontsize=14)
#    plt.title('H-line G Bind\n tp = {}$\mu$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e6,dr,box_size),fontsize=18)
#    plt.show()


#    ###  Grafico solo termino triplete
#    plt.figure(2)
#    plt.semilogx((dr*x), Gt_norm,'-.',label=f'D={d}')
#    plt.legend()
#    plt.xlabel(r'pixel shift $\xi$ - $\mu m$',fontsize=14)
#    plt.ylabel(r'Gtrip($\xi$)',fontsize=14)
#    plt.title('H-line G trip \n tp = {}$\mu$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e6,dr,box_size),fontsize=18)
#    plt.show()
#    

    
#    ###  Grafico solo termino difussivo
#    plt.figure(3)
#    plt.semilogx((dr*x), G_norm,'-.',label=f'D={d}')
#    plt.legend()
#    plt.xlabel(r'pixel shift $\xi$ - $\mu m$',fontsize=14)
#    plt.ylabel(r'Gdiff($\xi$)',fontsize=14)
#    plt.title('H-line G diff  \n tp = {}$\mu$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e6,dr,box_size),fontsize=18)
#    plt.show()
#    
#    
#    ###  Grafico solo termino Scanning
#    plt.figure(4)
#    plt.semilogx((dr*x), S_norm,'-.',label=f'D={d}')
#    plt.legend()
#    plt.xlabel(r'pixel shift $\xi$ - $\mu m$',fontsize=14)
#    plt.ylabel(r'Scanning($\xi$)',fontsize=14)
#    plt.title('H-line  SCANNING  \n tp = {}$\mu$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e6,dr,box_size),fontsize=18)
#    plt.show()
#
    ###  Grafico funcion de correlación  total
#    plt.figure(5)
#    plt.semilogx((dr*x), Gtotal_normalizada,'-.',label=r'D={} $\tau$={} seg'.format(d,t))
#    plt.legend()
#    plt.xlabel(r'pixel shift $\xi$ - $\mu m$',fontsize=14)
#    plt.ylabel(r'Gtot($\xi$)',fontsize=14)
#    plt.title('H-line G total \n tp = {}$\mu$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e6,dr,box_size),fontsize=18)
#    plt.show()
#    
    
    
####################     V - line      #################################    
    
### -- #si quiero mirar la V line  hago tp=tl, esto puedo hacerlo que poner "x=0" y activar "y=np.arange(1,xf-x0)" es lo mismo que dejar y=0 y poner tp=tl
tp=tl

fig = plt.figure()
t_V_line = [1e-3,1e-2,0.16,1,60]

for t in t_V_line:

    S = np.exp(-0.5*((2*x*dr/w0)**2+(2*y*dr/w0)**2)/(1 + 4*d*(tp*x+tl*y)/(w0**2)))
###termino de correlación espacial
    G = (gamma/N)*( 1 + 4*d*(tp*x+tl*y) / w0**2 )**(-1) * ( 1 + 4*d*(tp*x+tl*y) / wz**2 )**(-1/2)
    
    
####termino de correlación blinking o triplete
#    Gt = 1 + A * np.exp(-(tp*x+tl*y)/t)
    
###termino de correlación blinking o triplete
    GB = A * np.exp(-((x*dr/w0)**2+(y*dr/w0)**2)-(tp*x+tl*y)/t)

    Gtotal = S * G + GB #NO estoy viendo Gt
 
    
    Gtotal_normalizada = Gtotal/max(Gtotal)  #G total normalizada
    S_norm = S/max(S)  #scanning normalizado
    G_norm = G/max(G)  #difusivo normalizado
#    Gt_norm = Gt/max(Gt)  #triplete normalizado    
    GB_norm = GB/max(GB)  #Binding normalizado

    ###  Grafico solo termino binding
    plt.figure(6)
    plt.semilogx((dr*x), GB_norm,'-.',label=r'$\tau$={} seg'.format(t))
    plt.legend()
    plt.xlabel(r'pixel shift $\xi$ - $\mu m$',fontsize=14)
    plt.ylabel(r'GBind($\xi$)',fontsize=14)
    plt.title('V-line G Bind\n tp = {}$\mu$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e6,dr,box_size),fontsize=18)
    plt.show()
#
#    
#    ###  Grafico solo termino triplete
#    plt.figure(7)
#    plt.semilogx((dr*x), GB_norm,'-.',label=f'\tau={t}')
#    plt.legend()
#    plt.xlabel(r'pixel shift $\xi$ - $\mu m$',fontsize=14)
#    plt.ylabel(r'GBind($\xi$)',fontsize=14)
#    plt.title('V-line G Bind\n  tl = {}$ms$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e3,dr,box_size),fontsize=18)
#    plt.show()
#
#
#    
#    ###  Grafico solo termino difussivo
#    plt.figure(8)
#    plt.semilogx((dr*x), G_norm,'-.',label=f'D={d}')
#    plt.legend()
#    plt.xlabel(r'pixel shift $\psi$ - $\mu m$',fontsize=14)
#    plt.ylabel(r'Gdiff($\psi$)',fontsize=14)
#    plt.title('V-line G diff  \n tl = {}$ms$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e3,dr,box_size),fontsize=18)
#    plt.show()
#    
#    
#    ###  Grafico solo termino Scanning
#    plt.figure(9)
#    plt.semilogx((dr*x), S_norm,'-.',label=f'D={d}')
#    plt.legend()
#    plt.xlabel(r'pixel shift $\psi$ - $\mu m$',fontsize=14)
#    plt.ylabel(r'Scanning($\psi$)',fontsize=14)
#    plt.title('V-line  SCANNING  \n tl = {}$ms$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e3,dr,box_size),fontsize=18)
#    plt.show()

    ###  Grafico funcion de correlación  total
    plt.figure(10)
    plt.semilogx((dr*x), Gtotal_normalizada,'-.',label=r'D={} $\tau$={} seg'.format(d,t))
    plt.legend()
    plt.xlabel(r'pixel shift $\psi$ - $\mu m$',fontsize=14)
    plt.ylabel(r'Gtot($\psi$)',fontsize=14)
    plt.title('V-line G total \n tl = {}$ms$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e3,dr,box_size),fontsize=18)
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
