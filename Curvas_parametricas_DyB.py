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
tp = 5e-6             #seg    
tl = box_size * tp    #seg
dr = 0.05             # delta r = pixel size =  micrometros
w0 = 0.25             #radio de la PSF  = micrometros    
wz = 1.5              #alto de la PSF desde el centro  = micrometros

gamma = 0.3536        #gamma factor de la 3DG
N = 0.0176            #Numero total de particulas en la PSF

At= 1                 #Amplitud de triplete
t_triplet = .16       #segundos    #Tiempo caracteristico de triplete 

Ab= 1                 #Amplitud de binding
t_binding= .16        #segundos     #Tiempo caracteristico de binding



D=[0.1,1,10,100,1000]
#D=np.arange(1,1050,200)   #micrones^2/seg
x0 = (box_size)/2  #seda inicial
xf = (box_size + roi)/2  #seda final
#xf=640
x=np.arange(1,xf-x0)

y = 0  #miro solo la componente H orizontal


###------------------------------------------------GRAFICO EN 2D
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


fig = plt.figure()    
plt.close('all') # amtes de graficar, cierro todos las figuras que estén abiertas
    
guardar_imagenes = False    
for d in D:
#    if d==D[-1]:
#        guardar_imagenes = True
#    else:
#        guardar_imagenes = False
#    
#==============================================================================
#                                  H - Line
#==============================================================================    

###termino de scanning
    S = np.exp(-0.5*((2*x*dr/w0)**2+(2*y*dr/w0)**2)/(1 + 4*d*(tp*x+tl*y)/(w0**2)))
###termino de correlación espacial
    G_diff = (gamma/N)*( 1 + 4*d*(tp*x+tl*y) / w0**2 )**(-1) * ( 1 + 4*d*(tp*x+tl*y) / wz**2 )**(-1/2)
###termino de triplete
    G_triplet = 1+ At * np.exp(-(tp*x+tl*y)/t_triplet)
###termino de binding
    G_bind =Ab * np.exp(-(x*dr/w0)**2-(y*dr/w0))* np.exp(-(tp*x+tl*y)/t_binding)

###Total ACF    
    Gtotal = S * G_diff * G_triplet + G_bind
    Gtotal_normalizada = Gtotal/max(Gtotal)

###Normalización de factores de la ACF    
    S_norm = S/max(S)
    G_diff_norm = G_diff/max(G_diff)
    G_triplet_norm = G_triplet/max(G_triplet)
    G_bind_norm = G_bind/max(G_bind)    
    
    ###  Grafico solo termino difussivo
    plt.figure(1)
#    plt.plot((dr*x), G_norm,'-.',label=f'D={d}',linewidth=1)
    plt.semilogx((dr*x), G_diff_norm,'-.',label=f'D={d}')
    plt.legend()
    plt.xlabel(r'pixel shift $\xi$ - $\mu m$',fontsize=14)
    plt.ylabel(r'Gdiff($\xi$)',fontsize=14)
#    plt.title('H-line G diff  \n tp = {}$\mu$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e6,dr,box_size),fontsize=18)
    plt.title('H-line G diff')
    plt.show()
    plt.tight_layout() #hace que no me corte los márgenes
    
    if guardar_imagenes:
        plt.savefig('C:\\Users\\LEC\\Desktop\\Poster TOPFOT 2019\\H_Line_Diff')
    
    ###  Grafico solo termino Scanning
    plt.figure(2)
#    plt.plot((dr*x), S_norm,'-.',label=f'D={d}',linewidth=1)
    plt.semilogx((dr*x), S_norm,'-.',label=f'D={d}')
    plt.legend()
    plt.xlabel(r'pixel shift $\xi$ - $\mu m$',fontsize=14)
    plt.ylabel(r'Scanning($\xi$)',fontsize=14)
#    plt.title('H-line  SCANNING  \n tp = {}$\mu$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e6,dr,box_size),fontsize=18)
    plt.title('H-line  SCANNING')
    plt.show()
    plt.tight_layout() #hace que no me corte los márgenes
    
    if guardar_imagenes:
        plt.savefig('C:\\Users\\LEC\\Desktop\\Poster TOPFOT 2019\\H_Line_Scanning')
        
    ###  Grafico funcion de correlación  total
    plt.figure(3)
#    plt.plot((dr*x), Gtotal_normalizada,'-.',label=f'D={d}',linewidth=1)
    plt.semilogx((dr*x), Gtotal_normalizada,'-.',label=f'D={d}')
    plt.legend()
    plt.xlabel(r'pixel shift $\xi$ - $\mu m$',fontsize=14)
    plt.ylabel(r'Gtot($\xi$)',fontsize=14)
    plt.title('H-line G total \n tp = {}$\mu$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e6,dr,box_size),fontsize=18)
    plt.title('H-line  G total')
    plt.show()
    plt.tight_layout() #hace que no me corte los márgenes
    
    if guardar_imagenes:
        plt.savefig('C:\\Users\\LEC\\Desktop\\Poster TOPFOT 2019\\H_Line_Gtot_norm')
    
#    i=0
#    while Gtotal_normalizada[i]>0.60:
#        i+=1
#    plt.semilogx(dr*x[i],Gtotal_normalizada[i], '*', label=f'dist para Gtotal/2 = {round(dr*x[i],5)}$\mu m$')      ###---> para graficar punto donde cae a la mitad la curva de correlación

    
#==============================================================================
#                                  V - Line
#==============================================================================    

####  si quiero mirar la V line  hago tp=tl, esto puedo hacerlo que poner "x=0" y activar "y=np.arange(1,xf-x0)" es lo mismo que dejar x=0 y poner tp=tl
    
    tp=tl
###termino de scanning
    S = np.exp(-0.5*((2*x*dr/w0)**2+(2*y*dr/w0)**2)/(1 + 4*d*(tp*x+tl*y)/(w0**2)))
###termino de correlación espacial
    G_diff = (gamma/N)*( 1 + 4*d*(tp*x+tl*y) / w0**2 )**(-1) * ( 1 + 4*d*(tp*x+tl*y) / wz**2 )**(-1/2)


    Gtotal = S * G_diff
    Gtotal_normalizada = Gtotal/max(Gtotal)
    S_norm = S/max(S)
    G_diff_norm = G_diff/max(G_diff)

    
    ###  Grafico solo termino difussivo
    plt.figure(4)
#    plt.plot((dr*x), G_norm,'-.',label=f'D={d}')
    plt.semilogx((dr*x), G_diff_norm,'-.',label=f'D={d}')
    plt.legend()
    plt.xlabel(r'pixel shift $\psi$ - $\mu m$',fontsize=14)
    plt.ylabel(r'Gdiff($\psi$)',fontsize=14)
#    plt.title('V-line G diff  \n tl = {}$ms$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e3,dr,box_size),fontsize=18)
    plt.title('V-line G diff',fontsize=18)
    plt.show()
    plt.tight_layout() #hace que no me corte los márgenes
    
    if guardar_imagenes:
        plt.savefig('C:\\Users\\LEC\\Desktop\\Poster TOPFOT 2019\\V_Line_Diff')
    
    
    ###  Grafico solo termino Scanning
    plt.figure(5)
#    plt.plot((dr*x), S_norm,'-.',label=f'D={d}')
    plt.semilogx((dr*x), S_norm,'-.',label=f'D={d}')
    plt.legend()
    plt.xlabel(r'pixel shift $\psi$ - $\mu m$',fontsize=14)
    plt.ylabel(r'Scanning($\psi$)',fontsize=14)
#    plt.title('V-line  SCANNING  \n tl = {}$ms$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e3,dr,box_size),fontsize=18)
    plt.title('V-line  SCANNING')
    plt.show()
    plt.tight_layout() #hace que no me corte los márgenes
    
    if guardar_imagenes:
        plt.savefig('C:\\Users\\LEC\\Desktop\\Poster TOPFOT 2019\\V_Line_Scanning')
    
    
    ###  Grafico funcion de correlación  total
    plt.figure(6)
#    plt.plot((dr*x), Gtotal_normalizada,'-.',label=f'D={d}')
    plt.semilogx((dr*x), Gtotal_normalizada,'-.',label=f'D={d}')
    plt.legend()
    plt.xlabel(r'pixel shift $\psi$ - $\mu m$',fontsize=14)
    plt.ylabel(r'Gtot($\psi$)',fontsize=14)
#    plt.title('V-line G total \n tl = {}$ms$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e3,dr,box_size),fontsize=18)
    plt.title('V-line G total')
    plt.show()
    plt.tight_layout() #hace que no me corte los márgenes
    
    if guardar_imagenes:
        plt.savefig('C:\\Users\\LEC\\Desktop\\Poster TOPFOT 2019\\V_Line_Gtot_norm')

    

guardar_imagenes = False
    
















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
