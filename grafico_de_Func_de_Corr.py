# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:09:45 2019

@author: Ignacio Sallaberry
"""

import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import norm

from scipy.optimize import curve_fit
import matplotlib.mlab as mlab

from lmfit import Parameters


#==============================================================================
#                                Tipografía de los gráficos
#==============================================================================    
SMALL_SIZE = 10
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.close('all') # antes de graficar, cierro todos las figuras que estén abiertas

#==============================================================================
#                 Datos originales:  Filtro el txt
#==============================================================================    
j=1
lista_S_2_Diff = []
while j<2:
    with open('C:\\Users\\LEC\\Nacho\\5-4-19\\histograma\\ajuste\\sim{}-5-4-19-DATA.txt'.format(j)) as fobj:
        DATA= fobj.read()
    j+=1 
Funcion_de_correlacion= re.split('\t|\n', DATA)
Funcion_de_correlacion.remove('X')
Funcion_de_correlacion.remove('Y')
Funcion_de_correlacion.remove('Z')
Funcion_de_correlacion.remove('')
Funcion_de_correlacion.remove('')



#==============================================================================
#                 Me armo tres listas: una con Seda, otra con Psi y otra con G
#==============================================================================
i=0
seda=[]
psi=[]
G=[]
    
while i< len(Funcion_de_correlacion):
    seda.append(np.float(Funcion_de_correlacion[i]))
    psi.append(float(Funcion_de_correlacion[i+2]))
    G.append(float(Funcion_de_correlacion[i+1]))
    
    i+=3


##==============================================================================
##                 Grafico la funcion de correlación en 2D
##==============================================================================        
plt.figure()
ACF = np.asarray(G)  #para graficar debo pasar de list a un array
plt.imshow(ACF.reshape(63,63))  
plt.show()
#### nota: que hace Reshape? lista.reshape()   toma la lista y, sin cambiar los valores, lo va cortando hasta acomodarlo en una matriz de nxm. Ojo que nxm debe ser = al len(lista)



##==============================================================================
##                 Filtro los valores que me interesan de G: en este caso es la linea horizontal
##==============================================================================    

ROI = max(seda)-min(seda)
i=min(seda)+ROI/2 #veo cual es el medio de mi ROI

##saco todas las lineas menores a, en este caso 128 
j=0
while psi[j] < i:
    G.remove(G[0])   #saco todas las lineas menores a, en este caso 128 
    G.remove(G[-1])
    j+=1

j=0
seda1=[]
while j<len(G):
    if G[0]<max(G):
        G.remove(G[0])
    seda1.append(j)
    j+=1

#gragico la linea Horizontal
plt.figure()
plt.plot(G, 'b*', label='ACF')
plt.show()

##==============================================================================
##                 AJUSTE DE LA LINEA HORIZONTAL
##==============================================================================    


#                                  Parametros globales iniciales
box_size = 256
roi = 128
tp = 5e-6             #seg    
tl = box_size * tp    #seg
dr = 0.05             # delta r = pixel size =  micrometros
w0 = 0.25             #radio de la PSF  = micrometros    
wz = 1.5              #alto de la PSF desde el centro  = micrometros
a = w0/wz

gamma = 0.3536        #gamma factor de la 3DG
N = 200            #Numero total de particulas en la PSF



#                                  Parametros globales iniciales
def Scanning (x,y,dr,w0,tp,tl):
    
    return np.exp(-0.5*((2*x*dr/w0)**2+(2*y*dr/w0)**2)/(1 + 4*D*(tp*x+tl*y)/(w0**2)))


def Difusion (x,D,N):
    box_size = 256
    roi = 128
    tp = 5e-6             #seg    
    tl = box_size * tp    #seg
    dr = 0.05             # delta r = pixel size =  micrometros
    w0 = 0.25             #radio de la PSF  = micrometros    
    wz = 1.5              #alto de la PSF desde el centro  = micrometros
    a = w0/wz
    
    gamma = 0.3536        #gamma factor de la 3DG
   
    
    y=0
    
    return (gamma/N)*( 1 + 4*D*(tp*x+tl*y) / w0**2 )**(-1) * ( 1 + 4*D*(tp*x+tl*y) / wz**2 )**(-1/2)   * np.exp(-0.5*((2*x*dr/w0)**2+(2*y*dr/w0)**2)/(1 + 4*D*(tp*x+tl*y)/(w0**2)))



#def Difusion (x,y,gamma,N,w0,wz,tp,tl):
#
#    return (gamma/N)*( 1 + 4*d*(tp*x+tl*y) / w0**2 )**(-1) * ( 1 + 4*D*(tp*x+tl*y) / wz**2 )**(-1/2)



#def Triplete (x,y,tp,tl,At,t_triplet):
#    
#    return 1+ At * np.exp(-(tp*x+tl*y)/t_triplet)
#    
#
#def Binding(x,y,tp,tl,Ab,t_binding):
#    
#    return Ab * np.exp(-(x*dr/w0)**2-(y*dr/w0))* np.exp(-(tp*x+tl*y)/t_binding)


#x = np.arange(128, 160)
x = np.arange(1, 33)
y = G

popt, pcov = curve_fit(Difusion, x, y, p0=(9.5,0.017))
#
plt.plot(x, Difusion(x,popt[0],popt[1]), 'r-', label='Ajuste')

plt.plot(x, Difusion(x,10,0.017), 'g-', label='Dibujada' )

plt.xlabel(r'pixel shift $\xi$ - $\mu m$',fontsize=14)
plt.ylabel(r'G($\xi$)',fontsize=14)
#    plt.title('H-line  SCANNING  \n tp = {}$\mu$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e6,dr,box_size),fontsize=18)
plt.title('H-line')
plt.legend()
plt.show()
plt.tight_layout() #hace que no me corte los márgenes










##==============================================================================
##                 AJUSTE DE LA LINEA HORIZONTAL        VERSION   LMFIT
##==============================================================================    
params = Parameters()
params.add('boxsize', value=256, vary=False)
params.add('roi', value=128, vary=False)
params.add('tp', value=5e-6, vary=False)
params.add('dr', value=0.05, vary=False)
params.add('w0', value=0.25, vary=False)
params.add('dz', value=0.05, vary=False)


def Difusion_lmifit(params):
    box_size = params['boxsize']
    roi = params['roi']
    tp = params['tp']             #seg    
    tl = box_size * tp            #seg
    dr = params['dr']             # delta r = pixel size =  micrometros
    w0 = params['w0']             #radio de la PSF  = micrometros    
    wz = params['wz']             #alto de la PSF desde el centro  = micrometros
    a = w0/wz

    return (gamma/N)*( 1 + 4*d*(tp*x+tl*y) / w0**2 )**(-1) * ( 1 + 4*d*(tp*x+tl*y) / wz**2 )**(-1/2)






###---> np.meshgrid() #Return coordinate matrices from coordinate vectors.
#x = np.arange(64)
#y = x
#
#
#
#xx, yy = np.meshgrid(x, y, sparse=True)
#z = G
#
#plt.pcolor(x, y, z)
#plt.colorbar()
#plt.title("ACF")
#plt.show()    


#x, y = np.meshgrid(seda1, psi1)
#
#plt.figure()
#plt.imshow(G)
#plt.colorbar()