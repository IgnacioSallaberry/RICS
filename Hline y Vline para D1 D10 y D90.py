# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:09:45 2019

@author: Ignacio Sallaberry
"""

import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import norm

#from scipy.optimize import curve_fit, minimize
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
import lmfit 
from scipy.optimize import minimize
from scipy import interpolate

#==============================================================================
#                                Tipografía de los gráficos
#==============================================================================    
SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

tamaño_de_punto = 10

plt.close('all') # antes de graficar, cierro todos las figuras que estén abiertas

#==============================================================================
#                 Datos originales:  Filtro el txt
#==============================================================================    

with open('C:\\Users\\ETcasa\\Desktop\\D1.txt') as fobj:
    DATA= fobj.read()
D_1= re.split('\t|\n', DATA)
with open('C:\\Users\\ETcasa\\Desktop\\D10.txt') as fobj:
    DATA= fobj.read()
D_10= re.split('\t|\n', DATA)
with open('C:\\Users\\ETcasa\\Desktop\\D90.txt') as fobj:
    DATA= fobj.read()
D_90= re.split('\t|\n', DATA)


D1=[]
D10=[]
D90=[]
d=0    
while d < len(D_1):
    D1.append(np.float(D_1[d])/np.float(D_1[0]))
    D10.append(np.float(D_10[d])/np.float(D_10[0]))
    D90.append(np.float(D_90[d])/np.float(D_90[0]))
    d+=1
#==============================================================================
#                 Grafico     la    H - Line
#==============================================================================    
dr=0.05

x = np.arange(1, len(D1)+1)

plt.figure()
plt.semilogx(dr*x, D1, '-*', label='D=1', markersize=tamaño_de_punto )
plt.semilogx(dr*x, D10, '-*', label='D=10', markersize=tamaño_de_punto )
plt.semilogx(dr*x, D90, '-*', label='D=90', markersize=tamaño_de_punto )


plt.xlabel(r'pixel shift $\xi$ - $\mu m$')
plt.ylabel(r'G($\xi$)')
#    plt.title('H-line  SCANNING  \n tp = {}$\mu$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e6,dr,box_size),fontsize=18)
plt.title('Linea Horizontal')
plt.legend()
plt.show()
plt.tight_layout() #hace que no me corte los márgenes
        
    
plt.savefig('C:\\Users\\ETcasa\\Desktop\\H-Line_D1_D10_D90')



#==============================================================================
#                 Datos originales:  Filtro el txt
#==============================================================================    

with open('C:\\Users\\ETcasa\\Desktop\\D1Vline.txt') as fobj:
    DATA= fobj.read()
D_1= re.split('\t|\n', DATA)
with open('C:\\Users\\ETcasa\\Desktop\\D10Vline.txt') as fobj:
    DATA= fobj.read()
D_10= re.split('\t|\n', DATA)
with open('C:\\Users\\ETcasa\\Desktop\\D90Vline.txt') as fobj:
    DATA= fobj.read()
D_90= re.split('\t|\n', DATA)


D1=[]
D10=[]
D90=[]
d=0    
while d < len(D_1):
    D1.append(np.float(D_1[d])/np.float(D_1[0]))
    D10.append(np.float(D_10[d])/np.float(D_10[0]))
    D90.append(np.float(D_90[d])/np.float(D_90[0]))
    d+=1
#==============================================================================
#                 Grafico     la    V - Line
#==============================================================================    
dr=0.05

x = np.arange(1, len(D1)+1)

plt.figure()
plt.semilogx(dr*x, D1, '-*', label='D=1', markersize=tamaño_de_punto )
plt.semilogx(dr*x, D10, '-*', label='D=10', markersize=tamaño_de_punto )
plt.semilogx(dr*x, D90, '-*', label='D=90', markersize=tamaño_de_punto )


plt.xlabel(r'pixel shift $\xi$ - $\mu m$')
plt.ylabel(r'G($\xi$)')
#    plt.title('H-line  SCANNING  \n tp = {}$\mu$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e6,dr,box_size),fontsize=18)
plt.title('Linea Vertical')
plt.legend()
plt.show()
plt.tight_layout() #hace que no me corte los márgenes
        
plt.savefig('C:\\Users\\ETcasa\\Desktop\\V-Line_D1_D10_D90')
   