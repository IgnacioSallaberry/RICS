# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 11:10:12 2019

@author: LEC
"""

import matplotlib.pyplot as plt
import numpy as np
import re
from statistics import mode
import seaborn as sns
import pandas as pd

plt.close('all') 
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


"""
Created on Tue Jun  4 16:09:45 2019

@author: Ignacio Sallaberry
"""

import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import norm

from scipy.optimize import curve_fit, minimize
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
import lmfit 
from scipy import interpolate

#==============================================================================
#                                Tipografía de los gráficos
#==============================================================================    
mostrar_imagenes = False

SMALL_SIZE = 14
MEDIUM_SIZE = 17
BIGGER_SIZE = 28

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
guess_val = [90,0.05]  #inicializo valores con los que voy a ajustar
### que proceso vas a ver?
###elegir un nombre para el nombre del txt que se va a crear con la lista de S2




q=1
S_2_ajustado_por_python_Diff = []

#proceso = 'difusion pura'
#while q<101:
#    print('simulacion numero {}'.format(q))
#    if q==12 or q==47:
#        q+=1
#    else:
#        pass
#    with open('C:\\Users\\LEC\\Desktop\\S2\\S2 una especie y un proceso - Difusion PURA\\sim{}-5-4-19-DATA.txt'.format(q)) as fobj:

proceso = 'Difusion D=90'        
while q<21:
    print('simulacion numero {}'.format(q))
    with open('C:\\Users\\LEC\\Desktop\\S2 y D=90\\sim{}-DATA.txt'.format(q)) as fobj:
        DATA= fobj.read()
        
    Funcion_de_correlacion= re.split('\t|\n', DATA)
    Funcion_de_correlacion.remove('X')
    Funcion_de_correlacion.remove('Y')
    Funcion_de_correlacion.remove('Z')
    Funcion_de_correlacion.remove('')
    Funcion_de_correlacion.remove('')
    
    

    ##==============================================================================
    ##                 AJUSTE DE SUPERFICIE 3D
    ##==============================================================================  
    i=0
    seda=[]
    psi=[]
    G=[]
        
    while i< len(Funcion_de_correlacion):
        seda.append(np.float(Funcion_de_correlacion[i]))
        psi.append(float(Funcion_de_correlacion[i+2]))
        G.append(float(Funcion_de_correlacion[i+1]))
        
        i+=3
    
    
    def Difusion (X_DATA, D, N):
        box_size = 256
        roi = 128
        tp = 5e-6             #seg    
        tl = box_size * tp    #seg
        dr = 0.05             # delta r = pixel size =  micrometros
        w0 = 0.25             #radio de la PSF  = micrometros    
        wz = 1.5              #alto de la PSF desde el centro  = micrometros
        a = w0/wz
        
        gamma = 0.3536        #gamma factor de la 3DG
        
        x = np.abs(np.asarray([N[0] for N in X_DATA]))
        y = np.abs(np.asarray([N[1] for N in X_DATA]))
    #    x = np.asarray(X_DATA[0])
    #    y = np.asarray(X_DATA[1])
    #    
        
        Dif = (gamma/N)*( 1 + 4*D*(tp*x+tl*y) / w0**2 )**(-1) * ( 1 + 4*D*(tp*x+tl*y) / wz**2 )**(-1/2)
        Scan = np.exp(-((x*dr/w0)**2+(y*dr/w0)**2)/(1 + 4*D*(tp*x+tl*y)/(w0**2)))
        
        return (Dif * Scan)
    
    
    
    ##==============================================================================
    ##                 CREO MATRIZ de tuplas (i,j)
    ##============================================================================== 
    # ahora quiero hacer lo mismo pero que los valores vayan entre -31 y 31
    # ie: el ancho es 62, que es lo mismo que el roi que tengo, que sale de hacer 159-97 
    A=[]
    C=[]
    roi = max(seda)-min(seda)
    mitad_del_roi = roi/2
    seda_min = (-1)*mitad_del_roi
    seda_max = mitad_del_roi
    
    k=seda_min    #K va a recorrer las filas
    
    while k < mitad_del_roi+1:
        j = 0
        
        while j<roi+1:
            A.append((seda_min+j,k))  #parado en una fila, recorro las columnas
            j+=1
        
        #me guardo la fila completa.
        
        k+=1
    
    X_DATA = np.vstack((A))
    
    X = [N[0] for N in X_DATA]
    Y = [N[1] for N in X_DATA]
    
    
    
    fit_params, cov_mat = curve_fit(Difusion,X_DATA, G, p0=(guess_val[0],guess_val[1]))
    
    
    
    
    def chi2(X_DATA,D,N):
        return np.sum((Difusion(X_DATA, D, N) - G)**2)/3969
    
    
#    print(chi2(X_DATA,fit_params[0],fit_params[1]))   ## si quiero ver el valor del chi2 ajustado
    
    S_2_ajustado_por_python_Diff.append(chi2(X_DATA,fit_params[0],fit_params[1]))
    
    q+=1
    ##### para guardar los datos

with open('S2 calculado de ajustar con python __ Ajuste de proceso de {} por modelo de DIFUSION.txt'.format(proceso),'w') as f:
    for valor in S_2_ajustado_por_python_Diff:
        ##como el archivo no existe python lo va a crear
        ##si el archivo ya existiese entonces python lo reescribirá
        
        ##entonces: python crea(o reescribe) el archivo
        ##          Lo abre
        ##          Lo escribe
        f.write(str(valor))
        f.write('\n') #le decimos que luego de escribir un elemento de la lista oceanos, que empiece una nueva linea











#==============================================================================
#
#                               DATOS     DIFUSION
#
#==============================================================================  
with open('C:\\Users\\LEC\\Desktop\\S2\\S2 calculado de ajustar con python __ Ajuste de proceso de difusion pura por modelo de DIFUSION.txt') as fobj:
    Proceso_dif_ajustado_por_dif = fobj.read()
S2_D_10 = re.split('\n', Proceso_dif_ajustado_por_dif)
S2_D_10.remove('')
S2_D_10 = [float(i) for i in S2_D_10]

while len(S2_D_10)>20:
    S2_D_10.pop()

with open('C:\\Users\\LEC\\Desktop\\S2 y D=90\\S2 calculado de ajustar con python __ Ajuste de proceso de Difusion D=90 por modelo de DIFUSION.txt') as fobj:
    S2_D_90= fobj.read()
S2_D_90 = re.split('\n', S2_D_90)
S2_D_90.remove('')
S2_D_90 = [float(i) for i in S2_D_90]


f, axes = f, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'wspace':0.35})
colores_difusion=sns.color_palette("Blues")
sns.set_style("white")

sns.stripplot(data=S2_D_10, ax=ax1, color='mediumaquamarine', orient='h',size=10)
sns.boxplot(data=S2_D_10, ax=ax1, color='forestgreen', orient='h')

sns.stripplot(data=S2_D_90, ax=ax2, color='steelblue', orient='h',size=10)
sns.boxplot(data=S2_D_90, ax=ax2, color='b', orient='h')



ax1.set_xlim([min(S2_D_10)*0.9,max(S2_D_10)*1.01])
ax2.set_xlim([min(S2_D_90)*0.9,max(S2_D_90)*1.01])
ax1.set_ylabel('D=10')
ax2.set_ylabel('D=90')
#ax1.set_ylabel('3.18')
#ax2.set_xlabel('3.18')



#ax1.set_title('Ajuste: Difusion')
#ax2.set_title('Ajuste: Binding')

#ax1.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=f.transFigure)    #bbox_transform=f.transFigure le está diciendo que la legenda la ubique usando la figura grande, y NO el subplot
                                                                          #bbox_to_anchor=(1, 1) ubica en la posicion arriba a la derecha a la leyenda
#ax.get_xticklabels()  
                                                                       #loc=1 se refiere a la esquina derecha superior DE LA LEYENDA! NO DE LA FIGURA. ie: para ubicar la leyenda en el bbox_to_anchor utiliza el punto de arriba a la derecha de la leyenda
for ax in axes:
#    ax.set_xticks([])
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

#f.suptitle('DIFUSION \n')
plt.show()



