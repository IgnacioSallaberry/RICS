# -*- coding: utf-8 -*-
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
guess_val = [10,0.05]  #inicializo valores con los que voy a ajustar
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

proceso = 'binding puro'        
while q<61:
    print('simulacion numero {}'.format(q))
    with open('C:\\Users\\LEC\\Desktop\\Proceso de binding solo\\200 simulaciones Binding Puro\\sim{}-Proceso_de_binding_puro-DATA.txt'.format(q)) as fobj:
        
        
        
        DATA= fobj.read()
        
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
    ##                 Grafico la funcion de correlación en 3D
    ##============================================================================== 
    if mostrar_imagenes:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(seda, psi, G,cmap='viridis', edgecolor='none')
        ax.set_xlabel(r'pixel', fontsize=15)
        ax.set_ylabel(r'pixel',fontsize=15)
        ax.set_zlabel(r'G($\xi$,$\psi$)',fontsize=15)
        ax.set_title('ACF de txt original en 3D', )
        ax.xaxis.labelpad = 20   #me separa los nombres de los ejes de los numeros
        ax.yaxis.labelpad = 20
        ax.zaxis.labelpad = 20
        plt.show()
    else:
        pass
    
    ##==============================================================================
    ##                 Grafico la funcion de correlación en 2D
    ##==============================================================================        
    if mostrar_imagenes:
        plt.figure()
        ACF = np.asarray(G)  #para graficar debo pasar de list a un array
        plt.imshow(ACF.reshape(63,63))  
        plt.show()
    else:
        pass
    ### nota: que hace Reshape? lista.reshape()   toma la lista y, sin cambiar los valores, lo va cortando hasta acomodarlo en una matriz de nxm. Ojo que nxm debe ser = al len(lista)
    
    
    
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
    
    ##==============================================================================
    ##                 Grafico la linea horizontal en 1D
    ##============================================================================== 
    if mostrar_imagenes:
        plt.figure()
        plt.plot(G, 'b*', label='ACF que da el simFCS')
        plt.show()
    else:
        pass
    ##==============================================================================
    ##                 AJUSTE DE LA LINEA HORIZONTAL
    ##==============================================================================    
    
    
    ####                            Parametros globales iniciales
    box_size = 256
    roi = 128
    tp = 5e-6             #seg    
    tl = box_size * tp    #seg
    dr = 0.05             # delta r = pixel size =  micrometros
    w0 = 0.25             #radio de la PSF  = micrometros    
    wz = 1.5              #alto de la PSF desde el centro  = micrometros
    a = w0/wz
    
    gamma = 0.3536        #gamma factor de la 3DG
    N = 0.017           #Numero total de particulas en la PSF
    
    

    
    def Difusion_H_line (x,D,N):
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

        Dif = (gamma/N)*( 1 + 4*D*(tp*x+tl*y) / w0**2 )**(-1) * ( 1 + 4*D*(tp*x+tl*y) / wz**2 )**(-1/2)
        Scan = np.exp(-((x*dr/w0)**2+(y*dr/w0)**2)/(1 + 4*D*(tp*x+tl*y)/(w0**2)))
        
        return (Dif * Scan)
    
    

    x = np.arange(0, 32)
    y = G  #ACF escrito en tira, no en sabana
    
    popt, pcov = curve_fit(Difusion_H_line, x, y, p0=(guess_val[0],guess_val[1]))
    
    if mostrar_imagenes:
        plt.plot(x, Difusion_H_line(x,popt[0],popt[1]), 'r.-', label='Ajuste')
    
        plt.plot(x, Difusion_H_line(x,guess_val[0],guess_val[1]), 'g-', label='Dibujada poniendo \n valores conocidos' )
        
        plt.xlabel(r'pixel shift $\xi$')# - $\mu m$',fontsize=14)
        plt.ylabel(r'G($\xi$)')#,fontsize=14)
        plt.title(f'H-line \n Proceso:{proceso} - Ajuste:Difusion')
        plt.legend()
        plt.show()
        plt.tight_layout() #hace que no me corte los márgenes
    else:
        pass
    
    
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
    
    if mostrar_imagenes:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(X,Y, Difusion(X_DATA,guess_val[0],guess_val[1]),cmap='viridis', edgecolor='none')
        ax.set_xlabel(r'pixel shift $\xi$', fontsize=15)
        ax.set_ylabel(r'pixel shift $\psi$',fontsize=15)
        ax.set_zlabel(r'G($\xi$,$\psi$)',fontsize=15)
        ax.set_title('ACF de datos originales en 3D', )
        ax.xaxis.labelpad = 20   #me separa los nombres de los ejes de los numeros
        ax.yaxis.labelpad = 20
        ax.zaxis.labelpad = 20
        plt.show()
    else:
        pass
    
    
    fit_params, cov_mat = curve_fit(Difusion,X_DATA, G, p0=(guess_val[0],guess_val[1]))
    
    
    
    if mostrar_imagenes:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(X,Y, Difusion(X_DATA,fit_params[0],fit_params[1]),cmap='viridis', edgecolor='none')
        ax.set_xlabel(r'pixel shift $\xi$', fontsize=15)
        ax.set_ylabel(r'pixel shift $\psi$',fontsize=15)
        ax.set_zlabel(r'G($\xi$,$\psi$)',fontsize=15)
        ax.set_title('ACF: AJUSTE en 3D', )
        ax.xaxis.labelpad = 20   #me separa los nombres de los ejes de los numeros
        ax.yaxis.labelpad = 20
        ax.zaxis.labelpad = 20
        plt.show()    
    else:
        pass
    
    
    
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


###==============================================================================
###                 AJUSTE DE LA LINEA HORIZONTAL        VERSION   LMFIT
###==============================================================================    
#def Difusion_lmifit(x,boxsize,roi,tp,dr,w0,wz,D,N):
#    print('Estos son los valores de X_DATA = {}'.format(X_DATA))
#    x,y=np.abs(X_DATA[0])
##    y=X_DATA[0][1]
##    x = np.abs(np.asarray([N[0] for N in X_DATA]))
##    y = np.abs(np.asarray([N[1] for N in X_DATA]))
#
#    return (gamma/N)*( 1 + 4*D*(tp*x+tl*y) / w0**2 )**(-1) * ( 1 + 4*D*(tp*x+tl*y) / wz**2 )**(-1/2)
#
#
#
#lmfit_model = lmfit.Model(Difusion_lmifit, independent_vars=['D','N'])
#print(lmfit_model.param_names)
#print(lmfit_model.independent_vars)
#
#params = lmfit_model.make_params()
#params.add('boxsize', value=256, vary=False)
#params.add('roi', value=128, vary=False)
#params.add('tp', value=5e-6, vary=False)
#params.add('dr', value=0.05, vary=False)
#params.add('w0', value=0.25, vary=False)
#
#params.add('wz', value=0.05, vary=False)
#params.add('x', value=X_DATA.all(), vary=True)
#
#
##params = lmfit_model.make_params(x=X_DATA.all(), boxsize=256, roi=128, tp=5e-6, dr=0.05, w0=0.25, wz=1.5)   #creo los parametros
#
#
#
#guess_vals = (9.5,0.017)
#
#lmfit_result = lmfit_model.fit(G, 
#                               params,
#                               D=guess_vals[0],
#                               N=guess_vals[1])
##lmfit_Rsquared = 1 - lmfit_result.residual.var()/np.var(noise)
##
##
##print('Fit R-squared:', lmfit_Rsquared, '\n')
#print(lmfit_result.fit_report())

#
#fig = plt.figure()
#ax = fig.add_subplot(121, projection='3d')
#ax.plot_trisurf(X,Y, G,cmap='viridis', edgecolor='none')
#ax = fig.add_subplot(122, projection='3d')
#ax.plot_trisurf(X,Y, lmfit_result.init_fit,cmap='viridis', edgecolor='none')
#ax = fig.add_subplot(123, projection='3d')
#ax.plot_trisurf(X,Y, lmfit_result.best_fit,cmap='viridis', edgecolor='none')
#plt.show()

#
#
#params = lmfit.Parameters()
#params.add('boxsize', value=256, vary=False)
#params.add('roi', value=128, vary=False)
#params.add('tp', value=5e-6, vary=False)
#params.add('dr', value=0.05, vary=False)
#params.add('w0', value=0.25, vary=False)
#params.add('dz', value=0.05, vary=False)
#params.add('D', value=9.5, vary=True)
#params.add('N', value=0.017, vary=True)
#
#

#def Difusion_lmifit(X_DATA, paramss):
#    box_size = paramss['boxsize'].value
#    roi = paramss['roi'].value
#    tp = paramss['tp'].value             #seg    
#    tl = box_size * tp            #seg
#    dr = paramss['dr'].value             # delta r = pixel size =  micrometros
#    w0 = paramss['w0'].value             #radio de la PSF  = micrometros    
#    wz = paramss['wz'].value             #alto de la PSF desde el centro  = micrometros
#    a = w0/wz
#
#    x=X_DATA[0]
#    y=X_DATA[1]
#
#    return (gamma/N)*( 1 + 4*D*(tp*x+tl*y) / w0**2 )**(-1) * ( 1 + 4*D*(tp*x+tl*y) / wz**2 )**(-1/2)
#
#
#




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



#para guardar a la posteridad

##==============================================================================
##                 Ejemplo de matriz de 5x5 de tuplas (i,j)
##==============================================================================    
#A=[]
#B=[]
#C=[]
#k=-2
#while k<3:
#    j = 0
#    i=-2
#    while j<5:
#        C.append((i+j,k))
#        B.append((1,1))
#        
#        j+=1
#    
#    A.append(C)
#    C=[]
#    k+=1

## esto va a dar la siguiente matriz:
##[[(-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2)],
## [(-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1)],
## [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)],
## [(-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1)],
## [(-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2)]]
