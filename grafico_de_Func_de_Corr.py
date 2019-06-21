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
from mpl_toolkits.mplot3d import Axes3D
import lmfit 
from scipy import interpolate

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
    with open('C:\\Users\\LEC\\Desktop\\sim{}-5-4-19-DATA.txt'.format(j)) as fobj:
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
##                 Grafico la funcion de correlación en 3D
##============================================================================== 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(seda, psi, G,cmap='viridis', edgecolor='none')
plt.show()


##==============================================================================
##                 Grafico la funcion de correlación en 2D
##==============================================================================        
plt.figure()
ACF = np.asarray(G)  #para graficar debo pasar de list a un array
plt.imshow(ACF.reshape(63,63))  
plt.show()
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
plt.figure()
plt.plot(G, 'b*', label='ACF')
plt.show()

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
N = 200            #Numero total de particulas en la PSF



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
    
    return ((gamma/N)*( 1 + 4*D*(tp*x+tl*y) / w0**2 )**(-1) * ( 1 + 4*D*(tp*x+tl*y) / wz**2 )**(-1/2) *
            np.exp(-0.5*((2*x*dr/w0)**2+(2*y*dr/w0)**2)/(1 + 4*D*(tp*x+tl*y)/(w0**2))))



#def Difusion (x,y,gamma,N,w0,wz,tp,tl):
#
#    return (gamma/N)*( 1 + 4*d*(tp*x+tl*y) / w0**2 )**(-1) * ( 1 + 4*D*(tp*x+tl*y) / wz**2 )**(-1/2)

#def Triplete (x,y,tp,tl,At,t_triplet):
#    
#    return 1+ At * np.exp(-(tp*x+tl*y)/t_triplet)
#
#def Binding(x,y,tp,tl,Ab,t_binding):
#    
#    return Ab * np.exp(-(x*dr/w0)**2-(y*dr/w0))* np.exp(-(tp*x+tl*y)/t_binding)

x = np.arange(0, 32)
y = G

popt, pcov = curve_fit(Difusion, x, y, p0=(9.5,0.017))

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
##                 AJUSTE DE SUPERFICIE 2D
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
    
    return ((gamma/N)*( 1 + 4*D*(tp*x+tl*y) / w0**2 )**(-1) * ( 1 + 4*D*(tp*x+tl*y) / wz**2 )**(-1/2) *
            np.exp(-0.5*((2*x*dr/w0)**2+(2*y*dr/w0)**2)/(1 + 4*D*(tp*x+tl*y)/(w0**2))))



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

#M = np.reshape(63,63)

x_1 = np.arange(-31, 32, 1)
y_1 = np.arange(-31, 32, 1)
x, y = np.meshgrid(x_1, y_1)
print(x,y)


X_DATA = np.vstack((A))

X = [N[0] for N in X_DATA]
Y = [N[1] for N in X_DATA]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(X,Y, Difusion(X_DATA,10,0.017),cmap='viridis', edgecolor='none')
#ax.set_zlim(0,50)
plt.show()




fit_params, cov_mat = curve_fit(Difusion,X_DATA, G, p0=(9.9,0.017))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(X,Y, Difusion(X_DATA,fit_params[0],fit_params[1]),cmap='viridis', edgecolor='none')
#ax.set_zlim(0,50)
plt.show()

plt.title("ACF")
plt.show()    





########   version hernan
#A = (x + 1j *y).flatten()
#
#def chi2(pars):
#    return np.sum((Difusion(x, y, pars[0], pars[1]) - z)**2)
#
#print(minimize(chi2, (10, 0.017)))
#
#print(chi2((10, 0.017)))
#
#
#
#fit_params, cov_mat = curve_fit(Difusion, A, G, p0=(9.5,0.017))
#plt.figure()
#plt.contourf(x,y,z)
##plt.pcolor(x, y, z)
##plt.colorbar()
#plt.title("ACF")
#plt.show()    








##==============================================================================
##                 CREO MATRIZ de tuplas (i,j)
##============================================================================== 
## ahora quiero hacer lo mismo pero que los valores vayan entre -31 y 31
## ie: el ancho es 62, que es lo mismo que el roi que tengo, que sale de hacer 159-97 
#A=[]
#C=[]
#roi = max(seda)-min(seda)
#mitad_del_roi = roi/2
#seda_min = (-1)*mitad_del_roi
#seda_max = mitad_del_roi
#
#k=seda_min    #K va a recorrer las filas
#
#while k < mitad_del_roi+1:
#    j = 0
#    
#    while j<roi+1:
#        C.append((seda_min+j,k))  #parado en una fila, recorro las columnas
#        j+=1
#    
#    A.append(C)  #me guardo la fila completa.
#    C=[]
#    k+=1







#
###---> np.meshgrid() #Return coordinate matrices from coordinate vectors.
#x = np.arange(min(seda), max(seda)+1)
##x = np.linspace(-1, 1, 63)
#y = x
#xy_mesh = np.meshgrid(x, y, sparse=True)
#
#z = Difusion (xy_mesh, 9.5, 0.017)
#xx, yy = np.meshgrid(x, y, sparse=True)
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(xx, yy,  Difusion (xy_mesh, 9.5, 0.017))
#plt.show()
#
#
#print(Difusion(xy_mesh, 9.5, 0.017))
#
#
#z = np.asarray(G).reshape(np.outer(x, y).shape)
#
#print(z.shape)
#
#
#
#fit_params, cov_mat = curve_fit(Difusion, A, z, p0=(9.5,0.017))
#
#plt.figure()
#plt.contourf(x,y,z)
##plt.pcolor(x, y, z)
##plt.colorbar()
#plt.title("ACF")
#plt.show()    








##==============================================================================
##                 Ejemplo de matriz de 5x5 de tuplas (i,j)
##==============================================================================    
A=[]
B=[]
C=[]
k=-2
while k<3:
    j = 0
    i=-2
    while j<5:
        C.append((i+j,k))
        B.append((1,1))
        
        j+=1
    
    A.append(C)
    C=[]
    k+=1

## esto va a dar la siguiente matriz:
##[[(-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2)],
## [(-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1)],
## [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)],
## [(-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1)],
## [(-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2)]]










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

X_DATA = np.vstack((A))
guess_vals = (9.5,0.017)
lmfit_model = Model(Difusion_lmifit)
lmfit_result = lmfit_model.fit(np.ravel(G), 
                               xy_mesh=xy_mesh, 
                               D=guess_vals[0], 
                               N=guess_vals[1])
#
#lmfit_Rsquared = 1 - lmfit_result.residual.var()/np.var(noise)
#
#
#print('Fit R-squared:', lmfit_Rsquared, '\n')
#print(lmfit_result.fit_report())



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