# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 11:10:12 2019

@author: LEC
"""

import numpy as np
import re
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from scipy.integrate import simps, quadrature, trapz, quad
from scipy.stats import norm
from scipy.optimize import minimize

#==============================================================================
#                       Cargo LISTAS CON VALORES DE S2 
#==============================================================================    
with open('C:\\Users\\LEC\\Desktop\\S2\\S2 calculado de ajustar con python __ Ajuste de proceso de binding puro por modelo de DIFUSION.txt') as fobj:
    Proceso_BindingPURO_ajustado_por_dif = fobj.read()
S2_binding_puro_ajustado_por_dif = re.split('\n', Proceso_BindingPURO_ajustado_por_dif)
S2_binding_puro_ajustado_por_dif.remove('')
S2_binding_puro_ajustado_por_dif = [float(i) for i in S2_binding_puro_ajustado_por_dif]

with open('C:\\Users\\LEC\\Desktop\\S2\\S2 calculado de ajustar con python __ Ajuste de proceso de binding puro por modelo de Binding PURO.txt') as fobj:
    Proceso_BindingPURO_ajustado_por_BindingPURO = fobj.read()
S2_binding_puro_ajustado_por_bind = re.split('\n', Proceso_BindingPURO_ajustado_por_BindingPURO)
S2_binding_puro_ajustado_por_bind.remove('')
S2_binding_puro_ajustado_por_bind = [float(i) for i in S2_binding_puro_ajustado_por_bind]

with open('C:\\Users\\LEC\\Desktop\\S2\\S2 calculado de ajustar con python __ Ajuste de proceso de difusion pura por modelo de DIFUSION.txt') as fobj:
    Proceso_dif_ajustado_por_dif = fobj.read()
S2_dif_ajustado_por_dif = re.split('\n', Proceso_dif_ajustado_por_dif)
S2_dif_ajustado_por_dif.remove('')
S2_dif_ajustado_por_dif = [float(i) for i in S2_dif_ajustado_por_dif]

with open('C:\\Users\\LEC\\Desktop\\S2\\S2 calculado de ajustar con python __ Ajuste de proceso de difusion pura por modelo de Binding PURO.txt') as fobj:
    Proceso_dif_ajustado_por_bind = fobj.read()
S2_dif_ajustado_por_bind = re.split('\n', Proceso_dif_ajustado_por_bind)
S2_dif_ajustado_por_bind.remove('')
S2_dif_ajustado_por_bind = [float(i) for i in S2_dif_ajustado_por_bind]

#with open('C:\\Users\\LEC\\Desktop\\Datos para calcular S2\\S2 calculado de ajustar con python __ Ajuste de proceso de difusion pura por modelo de DIFUSION_y_BINDING.txt') as fobj:
#    Proceso_dif_ajustado_por_dify_bind = fobj.read()
#S2_dif_ajustado_por_dif_y_bind = re.split('\n', Proceso_dif_ajustado_por_dify_bind)
#S2_dif_ajustado_por_dif_y_bind.remove('')
#S2_dif_ajustado_por_dif_y_bind = [float(i)*3969 for i in S2_dif_ajustado_por_dif_y_bind]

#with open('C:\\Users\\LEC\\Desktop\\Datos para calcular S2\\S2 calculado de ajustar con python __ Ajuste de proceso de difusion y binding por modelo de DIFUSION.txt') as fobj:
#    Proceso_dif_y_bind_ajustado_por_dif = fobj.read()
#S2_dif_y_bind_ajustado_por_dif = re.split('\n', Proceso_dif_y_bind_ajustado_por_dif )
#S2_dif_y_bind_ajustado_por_dif .remove('')
#S2_dif_y_bind_ajustado_por_dif = [float(i)*3969 for i in S2_dif_y_bind_ajustado_por_dif]

#with open('C:\\Users\\LEC\\Desktop\\Datos para calcular S2\\S2 calculado de ajustar con python __ Ajuste de proceso de difusion y binding por modelo de DIFUSION_y_BINDING.txt') as fobj:
#    Proceso_dif_y_bind_ajustado_por_dify_bind = fobj.read()
#    
#S2_dif_y_bind_ajustado_por_dif_y_bind = re.split('\n', Proceso_dif_y_bind_ajustado_por_dify_bind )
#S2_dif_y_bind_ajustado_por_dif_y_bind.remove('')
#S2_dif_y_bind_ajustado_por_dif_y_bind = [float(i)*3969 for i in S2_dif_y_bind_ajustado_por_dif_y_bind]

#==============================================================================
# PARAMETROS DE TAMAÑAO DE LETRA DE LOS GRAFICOS
#==============================================================================
plt.close('all') # amtes de graficar, cierro todos las figuras que estén abiertas

SMALL_SIZE = 22
MEDIUM_SIZE = 26
BIGGER_SIZE = 30


plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title 
plt.rc('lines', linewidth=4)

#==============================================================================
#                        #####      Gaussiana      #####
#==============================================================================
#==============================================================================    
#                               Proceso Binding PURO
#                               Ajuste: Binding PURO
#==============================================================================    
mu_ajuste_BindingPURO_por_BindingPURO=np.mean(S2_binding_puro_ajustado_por_bind) # media
sigma_ajuste_BindingPURO_por_BindingPURO=np.std(S2_binding_puro_ajustado_por_bind) #desviación estándar     std = sqrt(mean(abs(x - x.mean())**2))
N_ajuste_BindingPURO_por_BindingPURO=len(S2_binding_puro_ajustado_por_bind) # número de mediciones
std_err_ajuste_BindingPURO_por_BindingPURO = sigma_ajuste_BindingPURO_por_BindingPURO/ N_ajuste_BindingPURO_por_BindingPURO # error estándar

#==============================================================================    
#                               Proceso Binding PURO
#                               Ajuste: Difusion
#==============================================================================    
mu_ajuste_BindingPURO_por_Dif=np.mean(S2_binding_puro_ajustado_por_dif) # media
sigma_ajuste_BindingPURO_por_Dif=np.std(S2_binding_puro_ajustado_por_dif) #desviación estándar     std = sqrt(mean(abs(x - x.mean())**2))
N_ajuste_BindingPURO_por_Dif=len(S2_binding_puro_ajustado_por_dif) # número de mediciones
std_err_ajuste_BindingPURO_por_Dif = sigma_ajuste_BindingPURO_por_Dif / N_ajuste_BindingPURO_por_Dif # error estándar

#==============================================================================    
#                               Proceso Binding PURO
#                               Ajuste: Difusion y Binding
#==============================================================================    
#mu_ajuste_BindingPURO_por_dif_y_bind=np.mean(S2_BindingPURO_ajustado_por_dif_y_bind) # media
#sigma_ajuste_BindingPURO_por_dif_y_bind=np.std(S2_BindingPURO_ajustado_por_dif_y_bind) #desviación estándar     std = sqrt(mean(abs(x - x.mean())**2))
#N_ajuste_BindingPURO_por_dif_y_bind=len(S2_BindingPURO_ajustado_por_dif_y_bind) # número de mediciones
#std_err_ajuste_BindingPURO_por_dif_y_bind = sigma_ajuste_BindingPURO_por_dif_y_bind/ N_ajuste_BindingPURO_por_dif_y_bind # error estándar

#==============================================================================    
#                               Proceso Difusion
#                               Ajuste: Binding
#==============================================================================    
mu_ajuste_Dif_por_Bind=np.mean(S2_dif_ajustado_por_bind) # media
sigma_ajuste_Diff_por_Bind=np.std(S2_dif_ajustado_por_bind) #desviación estándar     std = sqrt(mean(abs(x - x.mean())**2))
N_ajuste_Diff_por_Bind=len(S2_dif_ajustado_por_bind) # número de mediciones
std_err_ajuste_Diff_por_Bind = sigma_ajuste_Diff_por_Bind/ N_ajuste_Diff_por_Bind # error estándar

#==============================================================================    
#                               Proceso Difusion
#                               Ajuste: Difusion
#==============================================================================    
mu_ajuste_Dif_por_Dif=np.mean(S2_dif_ajustado_por_dif) # media
sigma_ajuste_Diff_por_Dif=np.std(S2_dif_ajustado_por_dif) #desviación estándar     std = sqrt(mean(abs(x - x.mean())**2))
N_ajuste_Diff_por_Dif=len(S2_dif_ajustado_por_dif) # número de mediciones
std_err_ajuste_Diff_por_Dif = sigma_ajuste_Diff_por_Dif / N_ajuste_Diff_por_Dif # error estándar

#==============================================================================    
#                               Proceso Difusion
#                               Ajuste: Difusion y binding
#==============================================================================    
#mu_ajuste_Dif_por_Dif_y_Bind=np.mean(S2_dif_ajustado_por_dif_y_bind) # media
#sigma_ajuste_Dif_por_Dif_y_Bind=np.std(S2_dif_ajustado_por_dif_y_bind) #desviación estándar     std = sqrt(mean(abs(x - x.mean())**2))
#N_ajuste_Dif_por_Dif_y_Bind=len(S2_dif_ajustado_por_dif_y_bind) # número de mediciones
#std_err_ajuste_Dif_por_Dif_y_Bind = sigma_ajuste_Dif_por_Dif_y_Bind / N_ajuste_Dif_por_Dif_y_Bind # error estándar

#==============================================================================    
#                               Proceso Difusion y binding
#                               Ajuste: Difusion 
#==============================================================================    
#mu_ajuste_Dif_y_Bind_por_Dif=np.mean(S2_dif_y_bind_ajustado_por_dif) # media
#sigma_ajuste_Dif_y_Bind_por_Dif=np.std(S2_dif_y_bind_ajustado_por_dif) #desviación estándar     std = sqrt(mean(abs(x - x.mean())**2))
#N_ajuste_Dif_y_Bind_por_Dif=len(S2_dif_y_bind_ajustado_por_dif) # número de mediciones
#std_err_ajuste_Diff = sigma_ajuste_Dif_y_Bind_por_Dif / N_ajuste_Dif_y_Bind_por_Dif # error estándar

#==============================================================================    
#                               Proceso Difusion y binding
#                               Ajuste: Difusion y binding
#==============================================================================    
#mu_ajuste_Dif_y_Bind_por_Dif_y_Bind = np.mean(S2_dif_y_bind_ajustado_por_dif_y_bind) # media
#sigma_ajuste_Dif_y_Bind_por_Dif_y_Bind=np.std(S2_dif_y_bind_ajustado_por_dif_y_bind) #desviación estándar     std = sqrt(mean(abs(x - x.mean())**2))
#N_ajuste_Dif_y_Bind_por_Dif_y_Bind=len(S2_dif_y_bind_ajustado_por_dif_y_bind) # número de mediciones
#std_err_ajuste_Diff = sigma_ajuste_Dif_y_Bind_por_Dif_y_Bind / N_ajuste_Dif_y_Bind_por_Dif_y_Bind# error estándar






#==============================================================================
#                           #####     Funcion Gaussiana  #####
#==============================================================================   
def p_valor_minimize(int1, int2):

    resta_de_p_val = []
    i=0
    while i <len(int1):
        resta_de_p_val.append(abs(int1[i]-int2[i]))
        i+=1
        
    minimo_resta_p_val = resta_de_p_val.index(min(resta_de_p_val))    
    
    return minimo_resta_p_val  


def Gaussiana(mu,sigma):
     
    x_inicial = mu-10*sigma
    x_final = mu+10*sigma
    x_gaussiana=np.linspace(x_inicial,x_final,num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste
   
    gaussiana3=(1/np.sqrt(2*np.pi*(sigma**2)))*np.exp((-.5)*((x_gaussiana-mu)/(sigma))**2)

    return (x_gaussiana,gaussiana3)

def Gaussiana2(x,mu,sigma):

    return (1/np.sqrt(2*np.pi*(sigma**2)))*np.exp((-.5)*((x-mu)/(sigma))**2)


#==============================================================================    
#                               Proceso Difusion         Ajuste: Difusion  y   BINDING 
#==============================================================================   
S2_X=np.linspace(mu_ajuste_Dif_por_Dif, mu_ajuste_Dif_por_Bind,200)
X=Gaussiana(mu_ajuste_Dif_por_Dif,999999999)[0]

S2_X=np.linspace(mu_ajuste_Dif_por_Dif, mu_ajuste_Dif_por_Bind,2000)
int1=[]
int2=[]

for s in S2_X:
    int1.append((quad(Gaussiana2, s, np.inf, args=(mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif)))[0])
    int2.append((quad(Gaussiana2, -np.inf, s, args=(mu_ajuste_Dif_por_Bind,sigma_ajuste_Diff_por_Bind)))[0])

p__valor = S2_X[p_valor_minimize(int1, int2)]
print(p__valor)

plt.figure()
plt.subplot(1,2,1)
plt.semilogy(S2_X, int1,'--', color='forestgreen', label= 'Dif ajustado por Dif')
plt.semilogy(S2_X, int2,'b--', label= 'Dif ajustado por Bind')
plt.xlim(0.02,0.13)
plt.ylim(1e-10,50)
plt.xlabel('S2')
plt.ylabel('log(p-valor)')
plt.axvline(p__valor,color='r')
plt.text(p__valor+p__valor*.03,1e-5,s='$S^2$ critico = {:.3f}'.format(p__valor), color='r',rotation=90)
plt.legend(loc='upper right')


plt.subplot(1,2,2)
plt.plot(Gaussiana(mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif)[0],
         Gaussiana(mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif)[1],
         '--',color='forestgreen', 
         label='Proceso: Dif \n Ajuste: Dif \n $\mu$= {:.4f} - $\sigma$ = {:.4f}'.format(mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)
         )

plt.plot(Gaussiana(mu_ajuste_Dif_por_Bind,sigma_ajuste_Diff_por_Bind)[0],
         Gaussiana(mu_ajuste_Dif_por_Bind,sigma_ajuste_Diff_por_Bind)[1],
         'b--', label='Proceso: Dif \n Ajuste: Bind \n $\mu$= {:.4f} - $\sigma$ = {:.4f}'.format(mu_ajuste_Dif_por_Bind, sigma_ajuste_Diff_por_Bind)
         )

plt.xlim(-0.05,0.3)
plt.xlabel('S2')
#plt.ylabel('Frecuencia')
plt.legend(loc='upper right')
plt.show()




#==============================================================================    
#                               Proceso Binding          Ajuste: Difusion  y   BINDING 
#==============================================================================   
S2_X=np.linspace(mu_ajuste_BindingPURO_por_BindingPURO, mu_ajuste_BindingPURO_por_Dif,200)
int1=[]
int2=[]

for s in S2_X:
    int1.append((quad(Gaussiana2, s, np.inf, args=(mu_ajuste_BindingPURO_por_BindingPURO,sigma_ajuste_BindingPURO_por_BindingPURO)))[0])
    int2.append((quad(Gaussiana2, -np.inf, s, args=(mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif)))[0])

p__valor = S2_X[p_valor_minimize(int1, int2)]
print(p__valor)

plt.figure()
plt.subplot(1,2,1)
plt.semilogy(S2_X, int1,'--', color='tomato', label= 'Bind ajustado por Bind')
plt.semilogy(S2_X, int2,'--', color='#8f1402', label= 'Bind ajustado por Dif')
plt.ylim(1e-32,100)
plt.xlabel('S2')
plt.ylabel('log(p-valor)')
plt.axvline(p__valor,color='r')
plt.text(p__valor+p__valor*.025,1e-5,s='$S^2$ critico = {:.3f}'.format(p__valor), color='r',rotation=90)
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.plot(Gaussiana(mu_ajuste_BindingPURO_por_BindingPURO,sigma_ajuste_BindingPURO_por_BindingPURO)[0],
         Gaussiana(mu_ajuste_BindingPURO_por_BindingPURO,sigma_ajuste_BindingPURO_por_BindingPURO)[1],
         '--', color='tomato', label='Proceso: Bind \n Ajuste: Bind \n $\mu$= {:.3f} - $\sigma$ = {:.3f}'.format(mu_ajuste_BindingPURO_por_BindingPURO, sigma_ajuste_BindingPURO_por_BindingPURO)
         )

plt.plot(Gaussiana(mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif)[0],
         Gaussiana(mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif)[1],
         '--', color='#8f1402', label='Proceso: Bind \n Ajuste: Dif \n $\mu$= {:.3f} - $\sigma$ = {:.3f}'.format(mu_ajuste_BindingPURO_por_Dif, sigma_ajuste_BindingPURO_por_Dif)
         )
plt.xlabel('S2')
#plt.ylabel('p-valor')
plt.legend(loc='upper right')
plt.show()





#==============================================================================    
#                               Proceso Binding y Difusion      Ajuste: BINDING 
#==============================================================================   
S2_X=np.linspace(mu_ajuste_BindingPURO_por_BindingPURO, mu_ajuste_Dif_por_Bind,200)
int1=[]
int2=[]

for s in S2_X:
    int1.append((quad(Gaussiana2, s, np.inf, args=(mu_ajuste_BindingPURO_por_BindingPURO,sigma_ajuste_BindingPURO_por_BindingPURO)))[0])
    int2.append((quad(Gaussiana2, -np.inf, s, args=(mu_ajuste_Dif_por_Bind,sigma_ajuste_Diff_por_Bind)))[0])

p__valor = S2_X[p_valor_minimize(int1, int2)]
print(p__valor)

plt.figure()
plt.subplot(1,2,1)
plt.loglog(S2_X, int1,'--', color='tomato', label= 'Bind ajustado por Bind')
plt.loglog(S2_X, int2,'b--', label= 'Dif ajustado por Bind')
plt.ylim(1e-8,10)
plt.xlabel('log(S2)')
plt.ylabel('log(p-valor)')
plt.axvline(p__valor,color='r')
plt.text(p__valor-p__valor*.1,1e-4,s='$S^2$ critico = {:.3f}'.format(p__valor), color='r',rotation=90)
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.plot(Gaussiana(mu_ajuste_BindingPURO_por_BindingPURO,sigma_ajuste_BindingPURO_por_BindingPURO)[0],
         Gaussiana(mu_ajuste_BindingPURO_por_BindingPURO,sigma_ajuste_BindingPURO_por_BindingPURO)[1],
         '--', color='tomato', label='Proceso: Bind \n Ajuste: Bind \n $\mu$= {:.3f} - $\sigma$ = {:.3f}'.format(mu_ajuste_BindingPURO_por_BindingPURO, sigma_ajuste_BindingPURO_por_BindingPURO)
         )

plt.plot(Gaussiana(mu_ajuste_Dif_por_Bind,sigma_ajuste_Diff_por_Bind)[0],
         Gaussiana(mu_ajuste_Dif_por_Bind,sigma_ajuste_Diff_por_Bind)[1],
         'b--', label='Proceso: Dif \n Ajuste: Bind \n $\mu$= {:.3f} - $\sigma$ = {:.3f}'.format(mu_ajuste_Dif_por_Bind, sigma_ajuste_Diff_por_Bind)
         )
plt.xlabel('S2')
#plt.ylabel('p-valor')

plt.xlim(-0.05,0.3)
plt.legend(loc='upper right')
plt.show()

#==============================================================================    
#                               Proceso Binding y Difusion      Ajuste: DIFUSION 
#==============================================================================   
S2_X=np.linspace(mu_ajuste_Dif_por_Dif, mu_ajuste_BindingPURO_por_Dif,200)
int1=[]
int2=[]

for s in S2_X:
    int1.append((quad(Gaussiana2, s, np.inf, args=(mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif)))[0])
    int2.append((quad(Gaussiana2, -np.inf, s, args=(mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif)))[0])

p__valor = S2_X[p_valor_minimize(int1, int2)]
print(p__valor)

plt.figure()
plt.subplot(1,2,1)
plt.semilogy(S2_X, int1,'--', color='forestgreen', label= 'Dif ajustado por Dif')
plt.semilogy(S2_X, int2,'--', color='#8f1402', label= 'Bind ajustado por Dif')
plt.ylim(1e-15,100)
plt.xlabel('S2')
plt.ylabel('log(p-valor)')
plt.axvline(p__valor,color='r')
plt.text(p__valor+p__valor*.025,1e-8,s='$S^2$ critico = {:.3f}'.format(p__valor), color='r',rotation=90)
plt.legend(loc='upper left')

plt.subplot(1,2,2)
plt.plot(Gaussiana(mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif)[0],
         Gaussiana(mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif)[1],
         '--',color='forestgreen',
         label='Proceso: Dif \n Ajuste: Dif \n $\mu$= {:.3f} - $\sigma$ = {:.3f}'.format(mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif)
         )

plt.plot(Gaussiana(mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif)[0],
         Gaussiana(mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif)[1],
         '--', color='#8f1402',
         label='Proceso: Bind \n Ajuste: Dif \n $\mu$= {:.3f} - $\sigma$ = {:.3f}'.format(mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif)
         )
plt.xlabel('S2')
#plt.ylabel('p-valor')
plt.xlim(-0.075,0.15)
plt.legend(loc='upper left')
plt.show()

