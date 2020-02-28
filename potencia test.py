# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 11:10:12 2019
@author: ETCasa
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
with open('C:\\Users\\ETcasa\\Desktop\\S2\\S2 calculado de ajustar con python __ Ajuste de proceso de binding puro por modelo de DIFUSION.txt') as fobj:
    Proceso_BindingPURO_ajustado_por_dif = fobj.read()
S2_binding_puro_ajustado_por_dif = re.split('\n', Proceso_BindingPURO_ajustado_por_dif)
S2_binding_puro_ajustado_por_dif.remove('')
S2_binding_puro_ajustado_por_dif = [float(i) for i in S2_binding_puro_ajustado_por_dif]

with open('C:\\Users\\ETcasa\\Desktop\\S2\\S2 calculado de ajustar con python __ Ajuste de proceso de binding puro por modelo de Binding PURO.txt') as fobj:
    Proceso_BindingPURO_ajustado_por_BindingPURO = fobj.read()
S2_binding_puro_ajustado_por_bind = re.split('\n', Proceso_BindingPURO_ajustado_por_BindingPURO)
S2_binding_puro_ajustado_por_bind.remove('')
S2_binding_puro_ajustado_por_bind = [float(i) for i in S2_binding_puro_ajustado_por_bind]

with open('C:\\Users\\ETcasa\\Desktop\\S2\\S2 calculado de ajustar con python __ Ajuste de proceso de difusion pura por modelo de DIFUSION.txt') as fobj:
    Proceso_dif_ajustado_por_dif = fobj.read()
S2_dif_ajustado_por_dif = re.split('\n', Proceso_dif_ajustado_por_dif)
S2_dif_ajustado_por_dif.remove('')
S2_dif_ajustado_por_dif = [float(i) for i in S2_dif_ajustado_por_dif]

with open('C:\\Users\\ETcasa\\Desktop\\S2\\S2 calculado de ajustar con python __ Ajuste de proceso de difusion pura por modelo de Binding PURO.txt') as fobj:
    Proceso_dif_ajustado_por_bind = fobj.read()
S2_dif_ajustado_por_bind = re.split('\n', Proceso_dif_ajustado_por_bind)
S2_dif_ajustado_por_bind.remove('')
S2_dif_ajustado_por_bind = [float(i) for i in S2_dif_ajustado_por_bind]


#==============================================================================
# PARAMETROS DE TAMAÑAO DE LETRA DE LOS GRAFICOS
#==============================================================================
plt.close('all') # amtes de graficar, cierro todos las figuras que estén abiertas

SMALL_SIZE = 26
MEDIUM_SIZE = 30
BIGGER_SIZE = 32

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=50)  # fontsize of the figure title 
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







def Gaussiana(mu,sigma):
     
    x_inicial = mu-10*sigma
    x_final = mu+10*sigma
    x_gaussiana=np.linspace(x_inicial,x_final,num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste
   
    gaussiana3=(1/np.sqrt(2*np.pi*(sigma**2)))*np.exp((-.5)*((x_gaussiana-mu)/(sigma))**2)

    return (x_gaussiana,gaussiana3)

def Gaussiana2(x,mu,sigma):

    return (1/np.sqrt(2*np.pi*(sigma**2)))*np.exp((-.5)*((x-mu)/(sigma))**2)

def alpha(s,mu, sigma):
    return (quad(Gaussiana2, s, np.inf, args=(mu, sigma)))[0]

def beta(s,mu, sigma):
    return (quad(Gaussiana2, -np.inf, s, args=(mu, sigma)))[0]




#==============================================================================    
#                               Busco S2ciritco en el proceso de Difusion ajustado por difusión pidiendole que alpha=0.05
#==============================================================================   
s=mu_ajuste_Dif_por_Dif
significancia=0.05

while alpha(s, mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)>significancia:
#    print(mu_ajuste_Dif_por_Dif)
    s=s+0.001
#    print(alpha(s, mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif))
    

S2critico = s-0.001
ALPHA = alpha(S2critico, mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)    

#==============================================================================    
#                                Busco beta  y la potencia del test    en el proceso de Difusion ajustado por asoci - disoc 
#==============================================================================   

BETA=beta(S2critico, mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif)
potencia=1.0-BETA



X=np.linspace(-0.01, 
              mu_ajuste_BindingPURO_por_Dif+5*sigma_ajuste_BindingPURO_por_Dif,1000)



plt.figure()
plt.subplot(2,1,1)
plt.tick_params(which='minor', length=8, width=3.5)
plt.tick_params(which='major', length=10, width=3.5)
figManager = plt.get_current_fig_manager()  ####   esto y la linea de abajo me maximiza la ventana de la figura
figManager.window.showMaximized()           ####   esto y la linea de arriba me maximiza la ventana de la figura
plt.show()
plt.plot(X, Gaussiana2(X,mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif),
         '--',color='forestgreen')
#         , label='Proceso: Dif \n Ajuste: Dif \n $\mu$= {:.4f} - $\sigma$ = {:.4f} \n $\\alpha$={:.3f}'
#         .format(mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif, ALPHA)
#         )
plt.text(0.06,18,s='Proceso: Dif \n Ajuste: Dif \n $\mu$= {:.4f} - $\sigma$ = {:.4f} \n $\\alpha$={:.3f}'
         .format(mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif, ALPHA),
         fontsize=24)


plt.ylabel('PDF')
plt.xlim(-0.015, max(X))
plt.axvline(S2critico,color='r')
plt.text(S2critico*1.1,10,s='$S^2_c$ = {:.3f}'.format(S2critico), color='r',rotation=90)
#plt.legend(loc='upper right')

#plt.figure()
plt.subplot(2,1,2)
plt.plot(X, Gaussiana2(X,mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif),
         '--',color='#8f1402')#, label='Proceso: Asoc-Disoc  \n Ajuste: Dif \n $\mu$= {:.4f} - $\sigma$ = {:.4f} \n $\\beta$= {:.3f} \n Potencia del test = {:.2f}'
#         .format(mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif, BETA, potencia)
#         )
plt.text(-0.005,25,s='Proceso: Asoc-Disoc  \n Ajuste: Dif \n $\mu$= {:.4f} - $\sigma$ = {:.4f} \n $\\beta$= {:.3f} \n Potencia del test = {:.2f}'
         .format(mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif, BETA, potencia),
         fontsize=24)

plt.xlabel('$S^2$')
plt.ylabel('PDF')
plt.xlim(-0.015, max(X))
plt.axvline(S2critico,color='r')
plt.text(S2critico*1.1,10,s='$S^2_c$ = {:.3f}'.format(S2critico), color='r',rotation=90)
#plt.legend(loc='upper left')
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#plt.tight_layout()
plt.tick_params(which='minor', length=8, width=3.5)
plt.tick_params(which='major', length=10, width=3.5)
figManager = plt.get_current_fig_manager()  ####   esto y la linea de abajo me maximiza la ventana de la figura
figManager.window.showMaximized()           ####   esto y la linea de arriba me maximiza la ventana de la figura
plt.show()
plt.show()










#==============================================================================    
#                               Busco S2ciritco en el proceso de Difusion ajustado por difusión pidiendole que alpha=0.05
#==============================================================================   
s=mu_ajuste_BindingPURO_por_BindingPURO

while alpha(s, mu_ajuste_BindingPURO_por_BindingPURO,sigma_ajuste_BindingPURO_por_BindingPURO)>significancia:
#    print(alpha(s, mu_ajuste_BindingPURO_por_BindingPURO,sigma_ajuste_BindingPURO_por_BindingPURO))
    s=s+0.0001
    
S2critico = s-0.0001
ALPHA = alpha(S2critico,mu_ajuste_BindingPURO_por_BindingPURO,sigma_ajuste_BindingPURO_por_BindingPURO)
#==============================================================================    
#                                Busco beta  y la potencia del test    en el proceso de Difusion ajustado por asoci - disoc 
#==============================================================================   

BETA=beta(S2critico, mu_ajuste_Dif_por_Bind, sigma_ajuste_Diff_por_Bind )
potencia=1-BETA



X=np.linspace(-0.005,
              mu_ajuste_Dif_por_Bind+3*sigma_ajuste_Diff_por_Bind,1000)



plt.figure()
plt.subplot(2,1,1)
plt.plot(X, Gaussiana2(X,mu_ajuste_BindingPURO_por_BindingPURO,sigma_ajuste_BindingPURO_por_BindingPURO),
         '--', color='tomato')
#         , label='Proceso: Asoc-disoc \n Ajuste: Asoc-disoc \n $\mu$= {:.4f} - $\sigma$ = {:.4f} \n $\\alpha$={:.3f}'
#         .format(mu_ajuste_BindingPURO_por_BindingPURO, sigma_ajuste_BindingPURO_por_BindingPURO, ALPHA)
#         )

plt.text(0.17,100,s='Proceso: Asoc-disoc \n Ajuste: Asoc-disoc \n $\mu$= {:.4f} - $\sigma$ = {:.4f} \n $\\alpha$={:.3f}'
         .format(mu_ajuste_BindingPURO_por_BindingPURO, sigma_ajuste_BindingPURO_por_BindingPURO, ALPHA)
         , fontsize=24)
         
#plt.xlabel('$S^2$')
plt.ylabel('PDF')
plt.axvline(S2critico,color='r')
plt.text(S2critico*1.15,20,s='$S^2_c$ = {:.3f}'.format(S2critico), color='r',rotation=90)
plt.xlim(min(X),0.3)
plt.ylim(-10,200)
#plt.legend(loc='upper right')
plt.tick_params(which='minor', length=8, width=3.5)
plt.tick_params(which='major', length=10, width=3.5)
figManager = plt.get_current_fig_manager()  ####   esto y la linea de abajo me maximiza la ventana de la figura
figManager.window.showMaximized()           ####   esto y la linea de arriba me maximiza la ventana de la figura
plt.show()
#plt.figure()
plt.subplot(2,1,2)
plt.plot(X, Gaussiana2(X,mu_ajuste_Dif_por_Bind,sigma_ajuste_Diff_por_Bind)
         , 'b--')
#         , label='Proceso: Dif \n Ajuste: Asociacion-Disociación \n $\mu$= {:.3f} - $\sigma$ = {:.3f} \n $\\beta$= {:.3f} \n Potencia del test = {:.8f}'
#         .format(mu_ajuste_Dif_por_Bind, sigma_ajuste_Diff_por_Bind, BETA, potencia)
#         )

plt.xlabel('$S^2$')
plt.ylabel('PDF')
plt.axvline(S2critico,color='r')
plt.text(S2critico*1.15,2,s='$S^2_c$ = {:.3f}'.format(S2critico), color='r',rotation=90)
plt.text(0.187,4,s='Proceso: Dif \n Ajuste: Asociacion-Disociación \n $\mu$= {:.3f} - $\sigma$ = {:.3f} \n $\\beta$= {:.3f} \n Potencia del test = {:.8f}'
         .format(mu_ajuste_Dif_por_Bind, sigma_ajuste_Diff_por_Bind, BETA, potencia)
         , fontsize=24)

plt.xlim(min(X),0.3)
plt.ylim(-0.5,15)
plt.tick_params(which='minor', length=8, width=3.5)
plt.tick_params(which='major', length=10, width=3.5)
figManager = plt.get_current_fig_manager()  ####   esto y la linea de abajo me maximiza la ventana de la figura
figManager.window.showMaximized()           ####   esto y la linea de arriba me maximiza la ventana de la figura
plt.show()
#plt.legend(loc='upper right')
#plt.legend(loc='upper left', bbox_to_anchor=[1,1])
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#plt.tight_layout()

#plt.savefig()