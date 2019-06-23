# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 11:10:12 2019

@author: ETCasa
"""

import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab

#==============================================================================
#                       Cargo LISTAS CON VALORES DE S2 
#==============================================================================    
with open('C:\\Users\\ETCasa\\Desktop\\S2 calculado de ajustar con python __ Ajuste de proceso de difusion por modelo de DIFUSION.txt') as fobj:
    Proceso_dif_ajustado_por_dif = fobj.read()
S2_dif_ajustado_por_dif = re.split('\n', Proceso_dif_ajustado_por_dif)
S2_dif_ajustado_por_dif.remove('')
S2_dif_ajustado_por_dif = [float(i)*3969 for i in S2_dif_ajustado_por_dif]

with open('C:\\Users\\ETCasa\\Desktop\\S2 calculado de ajustar con python __ Ajuste de proceso de difusion por modelo de DIFUSION_y_BINDING.txt') as fobj:
    Proceso_dif_ajustado_por_dify_bind = fobj.read()
S2_dif_ajustado_por_dif_y_bind = re.split('\n', Proceso_dif_ajustado_por_dify_bind)
S2_dif_ajustado_por_dif_y_bind.remove('')
S2_dif_ajustado_por_dif_y_bind = [float(i)*3969 for i in S2_dif_ajustado_por_dif_y_bind]

with open('C:\\Users\\ETCasa\\Desktop\\S2 calculado de ajustar con python __ Ajuste de proceso de difusion y binding por modelo de DIFUSION.txt') as fobj:
    Proceso_dif_y_bind_ajustado_por_dif = fobj.read()
S2_dif_y_bind_ajustado_por_dif = re.split('\n', Proceso_dif_y_bind_ajustado_por_dif )
S2_dif_y_bind_ajustado_por_dif .remove('')
S2_dif_y_bind_ajustado_por_dif = [float(i)*3969 for i in S2_dif_y_bind_ajustado_por_dif]

with open('C:\\Users\\ETCasa\\Desktop\\S2 calculado de ajustar con python __ Ajuste de proceso de difusion y binding por modelo de DIFUSION_y_BINDING.txt') as fobj:
    Proceso_dif_y_bind_ajustado_por_dify_bind = fobj.read()
    
S2_dif_y_bind_ajustado_por_dif_y_bind = re.split('\n', Proceso_dif_y_bind_ajustado_por_dify_bind )
S2_dif_y_bind_ajustado_por_dif_y_bind.remove('')
S2_dif_y_bind_ajustado_por_dif_y_bind = [float(i)*3969 for i in S2_dif_y_bind_ajustado_por_dif_y_bind]

#==============================================================================
# PARAMETROS DE TAMAÑAO DE LETRA DE LOS GRAFICOS
#==============================================================================
plt.close('all') # amtes de graficar, cierro todos las figuras que estén abiertas

SMALL_SIZE = 12
MEDIUM_SIZE = 20
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title 


#==============================================================================
#                        #####      Gaussiana      #####
#==============================================================================
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
mu_ajuste_Dif_por_Dif_y_Bind=np.mean(S2_dif_ajustado_por_dif_y_bind) # media
sigma_ajuste_Dif_por_Dif_y_Bind=np.std(S2_dif_ajustado_por_dif_y_bind) #desviación estándar     std = sqrt(mean(abs(x - x.mean())**2))
N_ajuste_Dif_por_Dif_y_Bind=len(S2_dif_ajustado_por_dif_y_bind) # número de mediciones
std_err_ajuste_Dif_por_Dif_y_Bind = sigma_ajuste_Dif_por_Dif_y_Bind / N_ajuste_Dif_por_Dif_y_Bind # error estándar

#==============================================================================    
#                               Proceso Difusion y binding
#                               Ajuste: Difusion 
#==============================================================================    
mu_ajuste_Dif_y_Bind_por_Dif=np.mean(S2_dif_y_bind_ajustado_por_dif) # media
sigma_ajuste_Dif_y_Bind_por_Dif=np.std(S2_dif_y_bind_ajustado_por_dif) #desviación estándar     std = sqrt(mean(abs(x - x.mean())**2))
N_ajuste_Dif_y_Bind_por_Dif=len(S2_dif_y_bind_ajustado_por_dif) # número de mediciones
std_err_ajuste_Diff = sigma_ajuste_Dif_y_Bind_por_Dif / N_ajuste_Dif_y_Bind_por_Dif # error estándar

#==============================================================================    
#                               Proceso Difusion y binding
#                               Ajuste: Difusion y binding
#==============================================================================    
mu_ajuste_Dif_y_Bind_por_Dif_y_Bind = np.mean(S2_dif_y_bind_ajustado_por_dif_y_bind) # media
sigma_ajuste_Dif_y_Bind_por_Dif_y_Bind=np.std(S2_dif_y_bind_ajustado_por_dif_y_bind) #desviación estándar     std = sqrt(mean(abs(x - x.mean())**2))
N_ajuste_Dif_y_Bind_por_Dif_y_Bind=len(S2_dif_y_bind_ajustado_por_dif_y_bind) # número de mediciones
std_err_ajuste_Diff = sigma_ajuste_Dif_y_Bind_por_Dif_y_Bind / N_ajuste_Dif_y_Bind_por_Dif_y_Bind# error estándar


#==============================================================================
#                           #####     HISTOGRAMAS    #####
#==============================================================================    
fig = plt.figure()
#==============================================================================    
#                               Proceso Difusion
#                               Ajuste: Difusion
#==============================================================================    
ax1 =fig.add_subplot(2,2,1)
#bins_ajuste = np.arange(min(lista_S_2), max(lista_S_2)) # le pongo los bins a mano
ax1.set_xlabel('$S^2$')
ax1.set_ylabel('Proceso:\n Difusion')
ax1.set_title('Ajuste:\n Difusion', fontsize=20)
n,bin_positions_ajuste,p  = plt.hist(S2_dif_ajustado_por_dif, bins=25, color='C1', normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma
x_gaussiana=np.linspace(mu_ajuste_Dif_por_Dif-3.5*sigma_ajuste_Diff_por_Dif,500,num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste


#gaussiana=mlab.normpdf(x_gaussiana, mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)*N_ajuste_Diff_por_Dif*bin_size_ajuste # dibujo la gaussiana que corresponde al histograma
gaussiana3=(1/np.sqrt(2*np.pi*(sigma_ajuste_Diff_por_Dif**2)))*np.exp((-.5)*((x_gaussiana-mu_ajuste_Dif_por_Dif)/(sigma_ajuste_Diff_por_Dif))**2)

#plt.plot(x_gaussiana,gaussiana,'r--', linewidth=2, label='Diff y Bind\n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)) #grafico la gaussiana
plt.plot(x_gaussiana,gaussiana3,'r--', linewidth=2, label='Proceso: D \n Ajuste: D \n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)) #grafico la gaussiana
plt.legend()
#==============================================================================    
#                               Proceso Difusion
#                               Ajuste: Difusion y binding
#==============================================================================    
ax2 =fig.add_subplot(2,2,2)
ax2.set_xlabel('$S^2$')
#ax2.set_ylabel('Proceso:\n Difusion')
ax2.set_title('Ajuste:\n Difusion y Binding', fontsize=20)

#bins_ajuste = np.arange(min(lista_S_2), max(lista_S_2)) # le pongo los bins a mano

n,bin_positions_ajuste,p  = plt.hist(S2_dif_ajustado_por_dif_y_bind, bins=25,color='C2', normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma
x_gaussiana=np.linspace(mu_ajuste_Dif_por_Dif_y_Bind-3.5*sigma_ajuste_Dif_por_Dif_y_Bind,500,num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste


#gaussiana=mlab.normpdf(x_gaussiana, mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)*N_ajuste_Diff_por_Dif*bin_size_ajuste # dibujo la gaussiana que corresponde al histograma
gaussiana3=(1/np.sqrt(2*np.pi*(sigma_ajuste_Dif_por_Dif_y_Bind**2)))*np.exp((-.5)*((x_gaussiana-mu_ajuste_Dif_por_Dif_y_Bind)/(sigma_ajuste_Dif_por_Dif_y_Bind))**2)

#plt.plot(x_gaussiana,gaussiana,'r--', linewidth=2, label='Diff y Bind\n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)) #grafico la gaussiana
plt.plot(x_gaussiana,gaussiana3,'g--', linewidth=2, label='Proceso: D \n Ajuste: D y B \n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Dif_por_Dif_y_Bind, sigma_ajuste_Dif_por_Dif_y_Bind)) #grafico la gaussiana
plt.legend()
#==============================================================================    
#                               Proceso Difusion y Binding
#                               Ajuste: Difusion
#==============================================================================    
ax3 =fig.add_subplot(2,2,3)
ax3.set_xlabel('$S^2$ \n Ajuste: Difusion', fontsize=20)
ax3.set_ylabel('Proceso:\n Difusion y Binding')
#ax1.set_title('', )

#bins_ajuste = np.arange(min(lista_S_2), max(lista_S_2)) # le pongo los bins a mano

n,bin_positions_ajuste,p  = plt.hist(S2_dif_y_bind_ajustado_por_dif, bins=25,color='C0', normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma
x_gaussiana=np.linspace(mu_ajuste_Dif_y_Bind_por_Dif-3.5*sigma_ajuste_Dif_y_Bind_por_Dif,500,num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste


#gaussiana=mlab.normpdf(x_gaussiana, mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)*N_ajuste_Diff_por_Dif*bin_size_ajuste # dibujo la gaussiana que corresponde al histograma
gaussiana3=(1/np.sqrt(2*np.pi*(sigma_ajuste_Dif_y_Bind_por_Dif**2)))*np.exp((-.5)*((x_gaussiana-mu_ajuste_Dif_y_Bind_por_Dif)/(sigma_ajuste_Dif_y_Bind_por_Dif))**2)

#plt.plot(x_gaussiana,gaussiana,'r--', linewidth=2, label='Diff y Bind\n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)) #grafico la gaussiana
plt.plot(x_gaussiana,gaussiana3,'b--', linewidth=2, label='Proceso: D y B \n Ajuste: D  \n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Dif_y_Bind_por_Dif, sigma_ajuste_Dif_y_Bind_por_Dif)) #grafico la gaussiana
plt.legend()
#==============================================================================    
#                               Proceso Difusion y Binding
#                               Ajuste: Difusion y Binding
#==============================================================================    
ax4 =fig.add_subplot(2,2,4)
ax4.set_xlabel('$S^2$ \n Ajuste: Difusion y Binding', fontsize=20)
#ax4.set_ylabel('Proceso:\n Difusion y Binding')

#bins_ajuste = np.arange(min(lista_S_2), max(lista_S_2)) # le pongo los bins a mano

n,bin_positions_ajuste,p  = plt.hist(S2_dif_y_bind_ajustado_por_dif_y_bind, bins=25,color = 'C4', normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma
x_gaussiana=np.linspace(mu_ajuste_Dif_y_Bind_por_Dif_y_Bind-3.5*sigma_ajuste_Dif_y_Bind_por_Dif_y_Bind,500,num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste

#gaussiana=mlab.normpdf(x_gaussiana, mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)*N_ajuste_Diff_por_Dif*bin_size_ajuste # dibujo la gaussiana que corresponde al histograma
gaussiana3=(1/np.sqrt(2*np.pi*(sigma_ajuste_Dif_y_Bind_por_Dif_y_Bind**2)))*np.exp((-.5)*((x_gaussiana-mu_ajuste_Dif_y_Bind_por_Dif_y_Bind)/(sigma_ajuste_Dif_y_Bind_por_Dif_y_Bind))**2)

#plt.plot(x_gaussiana,gaussiana,'r--', linewidth=2, label='Diff y Bind\n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)) #grafico la gaussiana
plt.plot(x_gaussiana,gaussiana3,'m--', linewidth=2, label='Proceso: D y B \n Ajuste: D y B\n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Dif_y_Bind_por_Dif_y_Bind, sigma_ajuste_Dif_y_Bind_por_Dif_y_Bind)) #grafico la gaussiana

plt.legend()
plt.show()



