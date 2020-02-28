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

#with open('C:\\Users\\ETCasa\\Desktop\\Datos para calcular S2\\S2 calculado de ajustar con python __ Ajuste de proceso de difusion pura por modelo de DIFUSION_y_BINDING.txt') as fobj:
#    Proceso_dif_ajustado_por_dify_bind = fobj.read()
#S2_dif_ajustado_por_dif_y_bind = re.split('\n', Proceso_dif_ajustado_por_dify_bind)
#S2_dif_ajustado_por_dif_y_bind.remove('')
#S2_dif_ajustado_por_dif_y_bind = [float(i)*3969 for i in S2_dif_ajustado_por_dif_y_bind]

#with open('C:\\Users\\ETCasa\\Desktop\\Datos para calcular S2\\S2 calculado de ajustar con python __ Ajuste de proceso de difusion y binding por modelo de DIFUSION.txt') as fobj:
#    Proceso_dif_y_bind_ajustado_por_dif = fobj.read()
#S2_dif_y_bind_ajustado_por_dif = re.split('\n', Proceso_dif_y_bind_ajustado_por_dif )
#S2_dif_y_bind_ajustado_por_dif .remove('')
#S2_dif_y_bind_ajustado_por_dif = [float(i)*3969 for i in S2_dif_y_bind_ajustado_por_dif]

#with open('C:\\Users\\ETCasa\\Desktop\\Datos para calcular S2\\S2 calculado de ajustar con python __ Ajuste de proceso de difusion y binding por modelo de DIFUSION_y_BINDING.txt') as fobj:
#    Proceso_dif_y_bind_ajustado_por_dify_bind = fobj.read()
#    
#S2_dif_y_bind_ajustado_por_dif_y_bind = re.split('\n', Proceso_dif_y_bind_ajustado_por_dify_bind )
#S2_dif_y_bind_ajustado_por_dif_y_bind.remove('')
#S2_dif_y_bind_ajustado_por_dif_y_bind = [float(i)*3969 for i in S2_dif_y_bind_ajustado_por_dif_y_bind]

#==============================================================================
# PARAMETROS DE TAMAÑAO DE LETRA DE LOS GRAFICOS
#==============================================================================
plt.close('all') # amtes de graficar, cierro todos las figuras que estén abiertas

SMALL_SIZE = 25
MEDIUM_SIZE = 31
BIGGER_SIZE = 34

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
#                           #####     HISTOGRAMAS  graficados en figura de 2x2  #####
#==============================================================================    

#==============================================================================
#                           #####     Funcion Gaussiana  #####
#==============================================================================  
def Gaussiana(mu,sigma):
     
    x_inicial = mu-3.5*sigma
    x_final = mu+3.5*sigma
    x_gaussiana=np.linspace(x_inicial,x_final,num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste
   
    gaussiana3=(1/np.sqrt(2*np.pi*(sigma**2)))*np.exp((-.5)*((x_gaussiana-mu)/(sigma))**2)

    return (x_gaussiana,gaussiana3)


fig = plt.figure(figsize=(11,10))
bineado = 10
#==============================================================================    
#                               Proceso Binding
#                               Ajuste: Binding
#==============================================================================    
ax1 =fig.add_subplot(2,2,1)
#bins_ajuste = np.arange(min(lista_S_2), max(lista_S_2)) # le pongo los bins a mano
ax1.set_xlabel('$S^2$')
ax1.set_ylabel('Proceso:\n Asociacion-Disociación\n')
ax1.set_title('Ajuste:\n Asociacion-Disociación')#, fontsize=20)
n,bin_positions_ajuste,p  = plt.hist(S2_binding_puro_ajustado_por_bind, bins=bineado, color='lightsalmon', normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma

plt.plot(Gaussiana(mu_ajuste_BindingPURO_por_BindingPURO,sigma_ajuste_BindingPURO_por_BindingPURO)[0],
         Gaussiana(mu_ajuste_BindingPURO_por_BindingPURO,sigma_ajuste_BindingPURO_por_BindingPURO)[1],
         '--', color='tomato',
         label='Proceso: Asociacion-Disociación \n Ajuste: Asociacion-Disociación \n $\mu$= {:.3f} - $\sigma$ = {:.3f}'.format(mu_ajuste_BindingPURO_por_BindingPURO, sigma_ajuste_BindingPURO_por_BindingPURO)
         )
ax1.tick_params(which='minor', length=8, width=3.5)
ax1.tick_params(which='major', length=6, width=3)
#figManager = plt.get_current_fig_manager()  ####   esto y la linea de abajo me maximiza la ventana de la figura
#figManager.window.showMaximized()           ####   esto y la linea de arriba me maximiza la ventana de la figura
plt.show()
plt.legend()
#==============================================================================    
#                               Proceso Binding
#                               Ajuste: Difusion
#==============================================================================    
ax1 =fig.add_subplot(2,2,2)
#bins_ajuste = np.arange(min(lista_S_2), max(lista_S_2)) # le pongo los bins a mano
ax1.set_xlabel('$S^2$')
ax1.set_ylabel('Proceso:\n Asociacion-Disociación \n')
ax1.set_title('Ajuste:\n Difusion')#, fontsize=20)
n,bin_positions_ajuste,p  = plt.hist(S2_binding_puro_ajustado_por_dif, bins=bineado, color='indianred', normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma

plt.plot(Gaussiana(mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif)[0],
         Gaussiana(mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif)[1],
         '--',color='#8f1402', label='Proceso: Asociacion-Disociación \n Ajuste: Dif \n $\mu$= {:.3f} - $\sigma$ = {:.3f}'.format(mu_ajuste_BindingPURO_por_Dif, sigma_ajuste_BindingPURO_por_Dif)
         )
ax1.tick_params(which='minor', length=8, width=3.5)
ax1.tick_params(which='major', length=6, width=3)

plt.legend()
#==============================================================================    
#                               Proceso Difusion
#                               Ajuste: Binding
#==============================================================================    
ax1 =fig.add_subplot(2,2,3)
#bins_ajuste = np.arange(min(lista_S_2), max(lista_S_2)) # le pongo los bins a mano
ax1.set_xlabel('$S^2$')
ax1.set_ylabel('Proceso:\n Difusion \n')
ax1.set_title('Ajuste:\n Asociacion-Disociación')#, fontsize=20)
n,bin_positions_ajuste,p  = plt.hist(S2_dif_ajustado_por_bind, bins=bineado, color='C0', normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma

plt.plot(Gaussiana(mu_ajuste_Dif_por_Bind,sigma_ajuste_Diff_por_Bind)[0],
         Gaussiana(mu_ajuste_Dif_por_Bind,sigma_ajuste_Diff_por_Bind)[1],
         'b--', label='Proceso: Dif \n Ajuste: Asociacion-Disociación \n $\mu$= {:.3f} - $\sigma$ = {:.3f}'.format(mu_ajuste_Dif_por_Bind, sigma_ajuste_Diff_por_Bind)
         )
ax1.tick_params(which='minor', length=8, width=3.5)
ax1.tick_params(which='major', length=6, width=3)
plt.legend()
#==============================================================================    
#                               Proceso Difusion
#                               Ajuste: Difusion
#==============================================================================    
ax1 =fig.add_subplot(2,2,4)
ax1.set_xlabel('$S^2$')
ax1.set_ylabel('Proceso:\n Difusion \n')
ax1.set_title('Ajuste:\n Difusion')#, fontsize=20)
n,bin_positions_ajuste,p  = plt.hist(S2_dif_ajustado_por_dif, bins=bineado, color='mediumaquamarine', normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma

plt.plot(Gaussiana(mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif)[0],
         Gaussiana(mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif)[1],
         '--',color='forestgreen', label='Proceso: Dif \n Ajuste: Dif \n $\mu$= {:.3f} - $\sigma$ = {:.3f}'.format(mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)
         )
ax1.tick_params(which='minor', length=8, width=3.5)
ax1.tick_params(which='major', length=6, width=3)
plt.legend()
plt.gcf().subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.show()







##==============================================================================
##                           #####     HISTOGRAMAS  Superpuestos  #####
##==============================================================================    
#fig = plt.figure()


##==============================================================================    
##                               Proceso Difusion y Binding
##                               Ajuste: Difusion
##==============================================================================    
##bins_ajuste = np.arange(min(lista_S_2), max(lista_S_2)) # le pongo los bins a mano
#plt.figure()
#n,bin_positions_ajuste,p  = plt.hist(S2_dif_y_bind_ajustado_por_dif, bins=bineado,color='C4', normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
#bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma
#x_gaussiana=np.linspace(mu_ajuste_Dif_y_Bind_por_Dif-3.5*sigma_ajuste_Dif_y_Bind_por_Dif,225,num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste
#
#
##gaussiana=mlab.normpdf(x_gaussiana, mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)*N_ajuste_Diff_por_Dif*bin_size_ajuste # dibujo la gaussiana que corresponde al histograma
#gaussiana3=(1/np.sqrt(2*np.pi*(sigma_ajuste_Dif_y_Bind_por_Dif**2)))*np.exp((-.5)*((x_gaussiana-mu_ajuste_Dif_y_Bind_por_Dif)/(sigma_ajuste_Dif_y_Bind_por_Dif))**2)
#
##plt.plot(x_gaussiana,gaussiana,'r--', linewidth=2, label='Diff y Bind\n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)) #grafico la gaussiana
#plt.plot(x_gaussiana,gaussiana3,'--', color='#cb416b', linewidth=3, label='Proceso: D y B \n Ajuste: D  \n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Dif_y_Bind_por_Dif, sigma_ajuste_Dif_y_Bind_por_Dif)) #grafico la gaussiana
#plt.legend()
#==============================================================================    
#                               Proceso Difusion y Binding
#                               Ajuste: Difusion y Binding
#==============================================================================    

#n,bin_positions_ajuste,p  = plt.hist(S2_dif_y_bind_ajustado_por_dif_y_bind, bins=bineado,color = 'C0', normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
#bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma
#x_gaussiana=np.linspace(mu_ajuste_Dif_y_Bind_por_Dif_y_Bind-3.5*sigma_ajuste_Dif_y_Bind_por_Dif_y_Bind,225,num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste
#
##gaussiana=mlab.normpdf(x_gaussiana, mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)*N_ajuste_Diff_por_Dif*bin_size_ajuste # dibujo la gaussiana que corresponde al histograma
#gaussiana3=(1/np.sqrt(2*np.pi*(sigma_ajuste_Dif_y_Bind_por_Dif_y_Bind**2)))*np.exp((-.5)*((x_gaussiana-mu_ajuste_Dif_y_Bind_por_Dif_y_Bind)/(sigma_ajuste_Dif_y_Bind_por_Dif_y_Bind))**2)
#
##plt.plot(x_gaussiana,gaussiana,'r--', linewidth=2, label='Diff y Bind\n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)) #grafico la gaussiana
#plt.plot(x_gaussiana,gaussiana3,'b--', linewidth=3, label='Proceso: D y B \n Ajuste: D y B\n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Dif_y_Bind_por_Dif_y_Bind, sigma_ajuste_Dif_y_Bind_por_Dif_y_Bind)) #grafico la gaussiana
#plt.xlabel('$S^2$')
#plt.ylabel('PDF')

#plt.legend()
#plt.show()
plt.figure()
#==============================================================================    
#                               Proceso Binding
#                               Ajuste: Difusion
#==============================================================================    
n,bin_positions_ajuste,p  = plt.hist(S2_binding_puro_ajustado_por_dif, bins=bineado, color='indianred', normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma

plt.plot(Gaussiana(mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif)[0],
         Gaussiana(mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif)[1],
         '--',color='#8f1402', label='Proceso: Asociacion-Disociación \n Ajuste: Dif \n $\mu$= {:.3f} - $\sigma$ = {:.3f}'.format(mu_ajuste_BindingPURO_por_Dif, sigma_ajuste_BindingPURO_por_Dif)
         )

plt.xlabel(r'$S^2$')
plt.ylabel('PDF')
plt.legend()


#==============================================================================    
#                               Proceso Binding
#                               Ajuste: Binding
#==============================================================================    
n,bin_positions_ajuste,p  = plt.hist(S2_binding_puro_ajustado_por_bind, bins=bineado, color='lightsalmon', normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma

plt.plot(Gaussiana(mu_ajuste_BindingPURO_por_BindingPURO,sigma_ajuste_BindingPURO_por_BindingPURO)[0],
         Gaussiana(mu_ajuste_BindingPURO_por_BindingPURO,sigma_ajuste_BindingPURO_por_BindingPURO)[1],
         '--', color='tomato',
         label='Proceso: Asociacion-Disociación \n Ajuste: Asociacion-Disociación \n $\mu$= {:.3f} - $\sigma$ = {:.3f}'.format(mu_ajuste_BindingPURO_por_BindingPURO, sigma_ajuste_BindingPURO_por_BindingPURO)
         )

plt.xlabel(r'$S^2$')
plt.ylabel('PDF')
plt.legend()

plt.tick_params(which='minor', length=8, width=3.5)
plt.tick_params(which='major', length=6, width=3)
plt.show()





plt.figure()
#==============================================================================    
#                               Proceso Difusion
#                               Ajuste: Difusion
#==============================================================================    
n,bin_positions_ajuste,p  = plt.hist(S2_dif_ajustado_por_dif, bins=bineado, color='mediumaquamarine', normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma

plt.plot(Gaussiana(mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif)[0],
         Gaussiana(mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif)[1],
         '--',color='forestgreen', label='Proceso: Dif \n Ajuste: Dif \n $\mu$= {:.3f} - $\sigma$ = {:.3f}'.format(mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)
         )
plt.xlabel(r'$S^2$')
plt.ylabel('PDF')
plt.legend()


#==============================================================================    
#                               Proceso Binding
#                               Ajuste: Binding
#==============================================================================    
n,bin_positions_ajuste,p  = plt.hist(S2_binding_puro_ajustado_por_bind, bins=bineado, color='lightsalmon', normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma

plt.plot(Gaussiana(mu_ajuste_BindingPURO_por_BindingPURO,sigma_ajuste_BindingPURO_por_BindingPURO)[0],
         Gaussiana(mu_ajuste_BindingPURO_por_BindingPURO,sigma_ajuste_BindingPURO_por_BindingPURO)[1],
         '--', color='tomato',
         label='Proceso: Asociacion-Disociación \n Ajuste: Asociacion-Disociación \n $\mu$= {:.3f} - $\sigma$ = {:.3f}'.format(mu_ajuste_BindingPURO_por_BindingPURO, sigma_ajuste_BindingPURO_por_BindingPURO)
         )

plt.xlabel(r'$S^2$')
plt.ylabel('PDF')
plt.legend()

plt.tick_params(which='minor', length=8, width=3.5)
plt.tick_params(which='major', length=6, width=3)
plt.show()






plt.figure()
##==============================================================================    
##                               Proceso Difusion
##                               Ajuste: Binding
##==============================================================================    
n,bin_positions_ajuste,p  = plt.hist(S2_dif_ajustado_por_bind, bins=bineado, color='steelblue', normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma

plt.plot(Gaussiana(mu_ajuste_Dif_por_Bind,sigma_ajuste_Diff_por_Bind)[0],
         Gaussiana(mu_ajuste_Dif_por_Bind,sigma_ajuste_Diff_por_Bind)[1],
         'b--', label='Proceso: Dif \n Ajuste: Asociacion-Disociación \n $\mu$= {:.3f} - $\sigma$ = {:.3f}'.format(mu_ajuste_Dif_por_Bind, sigma_ajuste_Diff_por_Bind)
         )

plt.legend()
#==============================================================================    
#                               Proceso Difusion
#                               Ajuste: Difusion
#==============================================================================    
n,bin_positions_ajuste,p  = plt.hist(S2_dif_ajustado_por_dif, bins=bineado, color='mediumaquamarine', normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma

plt.plot(Gaussiana(mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif)[0],
         Gaussiana(mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif)[1],
         '--',color='forestgreen', label='Proceso: Dif \n Ajuste: Dif \n $\mu$= {:.3f} - $\sigma$ = {:.3f}'.format(mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)
         )

plt.xlabel(r'$S^2$')
plt.ylabel('PDF')
plt.legend()
plt.gcf().subplots_adjust(bottom=0.15)

plt.tick_params(which='minor', length=8, width=3.5)
plt.tick_params(which='major', length=6, width=3)
plt.show()





plt.figure()
#==============================================================================    
#                               Proceso Binding
#                               Ajuste: Difusion
#==============================================================================    
n,bin_positions_ajuste,p  = plt.hist(S2_binding_puro_ajustado_por_dif, bins=bineado, color='indianred', normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma

plt.plot(Gaussiana(mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif)[0],
         Gaussiana(mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif)[1],
         '--',color='#8f1402', label='Proceso: Asociacion-Disociación \n Ajuste: Dif \n $\mu$= {:.3f} - $\sigma$ = {:.3f}'.format(mu_ajuste_BindingPURO_por_Dif, sigma_ajuste_BindingPURO_por_Dif)
         )

plt.xlabel(r'$S^2$')
plt.ylabel('PDF')
plt.legend()

#==============================================================================    
#                               Proceso Difusion
#                               Ajuste: Difusion
#==============================================================================    
n,bin_positions_ajuste,p  = plt.hist(S2_dif_ajustado_por_dif, bins=bineado, color='mediumaquamarine', normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma

plt.plot(Gaussiana(mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif)[0],
         Gaussiana(mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif)[1],
         '--',color='forestgreen', label='Proceso: Dif \n Ajuste: Dif \n $\mu$= {:.3f} - $\sigma$ = {:.3f}'.format(mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)
         )

plt.xlabel(r'$S^2$')
plt.ylabel('PDF')
plt.legend()
plt.gcf().subplots_adjust(bottom=0.15)

plt.tick_params(which='minor', length=8, width=3.5)
plt.tick_params(which='major', length=6, width=3)
plt.show()


#print('area_del_histo={}'.format(sum(np.diff(bin_positions_ajuste)*n)))
#print('area_de primer columna={}'.format((bin_positions_ajuste[1]-bin_positions_ajuste[0])*n[0]))
