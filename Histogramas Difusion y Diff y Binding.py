# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:45:10 2019

@author: LEC
"""


#==============================================================================
# COMPARACION DE HISTOGRAMAS PARA S2 = Ajuste - Datos 
#==============================================================================


import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab

#==============================================================================
# 
#                               DIFUSION
#
#==============================================================================
#==============================================================================
# CARGO LOS DATOS DE LAS 100 SIMULACIONES DE DIFUSIÓN DEL 5-4-19 Y CALCULO EL S2 USANDO ESTADÍSTICO A MANO
#==============================================================================

j=1
lista_S_2_Diff = []
while j<101:
    with open('C:\\Users\\LEC\\Nacho\\5-4-19\\histograma\\ajuste\\sim{}-5-4-19-DATA.txt'.format(j)) as fobj:
        datos_AJUSTE = fobj.read()
    with open('C:\\Users\\LEC\\Nacho\\5-4-19\\histograma\\ajuste\\sim{}-5-4-19-AJUSTE.txt'.format(j)) as fobj:
        datos_DATA = fobj.read()

    tabla_AJUSTE = re.split('\t|\n', datos_AJUSTE)
    tabla_DATA = re.split('\t|\n', datos_DATA)

    #Listas con los datos que voy a usar para calcular el chi2
    datos_final_DATA=[]
    datos_final_AJUSTE=[]
    
        
    #Calculo de S_2= S1+S2+S3+....+S100    donde Sj es el S2 de las j-esimas mediciones
    i=4
    S_2=0.0 #inicializo S_2 = 0
    k=0 #parametros libres
    while i<len(tabla_DATA)-2:
        datos_final_DATA.append(float(tabla_DATA[i]))
        datos_final_AJUSTE.append(float(tabla_AJUSTE[i]))
            
        S_2=S_2+((float(tabla_DATA[i])-float(tabla_AJUSTE[i]))**2)#S cuadrado de la sim_j
#        S_2=S_2+((float(tabla_DATA[i])-float(tabla_AJUSTE[i])))    #S cuadrado de la sim_j
        i+=3 
        
    #guardo chi2 calculado
    lista_S_2_Diff.append(S_2) #/(len(datos_final_DATA)-k))
    j+=1


#==============================================================================
# 
#                         DIFUSION Y BINDING
#
#==============================================================================
#==============================================================================
# CARGO LOS DATOS DE LAS 100 SIMULACIONES DE DIFUSIÓN Y BINDING DEL 5-4-19 Y CALCULO EL S2 USANDO ESTADÍSTICO A MANO
#==============================================================================

j=1  #ahora j es el numero de simulacion
lista_S_2_Diff_y_Bind = []

while j<101:
    with open('C:\\Users\\LEC\\Desktop\\Distribuciones S2\\S2 una especie y dos procesos\\sim{}_una_especie_AJUSTE.txt'.format(j)) as fobj:
        datos_AJUSTE = fobj.read()
    with open('C:\\Users\\LEC\\Desktop\\Distribuciones S2\\S2 una especie y dos procesos\\sim{}_una_especie_DATA.txt'.format(j)) as fobj:
        datos_DATA = fobj.read()

    tabla_AJUSTE = re.split('\t|\n', datos_AJUSTE)
    tabla_DATA = re.split('\t|\n', datos_DATA)

    #Listas con los datos que voy a usar para calcular el chi2
    datos_final_DATA=[]
    datos_final_AJUSTE=[]
    
        
    #Calculo de S_2= S1+S2+S3+....+S100    donde Sj es el S2 de las j-esimas mediciones
    i=4
    S_2=0.0 #inicializo S_2 = 0
    k=0 #parametros libres
    while i<len(tabla_DATA)-2:
        datos_final_DATA.append(float(tabla_DATA[i]))
        datos_final_AJUSTE.append(float(tabla_AJUSTE[i]))
            
        S_2=S_2+((float(tabla_DATA[i])-float(tabla_AJUSTE[i]))**2)    #S cuadrado de la sim_j
#        S_2=S_2+((float(tabla_DATA[i])-float(tabla_AJUSTE[i])))    #S cuadrado de la sim_j

        i+=3 
        
    #guardo chi2 calculado para la medicion j
    lista_S_2_Diff_y_Bind.append(S_2)#/(len(datos_final_DATA)-k))
    j+=1
    
    

#==============================================================================
# 
#                         HISTOGRAMAS
#
#==============================================================================
#==============================================================================
# CALCULAR LA MEDIA, LA DESVIACIÓN ESTÁNDAR DE LOS DATOS y EL NÚMERO TOTAL DE CUENTAS
#==============================================================================

##para los datos Diff
mu_ajuste_Diff=np.mean(lista_S_2_Diff) # media
sigma_ajuste_Diff=np.std(lista_S_2_Diff) #desviación estándar     std = sqrt(mean(abs(x - x.mean())**2))
N_ajuste_Diff=len(lista_S_2_Diff) # número de mediciones
std_err_ajuste_Diff = sigma_ajuste_Diff / N_ajuste_Diff # error estándar

##para los datos Diff y Bind
mu_ajuste_Diff_y_Bind = np.mean(lista_S_2_Diff_y_Bind) # media 
sigma_ajuste_Diff_y_Bind = np.std(lista_S_2_Diff_y_Bind) #desviación estándar
N_ajuste_Diff_y_Bind=len(lista_S_2_Diff_y_Bind) # número de cuentas
std_err_ajuste_Diff_y_Bind = sigma_ajuste_Diff_y_Bind / N_ajuste_Diff_y_Bind # error estándar


#==============================================================================
# PARAMETROS DE TAMAÑAO DE LETRA DE LOS GRAFICOS
#==============================================================================
plt.close('all') # amtes de graficar, cierro todos las figuras que estén abiertas

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title    

plt.title('Distribucion $S^2$ = $(Datos - Ajuste)^2$')
plt.xlabel('$S^2$')
plt.ylabel('Frecuencia')
#plt.tick_params(labelsize=13)




#==============================================================================
# GRAFICAR EL HISTOGRAMA DE LOS DATOS    DIFUSION Y BINDING
#==============================================================================
#bins_ajuste = np.arange(min(lista_S_2), max(lista_S_2)) # le pongo los bins a mano

n,bin_positions_ajuste,p  = plt.hist(lista_S_2_Diff_y_Bind, bins=10, normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma
#x_gaussiana=np.linspace(mu_ajuste_Diff_y_Bind-3.5*sigma_ajuste_Diff_y_Bind,mu_ajuste_Diff_y_Bind+10*sigma_ajuste_Diff_y_Bind,num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste
x_gaussiana=np.linspace(mu_ajuste_Diff_y_Bind-3.5*sigma_ajuste_Diff_y_Bind,300,num=1000) # armo una lista de puntos donde quiero graficar la distribución de ajuste


#gaussiana=mlab.normpdf(x_gaussiana, mu_ajuste_Diff_y_Bind, sigma_ajuste_Diff_y_Bind)*N_ajuste_Diff_y_Bind*bin_size_ajuste # dibujo la gaussiana que corresponde al histograma
gaussiana3=(1/np.sqrt(2*np.pi*(sigma_ajuste_Diff_y_Bind**2)))*np.exp((-.5)*((x_gaussiana-mu_ajuste_Diff_y_Bind)/(sigma_ajuste_Diff_y_Bind))**2)

#plt.plot(x_gaussiana,gaussiana,'r--', linewidth=2, label='Diff y Bind\n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Diff_y_Bind, sigma_ajuste_Diff_y_Bind)) #grafico la gaussiana
plt.plot(x_gaussiana,gaussiana3,'b--', linewidth=2, label='Difusión y asociación - disociación \n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Diff_y_Bind, sigma_ajuste_Diff_y_Bind)) #grafico la gaussiana




#==============================================================================
# GRAFICAR EL HISTOGRAMA DE LOS DATOS    DIFUSION
#==============================================================================
#bins_ajuste = np.arange(min(lista_S_2), max(lista_S_2)) # le pongo los bins a mano

n,bin_positions_Diff,p  = plt.hist(lista_S_2_Diff, bins=10, normed=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_Diff = bin_positions_Diff[1]-bin_positions_Diff[0] # calculo el ancho de los bins del histograma
#x_gaussiana=np.linspace(mu_ajuste_Diff-3.5*sigma_ajuste_Diff,mu_ajuste_Diff+10*sigma_ajuste_Diff,num=1000) # armo una lista de puntos donde quiero graficar la distribución de ajuste
x_gaussiana=np.linspace(mu_ajuste_Diff-3.5*sigma_ajuste_Diff,300,num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste

#gaussiana=mlab.normpdf(x_gaussiana, mu_ajuste_Diff, sigma_ajuste_Diff)*N_ajuste_Diff*bin_size_Diff # dibujo la gaussiana que corresponde al histograma
gaussiana2=(1/np.sqrt(2*np.pi*(sigma_ajuste_Diff**2)))*np.exp((-.5)*((x_gaussiana-mu_ajuste_Diff)/(sigma_ajuste_Diff))**2)
#plt.plot(x_gaussiana,gaussiana,'b--', linewidth=2, label='Diff \n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Diff, sigma_ajuste_Diff)) #grafico la gaussiana
plt.plot(x_gaussiana,gaussiana2,'r--', linewidth=2, label='Difusión \n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Diff, sigma_ajuste_Diff)) #grafico la gaussiana




#plt.legend('Difusión y asociación - disociación \n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Diff_y_Bind, sigma_ajuste_Diff_y_Bind))

plt.legend()
plt.show()

#plt.savefig('C:\\Users\\LEC\\Desktop\\Poster TOPFOT 2019\\Histogramas S2')
