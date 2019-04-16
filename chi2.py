# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:03:38 2019

@author: Sallaberry Ignacio
"""
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab

with open('chi2_simFCS.txt') as fobj:
        chi2_simFCS = fobj.read()

lista_simFCS = []
j=0
tabla_chi2_simFCS = re.split('\n', chi2_simFCS)

while j<100:
    
    lista_simFCS.append(float(tabla_chi2_simFCS[j]))
    j+=1


j=1
lista_S_2 = []

while j<101:
    with open('sim{}-5-4-19-AJUSTE.txt'.format(j)) as fobj:
        datos_AJUSTE = fobj.read()
    with open('sim{}-5-4-19-DATA.txt'.format(j)) as fobj:
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
        i+=3 
        
    #guardo chi2 calculado para la medicion j
    lista_S_2.append(S_2)#/(len(datos_final_DATA)-k))
    j+=1


##---->  puedo calcular el mu y el sigma desde un ajuste d los datos
mu, std = norm.fit(lista_S_2)    
print(mu,std)

##para los datos manejados a mano
mu_ajuste=round(np.mean(lista_S_2),2) # media
sigma_ajuste=round(np.std(lista_S_2),2) #desviación estándar     std = sqrt(mean(abs(x - x.mean())**2))
N_ajuste=len(lista_S_2) # número de mediciones
std_err_ajuste = sigma_ajuste / N_ajuste # error estándar

##para los datos del simFCS
mu_simFCS=round(np.mean(lista_simFCS),2) # media 
sigma_simFCS=round(np.std(lista_simFCS),2) #desviación estándar
N_simFCS=len(lista_simFCS) # número de cuentas
std_err_simFCS = sigma_simFCS / N_simFCS # error estándar

###grafico los dos histogramas
bins = np.arange(min(lista_S_2), max(lista_S_2), np.mean(lista_S_2)/10) # fixed bin size
plt.figure(1)
n,bin_positions,p  = plt.hist(lista_S_2, bins=bins)
plt.title('Distribucion $S^2$ - $(Datos - Ajuste)^2$ sin pesos \n $\mu$= {} - $\sigma$ = {}'.format(mu_ajuste, sigma_ajuste),fontsize=18)
plt.xlabel('$S^2$',fontsize=18)
plt.ylabel('freq',fontsize=18)
plt.tick_params(labelsize=13)

bins = np.arange(min(lista_simFCS), max(lista_simFCS), np.mean(lista_simFCS)/10) # fixed bin size
plt.figure(2)
n,bin_positions,p  = plt.hist(lista_simFCS, bins=bins)
plt.title('Distribucion $S^2$ - $simFCS$ sin pesos \n $\mu$= {} - $\sigma$ = {}'.format(mu_simFCS, sigma_simFCS),fontsize=18)

plt.xlabel('$S^2$',fontsize=18)
plt.ylabel('freq',fontsize=18)
plt.tick_params(labelsize=13)

plt.show()
  

##esto es un dibujo de la gaussiana con el mu y el sigma definido mas arriba
#lista_S_2_gaussiana=np.linspace(mu-5*sigma,mu+5*sigma,num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste
#bin_size=bin_positions[1]-bin_positions[0] # calculo el ancho de los bins del histograma
#gaussiana=mlab.normpdf(lista_S_2_gaussiana, mu, sigma)*N*bin_size # calculo la gaussiana que corresponde al histograma
#
#plt.plot(lista_S_2_gaussiana,gaussiana,'r--', linewidth=2, label='ajuste 1') #grafico la gaussiana

###### para guardar los datos
#with open('Scuadrado_que_sale_de_ajuste-datos.txt','w') as f:
#    for valor in lista_S_2:
#        #como el archivo "chi2.txt" no existe python lo va a crear
#        #si el archivo ya existiese entonces python lo reescribirá
#        
#        #entonces: python crea(o reescribe) el archivo
#        #          Lo abre
#        #          Lo escribe
#        f.write(str(valor))
#        f.write('\n') #le decimos que luego de escribir un elemento de la lista oceanos, que empiece una nueva linea
#

