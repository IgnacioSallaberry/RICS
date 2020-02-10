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
import matplotlib.ticker as mticker
#==============================================================================
# 
#                               DIFUSION
#
#==============================================================================
#==============================================================================
# CARGO LOS DATOS DE LAS 100 SIMULACIONES DE DIFUSIÓN DEL 5-4-19 Y CALCULO EL S2 USANDO ESTADÍSTICO A MANO
#==============================================================================
#23hs_cell4_rics_cyto_DIFFERENCE_DIFUSION
with open('D:\\Nacho Sallaberry - Next Cloud\\Datos manu\\ANALISIS DATOS MANU 6HS DENV CELULA1 RICS NUCLEO\\DIFFERENCE DIFUSION 6HS DENV CELULA1 RICS NUCLEO.txt') as fobj:
    datos_difusion = fobj.read()
datos_difusion = re.split('\t|\n', datos_difusion)
datos_difusion.remove('X')
datos_difusion.remove('Y')
datos_difusion.remove('Z')
datos_difusion.remove('')
datos_difusion.remove('')
    


with open('D:\\Nacho Sallaberry - Next Cloud\\Datos manu\\ANALISIS DATOS MANU 6HS DENV CELULA1 RICS NUCLEO\\DIFFERENCE BINDING 6HS DENV CELULA1 RICS NUCLEO.txt') as fobj:
    datos_binding= fobj.read()
datos_binding= re.split('\t|\n', datos_binding)
datos_binding.remove('X')
datos_binding.remove('Y')
datos_binding.remove('Z')
datos_binding.remove('')
datos_binding.remove('')


DIFERENCE_DIF=[]
DIFERENCE_BIND=[]
i=0
while i< len(datos_difusion ):
    DIFERENCE_DIF.append(float(datos_difusion [i+1]))
    DIFERENCE_BIND.append(float(datos_binding [i+1]))
    
    i+=3


plt.close('all')
#==============================================================================
# 
#                         HISTOGRAMAS
#
#==============================================================================
#==============================================================================
# CALCULAR LA MEDIA, LA DESVIACIÓN ESTÁNDAR DE LOS DATOS y EL NÚMERO TOTAL DE CUENTAS
#==============================================================================

##para los datos Diff
mu_ajuste_Diff=np.mean(DIFERENCE_DIF) # media
sigma_ajuste_Diff=np.std(DIFERENCE_DIF) #desviación estándar     std = sqrt(mean(abs(x - x.mean())**2))
N_ajuste_Diff=len(DIFERENCE_DIF) # número de mediciones
std_err_ajuste_Diff = sigma_ajuste_Diff / N_ajuste_Diff # error estándar

##para los datos Diff y Bind
mu_ajuste_Bind = np.mean(DIFERENCE_BIND) # media 
sigma_ajuste_Bind = np.std(DIFERENCE_BIND) #desviación estándar
N_ajuste_Bind=len(DIFERENCE_BIND) # número de cuentas
std_err_ajuste_Bind = sigma_ajuste_Bind / N_ajuste_Bind # error estándar


#==============================================================================
# PARAMETROS DE TAMAÑAO DE LETRA DE LOS GRAFICOS
#==============================================================================
plt.close('all') # amtes de graficar, cierro todos las figuras que estén abiertas
SMALL_SIZE = 28
MEDIUM_SIZE = 36
BIGGER_SIZE = 39
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=22)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def Gaussiana(mu,sigma):
     
    x_inicial = mu-3.5*sigma
    x_final = mu+3.5*sigma
    x_gaussiana=np.linspace(x_inicial,x_final,num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste
   
    gaussiana3=(1/np.sqrt(2*np.pi*(sigma**2)))*np.exp((-.5)*((x_gaussiana-mu)/(sigma))**2)

    return (x_gaussiana,gaussiana3)


#bineado = 'fd'
bineado = 50
##==============================================================================    
##                               Diferencias ajuste por Difusion
##==============================================================================    
n,bin_positions_ajuste,p  = plt.hist(DIFERENCE_DIF, bins=bineado, color='lightsalmon', density=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma

plt.plot(Gaussiana(mu_ajuste_Diff,sigma_ajuste_Diff)[0],
         Gaussiana(mu_ajuste_Diff,sigma_ajuste_Diff)[1],
         '--', color='tomato', label='Ajuste: Difusión \n $\mu$= {:10.3E} $\\sigma$ = {:10.3E}'.format(mu_ajuste_Diff, sigma_ajuste_Diff)

         )
#plt.errorbar(0.5*(bin_positions_ajuste[1:]+bin_positions_ajuste[:-1]), n, yerr=np.sqrt(n),
#             fmt='None', ecolor='r', elinewidth=3, capthick=10)
#plt.bar(bin_positions_ajuste,n, yerror=np.sqrt(n))#, alpha=0.5, ecolor='black', capsize=10)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.tick_params(which='minor', length=4, width=1.5)
plt.tick_params(which='major', length=7, width=2)

##==============================================================================    
##                               Diferencias ajuse por Binding
##==============================================================================    
n,bin_positions_ajuste,p  = plt.hist(DIFERENCE_BIND, bins=bineado, color='mediumaquamarine', density=True)    #esta funcion, ademas de graficar devuelve parametros del histograma, que guardamos en las variables n,bin_positions,p
bin_size_ajuste=bin_positions_ajuste[1]-bin_positions_ajuste[0] # calculo el ancho de los bins del histograma

plt.plot(Gaussiana(mu_ajuste_Bind,sigma_ajuste_Bind)[0],
         Gaussiana(mu_ajuste_Bind,sigma_ajuste_Bind)[1],
         '--',color='forestgreen', label= 'Ajuste: Asociacion-Disociación \n $\mu$= {:10.3E} $\\sigma$ = {:10.3E}'.format(mu_ajuste_Bind, sigma_ajuste_Bind)
         )
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#plt.errorbar(0.5*(bin_positions_ajuste[1:]+bin_positions_ajuste[:-1]), n,  yerr=np.sqrt(n), fmt='None', ecolor='b', elinewidth=3)
plt.xlabel(r'$Datos - Ajuste$')
plt.ylabel('')
plt.legend()
plt.title('Histograma normalizado')
plt.tick_params(which='minor', length=4, width=1.5)
plt.tick_params(which='major', length=7, width=2)
plt.show()
plt.tight_layout()
#plt.savefig('C:\\Users\\ETCasa\\Desktop\\Histogramas datos manu')
