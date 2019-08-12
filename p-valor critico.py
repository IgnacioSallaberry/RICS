# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 11:10:12 2019

@author: ETCasa
"""

import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.integrate import simps, quadrature, trapz
from scipy.stats import norm

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

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title 


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
def Gaussiana(mu,sigma):
     
    x_inicial = mu-3.5*sigma
    x_final = mu+3.5*sigma
    x_gaussiana=np.linspace(x_inicial,x_final,num=100) # armo una lista de puntos donde quiero graficar la distribución de ajuste
   
    gaussiana3=(1/np.sqrt(2*np.pi*(sigma**2)))*np.exp((-.5)*((x_gaussiana-mu)/(sigma))**2)

    return (x_gaussiana,gaussiana3)

#==============================================================================
#                           #####     Graficos  #####
#==============================================================================   
plt.figure()
plt.plot(Gaussiana(mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif)[0],
         Gaussiana(mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif)[1],
         '--',color='forestgreen', linewidth=3, 
         label='Proceso: Dif \n Ajuste: Dif \n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)
         )

plt.plot(Gaussiana(mu_ajuste_Dif_por_Bind,sigma_ajuste_Diff_por_Bind)[0],
         Gaussiana(mu_ajuste_Dif_por_Bind,sigma_ajuste_Diff_por_Bind)[1],
         'b--', linewidth=3, label='Proceso: Dif \n Ajuste: Bind \n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)
         )


plt.legend()
plt.show()

A=Gaussiana(mu_ajuste_Dif_por_Dif,sigma_ajuste_Diff_por_Dif)
print( simps(A[1],A[0]) )

print( trapz(A[1],A[0],max(A[0])/100) )





#==============================================================================
#                           #####     p-valor Critico      #####
#==============================================================================   

def p_valor(S2_1, mu_1, sigma_1, S2_2, mu_2, sigma_2):
    S2_critico = mu_1
    
    p_valor1=[]
    p_valor2=[]
    S2_X = []
    
    S2_CRITICO='hola'
    
    S2_critico=mu_1
    
    integral_1 = 1 - norm.cdf(S2_critico, mu_1, sigma_1)
    integral_2 = norm.cdf(S2_critico, mu_2, sigma_2)                                                            
         
    incremento = 0.0001
                                                 
    while integral_1 > 0.00001:
        S2_critico += incremento                                             
        p_valor1.append(integral_1)
        p_valor2.append(integral_2)
        S2_X.append(S2_critico)
        if integral_1 - integral_2> incremento:
            S2_CRITICO = S2_critico
        
        integral_1 = 1 - norm.cdf(S2_critico, mu_1, sigma_1)
        integral_2 = norm.cdf(S2_critico, mu_2, sigma_2) 

    return (S2_X, p_valor1, p_valor2, S2_CRITICO)
    


#==============================================================================    
#                               Proceso Difusion
#                               Ajuste: Difusion  y   BINDING
#==============================================================================        
P=p_valor(S2_dif_ajustado_por_dif, mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif, 
          S2_dif_ajustado_por_bind, mu_ajuste_Dif_por_Bind, sigma_ajuste_Diff_por_Bind)

plt.figure()
plt.semilogy(P[0],P[1],'--',color='forestgreen',label='Proceso: Dif \n Ajuste: Dif' )
plt.semilogy(P[0],P[2],'b--',label='Proceso: Dif \n Ajuste: Bind' )
plt.axvline(x=P[3],color='r')
plt.text(P[3]+0.0008,0.0007,s='$S^2$ critico = {:.5f}'.format(P[3]), color='r',rotation=90)
plt.xlabel('S2')
plt.ylabel('p-valor')
plt.legend()
plt.show()
    

#==============================================================================    
#                               Proceso BINDING
#                               Ajuste: Difusion y    BINDING
#==============================================================================    
plt.figure()
plt.plot(Gaussiana(mu_ajuste_BindingPURO_por_BindingPURO,sigma_ajuste_BindingPURO_por_BindingPURO)[0],
         Gaussiana(mu_ajuste_BindingPURO_por_BindingPURO,sigma_ajuste_BindingPURO_por_BindingPURO)[1],
         '--',color='forestgreen', linewidth=3, 
         label='Proceso: Bind \n Ajuste: Bind \n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)
         )

plt.plot(Gaussiana(mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif)[0],
         Gaussiana(mu_ajuste_BindingPURO_por_Dif,sigma_ajuste_BindingPURO_por_Dif)[1],
         'b--', linewidth=3, label='Proceso: Bind \n Ajuste: Bind \n $\mu$= {:.2f} - $\sigma$ = {:.2f}'.format(mu_ajuste_Dif_por_Dif, sigma_ajuste_Diff_por_Dif)
         )


plt.legend()
plt.show()

P=p_valor(S2_binding_puro_ajustado_por_bind, mu_ajuste_BindingPURO_por_BindingPURO, sigma_ajuste_BindingPURO_por_BindingPURO, 
          S2_binding_puro_ajustado_por_dif, mu_ajuste_BindingPURO_por_Dif, sigma_ajuste_BindingPURO_por_Dif)

plt.figure()
plt.semilogy(P[0],P[1],'--',color='forestgreen',label='Proceso: Bind \n Ajuste: Bind' )
plt.semilogy(P[0],P[2],'b--',label='Proceso: Bind \n Ajuste: Dif' )
plt.axvline(x=P[3],color='r')
plt.text(P[3]+0.0008,0.0007,s='$S^2$ critico = {:.5f}'.format(P[3]), color='r',rotation=90)
plt.xlabel('S2')
plt.ylabel('p-valor')
plt.legend()
plt.show()





