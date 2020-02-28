# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:25:06 2019
@author: Ignacio Sallaberry
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#==============================================================================
#                                  Parametros globales iniciales
#==============================================================================   
box_size = 512
roi = 128
tp = 1e-6             #seg    
tl = box_size * tp    #seg
dr = 0.05             # delta r = pixel size =  micrometros
w0 = 0.5              #radio de la PSF  = micrometros    
wz = 1.5              #alto de la PSF desde el centro  = micrometros
a = w0/wz
gamma = 0.3536
num_de_partic = 4000
v_eff=(w0**2)*wz*((np.pi)**(1.5))
concentracion = num_de_partic/((box_size*dr)**3)
N = v_eff * concentracion
print(f'N en PSF ={N}')
G_0 = 1/N
print(f'G0={G_0}')


        #gamma factor de la 3DG
#N = 0.05         #Numero total de particulas en la PSF





At= 0                 #Amplitud de triplete
#t_triplet = .16       #segundos    #Tiempo caracteristico de triplete 

Ab= 0.9                #Amplitud de binding
#t_binding= .16        #segundos     #Tiempo caracteristico de binding

D=90
#print('D = {}'.format(D))
#D=90
#D=[0.1,1,10,100,1000]
#D=np.arange(1,1050,200)   #micrones^2/seg

#==============================================================================
#                                Inicializo tau = valores del eje x
#==============================================================================    
## acá exploro distintas formas para ver como puedo hacer para tener más valores al ppio (ie: tau=0) y no tantos al final

#tau = np.arange(1e-6,1,1e-6)
#tau1 = np.logspace(1e-6, 1, num=50)
#plt.figure()
#plt.plot(tau, label='tau')
#plt.plot(tau1, label='tau1')
#plt.legend()
#plt.show()



#defino tau
#tau = np.logspace(0, 1, num=1000)/100000000
tau = np.geomspace(1e-6,10,num=1000)
#tau = np.linspace(1,10, num=100000)/1000000
#==============================================================================
#                                Defino tiempo de difusión de acuerdo a los parametros de arriba
#==============================================================================    

t_diff = w0**2/(4*D) 
print('tiempo de difusion = {} segundos'.format(t_diff))


t_binding=0.0005
#==============================================================================
#                                Inicializo T= t_diff / t_bind 
#==============================================================================    
#T=[0.01,0.1,1,10,100]
#T=[0.01,0.9,100]
#T=[0.25,1.75]
#T=[0.09]
T=[t_diff/t_binding]
print(T)

#t_binding=t_diff/T



#==============================================================================
#                                Creo el ruido para añadirle a las curvas
#==============================================================================    

#Ruido = np.random.randn(len(tau),1)

ruido1 = np.random.uniform(0,0.25, size=len(tau))
ruido2 = np.random.uniform(0,0.25, size=len(tau))
ruido3 = np.random.uniform(0,0.25, size=len(tau))
plt.figure()
plt.plot(ruido1, label='ruido1')
plt.plot(ruido2, label='ruido2')
plt.plot(ruido3, label='ruido3')
#plt.plot(tau1, label='tau1')
plt.legend()
plt.show()



#numero_random_para_modular = np.linspace(0, 1, len(ruido1))
#numero_random_para_modular=np.arange(0,1,0.001)
numero_random_para_modular=np.random.uniform(-1,1)
#modulacion1 = ruido1 * np.exp((numero_random_para_modular * tau**(0.5)))
#modulacion1 = np.exp(numero_random_para_modular * tau) * ruido1
#modulacion2 = np.exp(numero_random_para_modular * tau) * ruido2
#modulacion3 = np.exp(numero_random_para_modular * tau) * ruido3

modulacion1 = ruido1
modulacion2 = ruido2
modulacion3 = ruido3




s=1000
plt.figure()
plt.plot(modulacion1, label='modulacion 1')
plt.plot(modulacion2, label='modulacion 2')
plt.plot(modulacion3, label='modulacion 3')
#plt.plot(tau1, label='tau1')
plt.legend()
plt.show()

#def sigmoid(x, derivative=False):
#    sigm = 1. / (1. + np.exp(-x))
#

#==============================================================================
#                                Tipografía de los gráficos
#==============================================================================    
SMALL_SIZE = 30
MEDIUM_SIZE = 36
BIGGER_SIZE = 39

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#==============================================================================
#                                  
#                           GRAFICOS
#
#==============================================================================    
fig = plt.figure()    
plt.close('all') # amtes de graficar, cierro todos las figuras que estén abiertas
    
guardar_imagenes = False    
i=0
for t in T:
    if t==T[0]:
        ruido=ruido1
        modulacion = modulacion1
    elif t==T[1]:
        ruido=ruido2
        modulacion = modulacion2
    else:
        ruido=ruido3
        modulacion = modulacion3
    


    print(f'tiempo de binding= {t_binding} segundos')

#    if d==D[-1]:
#        guardar_imagenes = True
#    else:
#        guardar_imagenes = False
#    
    i=1   #con este indice voy a numerar las imagenes. Lo pongo dado que si quiero modificar el codigo tengo que modificar el plt.figure(3) por ejemplo.En cambio con esto es automático
#
#==============================================================================
#                                  G(tau)
#==============================================================================    

    ###termino de difusion
    G_diff = (gamma/N)*( 1 + tau/t_diff)**(-1) * ( 1 + tau*(a**2)/(t_diff))**(-1/2)
    ###termino de triplete
    G_triplet = 1+ At * np.exp(-tau/t_triplet)
    ###termino de binding
    G_bind =Ab * np.exp(-tau*t/(t_diff))
#    G_bind =Ab * np.exp(-tau/t_binding)


    ####   Esto es para usar la version de la tesis de Laura.
    G_diff_moño = G_diff + 1
    G_bind_moño = G_bind + 1
    G_triplet_moño = G_triplet + 1


    ###Normalización de factores de la ACF    
    G_diff_norm = G_diff_moño/max(G_diff_moño)
    G_triplet_norm = G_triplet_moño/max(G_triplet_moño)
    G_bind_norm = G_bind_moño/max(G_bind_moño)  


    ###-------- TOTAL ACF ------    
    Gtotal_moño = G_diff_moño * G_bind_moño 
    Gtotal = Gtotal_moño - 1
    Gtotal_normalizada = Gtotal/max(Gtotal)
    Gtotal_moño2 = G_diff * G_bind + G_diff + G_bind +1


#==============================================================================
#                                Ruido para las señales
#==============================================================================    
    Gtotal_ruido = Gtotal + Gtotal**(0.25) * modulacion3
    Gtotal_ruido_normalizada = Gtotal_ruido/max(Gtotal_ruido) 
    G_diff_ruido = (G_diff + G_diff**(0.25) * modulacion1)
    G_diff_ruido_normalizada = G_diff_ruido/max(G_diff_ruido)
    G_bind_ruido = (G_bind + G_bind**(0.25) * modulacion2)
    G_bind_ruido_normalizada = G_bind_ruido/max(G_bind_ruido)

#    G_diff_ruido = (G_diff_moño + G_diff_moño * modulacion)
#    G_diff_ruido_normalizada = G_diff_ruido/max(G_diff_ruido)
#    G_bind_ruido = (G_bind_moño  + G_bind_moño  * modulacion)
#    G_bind_ruido_normalizada = G_bind_ruido/max(G_bind_ruido)




    ###  Grafico funcion de correlación  total  CON RUIDO
    plt.figure(i)
#    plt.semilogx(tau, Gtotal_ruido_normalizada,'-',label=r'$G(\tau) - t\_bind={} - t\_dif={}$'.format(t, '%.2E' % t_binding, '%.2E' % t_diff),linewidth=2) 
#    plt.semilogx(tau, (G_bind + G_bind * modulacion)/max(G_bind),'-',label=f'T={t} - BINDING')    
    plt.semilogx(tau, Gtotal_ruido_normalizada,'-',label=f'difusión y asoc-disoc', linewidth=3)
    plt.semilogx(tau, G_diff_ruido_normalizada,'-',label='difusión', linewidth=3)
    plt.semilogx(tau, G_bind_ruido_normalizada,'-',label=r'asoc-disoc', linewidth=3)


    plt.legend(loc=3)
    plt.xlabel(r'$\tau$ - ($s$)')
    plt.ylabel(r'G($\tau$)')
#    plt.tick_params(direction='out', length=6, width=2, colors='r', grid_color='r', grid_alpha=0.5)
    plt.tick_params(which='minor', length=7.5, width=3.5)
    plt.tick_params(which='major', length=10, width=5)
    figManager = plt.get_current_fig_manager()  ####   esto y la linea de abajo me maximiza la ventana de la figura
    figManager.window.showMaximized()           ####   esto y la linea de arriba me maximiza la ventana de la figura
    plt.show()
#    plt.tight_layout() #hace que no me corte los márgenes
    
    if guardar_imagenes:
        plt.savefig('C:\\Users\\LEC\\Desktop')
    i+=1
    
    
    
    
    plt.figure(i)
#    plt.semilogx(tau, Gtotal_ruido_normalizada,'-',label=r'$G(\tau) - t\_bind={} - t\_dif={}$'.format(t, '%.2E' % t_binding, '%.2E' % t_diff),linewidth=2) 
    
    plt.semilogx(tau, Gtotal_normalizada,'-',label=f'difusión y asoc-disoc',linewidth=3)
    plt.semilogx(tau, G_diff/max(G_diff),'-',label='difusión',linewidth=3)
    plt.semilogx(tau, G_bind/max(G_bind),'-',label=r'asoc-disoc',linewidth=3)

    plt.legend(loc='lower left')
    plt.xlabel(r'$\tau$ - ($s$)')
    plt.ylabel(r'G($\tau$)')
    plt.tick_params(which='minor', length=7.5, width=3)
    plt.tick_params(which='major', length=10, width=4.5)
    figManager = plt.get_current_fig_manager()  ####   esto y la linea de abajo me maximiza la ventana de la figura
    figManager.window.showMaximized()           ####   esto y la linea de arriba me maximiza la ventana de la figura
    plt.show()
#    plt.tight_layout() #hace que no me corte los márgenes
    
    if guardar_imagenes:
        plt.savefig('C:\\Users\\LEC\\Desktop')
    i+=1
    
#    plt.semilogx(tau, G_diff/max(G_diff),'-',label='difusión',linewidth=3)
#
#    plt.legend()
#    plt.xlabel(r'$\tau$ ($s$)')
##    plt.ylabel(r'Gtot($\tau$) DIFUSION')
##    plt.title('tp = {}$\mu$s - pix size = {} $\mu$m \n box size = {} pix - D={}'.format(tp*1e6,dr,box_size,D))
##    plt.title('H-line  G total')
#    plt.show()
#    plt.tight_layout() #hace que no me corte los márgenes
#    
#    if guardar_imagenes:
#        plt.savefig('C:\\Users\\LEC\\Desktop')
#  
#
##    plt.plot((dr*x), Gtotal_normalizada,'-.',label=f'D={d}',linewidth=1)
##    plt.semilogx(tau, G_bind/max(G_bind),'-',label='G binding - tb={} '.format('%.2E' % t_binding),linewidth=2)
#    plt.semilogx(tau, G_bind/max(G_bind),'-',label=r'asoc-disoc',linewidth=3)
#
#    plt.legend()
#    plt.xlabel(r'$\tau$ - ($s$)')
##    plt.ylabel(r'Gtot($\tau$)')
##    plt.title('tp = {}$\mu$s - pix size = {} $\mu$m \n box size = {} pix - D={}'.format(tp*1e6,dr,box_size,D))
##    plt.title('H-line  G total')
#    plt.show()
#    plt.tight_layout() #hace que no me corte los márgenes
#    
#    if guardar_imagenes:
#        plt.savefig('C:\\Users\\LEC\\Desktop')
#    i+=1
    

    ###  Grafico funcion de correlación  total
#    plt.figure(i)
##    plt.plot((dr*x), Gtotal_normalizada,'-.',label=f'D={d}',linewidth=1)
#    plt.semilogx(tau, Gtotal_normalizada,'-',label=f'T={t} \n tb={t_binding}, td={t_diff}')
#    plt.semilogx(tau, G_diff_moño/max(G_bind_moño),'k')
#    plt.legend()
#    plt.xlabel(r'$\tau$ - ($s$)',fontsize=14)
#    plt.ylabel(r'Gtot($\tau$)',fontsize=14)
#    plt.title('G$(\tau) \_total$ \n tp = {}$\mu$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e6,dr,box_size),fontsize=18)
##    plt.title('H-line  G total')
#    plt.show()
#    plt.tight_layout() #hace que no me corte los márgenes
#    
#    if guardar_imagenes:
#        plt.savefig('C:\\Users\\LEC\\Desktop')
#    i+=1

#    
#    ###  Grafico funcion de correlación  total MOÑO
#    plt.figure(i)
##    plt.plot((dr*x), Gtotal_normalizada,'-.',label=f'D={d}',linewidth=1)
#    plt.semilogx(tau, Gtotal_moño,'-',label=f'T={t} \n tb={t_binding}, td={t_diff}')
#    plt.legend()
#    plt.xlabel(r'$\tau$ - ($s$)',fontsize=14)
#    plt.ylabel(r'Gtot MOÑO  ($\tau$)',fontsize=14)
#    plt.title('H-line G total \n tp = {}$\mu$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e6,dr,box_size),fontsize=18)
#    plt.title('H-line  G total')
#    plt.show()
#    plt.tight_layout() #hace que no me corte los márgenes
#    
#    if guardar_imagenes:
#        plt.savefig('C:\\Users\\LEC\\Desktop')
#    i+=1
#        
#    
#  ###  Grafico funcion de correlación  total MOÑO 2
#    plt.figure(i)
##    plt.plot((dr*x), Gtotal_normalizada,'-.',label=f'D={d}',linewidth=1)
#    plt.semilogx(tau, Gtotal_moño2,'-',label=f'T={t} \n tb={t_binding}, td={t_diff}')
#    plt.legend()
#    plt.xlabel(r'$\tau$ - ($s$)',fontsize=14)
#    plt.ylabel(r'Gtot MOÑO2 ($\tau$)',fontsize=14)
#    plt.title('H-line G total \n tp = {}$\mu$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e6,dr,box_size),fontsize=18)
#    plt.title('H-line  G total')
#    plt.show()
#    plt.tight_layout() #hace que no me corte los márgenes
#    
#    if guardar_imagenes:
#        plt.savefig('C:\\Users\\LEC\\Desktop')
#    i+=1
#            
#    
#    j=0
#    while Gtotal_normalizada[j]>0.60:
#        j+=1
#    plt.semilogx(tau[j],Gtotal_normalizada[j], 'r*', label=f'{round(tau[j],5)}$s$')      ###---> para graficar punto donde cae a la mitad la curva de correlación
###    plt.semilogx(dr*x[j],0, 'g*', label=f'dist para Gtotal/2 = {round(dr*x[j],5)}$\mu m$')      ###---> para graficar punto en x donde cae a la mitad la curva de correlación
#    plt.text(tau[j], Gtotal_normalizada[j], r'$\tau$ = {} $s$'.format(round(tau[j],3)), bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
#    plt.arrow(tau[j], Gtotal_normalizada[j], 0, -Gtotal_normalizada[j])
#
#
#    t+=1


#    j=0
#    while tau[j]<t_diff:
#        j+=1
#    plt.semilogx(tau[j],Gtotal_normalizada[j], 'm*')#, label=r'$\t_dif$ = {}$s$'.format(round(tau[j],5)))      ###---> para graficar punto donde cae a la mitad la curva de correlación
#    j=0
#    while tau[j]<t_binding:
#        j+=1
#    plt.semilogx(tau[j],Gtotal_normalizada[j], 'c*')#, label=r'$\t_bind$ = {}$s$'.format(round(tau[j],5)))#, label=f'{round(tau[j],5)}$s$')      ###---> para graficar punto donde cae a la mitad la curva de correlación
    ##    plt.semilogx(dr*x[j],0, 'g*', label=f'dist para Gtotal/2 = {round(dr*x[j],5)}$\mu m$')      ###---> para graficar punto en x donde cae a la mitad la curva de correlación
#    plt.text(tau[j], Gtotal_normalizada[j], r'$\tau$ = {} $s$'.format(round(tau[j],3)), bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
#    plt.arrow(tau[j], Gtotal_normalizada[j], 0, -Gtotal_normalizada[j])
    t+=1



plt.semilogx(tau, G_diff/max(G_diff),linewidth=3)    #Grafico la correlacion de DIFUSION 

##busco el valor de tau para el t_diff
j=0
while tau[j]<t_diff:
    j+=1
plt.semilogx(tau[j],G_diff[j]/max(G_diff), 'ko')   #marco el t_diff
plt.axvline(x=t_diff)  #linea que marca el valor del t_diff
plt.tick_params(which='minor', length=7.5, width=3)
plt.tick_params(which='major', length=10, width=4.5)
figManager = plt.get_current_fig_manager()  ####   esto y la linea de abajo me maximiza la ventana de la figura
figManager.window.showMaximized()           ####   esto y la linea de arriba me maximiza la ventana de la figura
plt.show()






#plt.figure(i)
##    plt.plot((dr*x), Gtotal_normalizada,'-.',label=f'D={d}',linewidth=1)
#plt.semilogx(tau, Gtotal_normalizada  + ruido,'-',label=f'T={t} \n tb={t_binding}, td={t_diff}')
#plt.legend()
#plt.xlabel(r'$\tau$ - ($s$)',fontsize=14)
#plt.ylabel(r'Gtot($\tau$)',fontsize=14)
#plt.title('G$(\tau) \_total$ \n tp = {}$\mu$  - pix size = {} $\mu$- box size = {} pix'.format(tp*1e6,dr,box_size),fontsize=18)
##    plt.title('H-line  G total')
#plt.show()
#plt.tight_layout() #hace que no me corte los márgenes