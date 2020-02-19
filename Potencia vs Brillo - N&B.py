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
#                                Tipografía de los gráficos
##==============================================================================    
#SMALL_SIZE = 28
#MEDIUM_SIZE = 36
#BIGGER_SIZE = 39

SMALL_SIZE = 15
MEDIUM_SIZE = 22
BIGGER_SIZE = 26
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#==============================================================================
#==============================================================================    
plt.close('all')



#===  A LA TARDE DEL MARTES 18/2

potencia = [0.1,0.5,1,1.5,2]#,10]
brillo = [0.77,0.79248,0.808,0.81,0.823]#,0.907]
#pendiente .0265
intensidad = [0.25875,0.28287,0.29732,0.30772,0.31303]
varianza = [0.20059,0.22422,0.24022,0.25169,0.25778]

i=0
cociente = []
while i<len(varianza):
    cociente.append(varianza[i]/intensidad[i])
    i+=1


###   GRAFICO INTENSIDAD

plt.figure()
plt.plot(potencia, intensidad,'^g',label='intensidad')
plt.plot(potencia, varianza,'^m',label='varianza')
plt.legend()
plt.title('TARDE DEL MARTES 18/2')
plt.xlabel('Potencia')
plt.ylabel('Intensidad')
plt.tight_layout() #hace que no me corte los márgenes
plt.tick_params(which='minor', length=2, width=1.5)
plt.tick_params(which='major', length=4, width=2)
plt.show()   


###   GRAFICO BRILLO
plt.figure()
plt.plot(potencia, brillo,'^r',linewidth=3,label='brillo')
plt.plot(potencia, brillo,'^b',linewidth=3,label='cociente')
plt.title('TARDE DEL MARTES 18/2')
plt.xlabel('Potencia')
plt.ylabel('Brillo')
plt.tight_layout() #hace que no me corte los márgenes
plt.tick_params(which='minor', length=2, width=1.5)
plt.tick_params(which='major', length=4, width=2)
plt.show() 









   
#===  A LA MAÑANA DEL MIERCOLES 19/2

potencia = [0.1,0.5,1,1.5,2]#,3,10,30]
brillo = [1.208,1.51,1.62,1.67150,1.705]#,1.736]#,1.811,1.795]
#pendiente 0.2485
intensidad = [1.10957,3.66853,6.67168,9.47475,12.25278]
#pendiente intensidad  
varianza = [1.31395,5.46812,10.75155,15.73560,20.77795]

i=0
cociente = []
while i<len(varianza):
    cociente.append(varianza[i]/intensidad[i])
    i+=1



###   GRAFICO INTENSIDAD
plt.figure()
plt.plot(potencia, intensidad,'^g',label='intensidad')
plt.plot(potencia, varianza,'^m',label='varianza')
plt.legend()
plt.title('MAÑANA DEL MIERCOLES 19/2')
plt.xlabel('Potencia')
plt.ylabel('Intensidad')
plt.tight_layout() #hace que no me corte los márgenes
plt.tick_params(which='minor', length=2, width=1.5)
plt.tick_params(which='major', length=4, width=2)
plt.show()    


###   GRAFICO BRILLO
plt.figure()
plt.plot(potencia, brillo,'^r',linewidth=3,label='brillo')
plt.plot(potencia, brillo,'^b',linewidth=3,label='cociente')
plt.legend()
plt.title('MAÑANA DEL MIERCOLES 19/2')
plt.xlabel('Potencia')
plt.ylabel('Brillo')
plt.tight_layout() #hace que no me corte los márgenes
plt.tick_params(which='minor', length=2, width=1.5)
plt.tick_params(which='major', length=4, width=2)
plt.show()    