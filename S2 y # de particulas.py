# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 11:10:12 2019

@author: ETCasa
"""

import matplotlib.pyplot as plt
import numpy as np
import re
from statistics import mode
import seaborn as sns
import pandas as pd

plt.close('all') 
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('lines', linewidth=3)

#==============================================================================
#
#                               DATOS     DIFUSION
#
#==============================================================================  
with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando w0 y wz y dejando fijo N - DIFUSION\\S2_w0=1_difusion_ajustado_por_difusion.txt') as fobj:
    S2_w0_1_difusion = fobj.read()
S2_w0_1_difusion = re.split('\n', S2_w0_1_difusion)
S2_w0_1_difusion = [float(i) for i in S2_w0_1_difusion]


with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando w0 y wz y dejando fijo N - DIFUSION\\S2_w0=0_5_difusion_ajustado_por_difusion.txt') as fobj:
    S2_w0_0_5_difusion = fobj.read()
S2_w0_0_5_difusion = re.split('\n', S2_w0_0_5_difusion)
S2_w0_0_5_difusion = [float(i) for i in S2_w0_0_5_difusion]


with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando w0 y wz y dejando fijo N - DIFUSION\\S2_w0=0_25_difusion_ajustado_por_difusion.txt') as fobj:
    S2_w0_0_25_difusion = fobj.read()
S2_w0_0_25_difusion = re.split('\n', S2_w0_0_25_difusion)
#S2_binding_puro_ajustado_por_dif.remove('')
S2_w0_0_25_difusion = [float(i) for i in S2_w0_0_25_difusion]



with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando w0 y wz y dejando fijo N - DIFUSION\\S2_w0=1_difusion_ajustado_por_binding.txt') as fobj:
    S2_w0_1_difusion_ajustado_binding = fobj.read()
S2_w0_1_difusion_ajustado_binding = re.split('\n', S2_w0_1_difusion_ajustado_binding)
S2_w0_1_difusion_ajustado_binding = [float(i) for i in S2_w0_1_difusion_ajustado_binding]


with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando w0 y wz y dejando fijo N - DIFUSION\\S2_w0=0_5_difusion_ajustado_por_binding.txt') as fobj:
    S2_w0_0_5_difusion_ajustado_binding = fobj.read()
S2_w0_0_5_difusion_ajustado_binding = re.split('\n', S2_w0_0_5_difusion_ajustado_binding)
S2_w0_0_5_difusion_ajustado_binding = [float(i) for i in S2_w0_0_5_difusion_ajustado_binding]


with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando w0 y wz y dejando fijo N - DIFUSION\\S2_w0=0_25_difusion_ajustado_por_binding.txt') as fobj:
    S2_w0_0_25_difusion_ajustado_binding = fobj.read()
S2_w0_0_25_difusion_ajustado_binding = re.split('\n', S2_w0_0_25_difusion_ajustado_binding)
#S2_binding_puro_ajustado_por_dif.remove('')
S2_w0_0_25_difusion_ajustado_binding = [float(i) for i in S2_w0_0_25_difusion_ajustado_binding]
#==============================================================================
#
#                               DATOS     BINDING
#
#==============================================================================  
with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando w0 y wz y dejando fijo N - BINDING\\S2_w0=1_binding_ajustado_por_binding.txt') as fobj:
    S2_w0_1_binding = fobj.read()
S2_w0_1_binding = re.split('\n', S2_w0_1_binding )
S2_w0_1_binding = [float(i) for i in S2_w0_1_binding ]


with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando w0 y wz y dejando fijo N - BINDING\\S2_w0=0_5_binding_ajustado_por_binding.txt') as fobj:
    S2_w0_0_5_binding = fobj.read()
S2_w0_0_5_binding = re.split('\n', S2_w0_0_5_binding )
S2_w0_0_5_binding = [float(i) for i in S2_w0_0_5_binding ]


with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando w0 y wz y dejando fijo N - BINDING\\S2_w0=0_25_binding_ajustado_por_binding.txt') as fobj:
    S2_w0_0_25_binding = fobj.read()
S2_w0_0_25_binding = re.split('\n', S2_w0_0_25_binding )
#S2_binding_puro_ajustado_por_dif.remove('')
S2_w0_0_25_binding  = [float(i) for i in S2_w0_0_25_binding ]



with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando w0 y wz y dejando fijo N - BINDING\\S2_w0=1_binding_ajustado_por_difusion.txt') as fobj:
    S2_w0_1_binding_ajustado_por_difusion = fobj.read()
S2_w0_1_binding_ajustado_por_difusion = re.split('\n', S2_w0_1_binding_ajustado_por_difusion )
S2_w0_1_binding_ajustado_por_difusion = [float(i) for i in S2_w0_1_binding_ajustado_por_difusion ]


with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando w0 y wz y dejando fijo N - BINDING\\S2_w0=0_5_binding_ajustado_por_difusion.txt') as fobj:
    S2_w0_0_5_binding_ajustado_por_difusion = fobj.read()
S2_w0_0_5_binding_ajustado_por_difusion = re.split('\n', S2_w0_0_5_binding_ajustado_por_difusion )
S2_w0_0_5_binding_ajustado_por_difusion = [float(i) for i in S2_w0_0_5_binding_ajustado_por_difusion ]


with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando w0 y wz y dejando fijo N - BINDING\\S2_w0=0_25_binding_ajustado_por_difusion.txt') as fobj:
    S2_w0_0_25_binding_ajustado_por_difusion = fobj.read()
S2_w0_0_25_binding_ajustado_por_difusion = re.split('\n', S2_w0_0_25_binding_ajustado_por_difusion )
S2_w0_0_25_binding_ajustado_por_difusion  = [float(i) for i in S2_w0_0_25_binding_ajustado_por_difusion ]









#==============================================================================   
#                                w0= 1 - wz= 6 - N en PSF = 3.18
#==============================================================================
#==============================================================================
#
#                               Boxplot DIFUSION
#
#============================================================================== 
  
#f, axes = f, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'wspace':0.35})
#colores_difusion=sns.color_palette("Blues")
#sns.set_style("white")
#
#sns.stripplot(x=S2_w0_1_difusion, ax=ax1, color='mediumaquamarine', size=10)
#sns.boxplot(x=S2_w0_1_difusion, ax=ax1, color='forestgreen')
#
#sns.stripplot(x=S2_w0_1_difusion_ajustado_binding, ax=ax2, color='steelblue', size=10)
#sns.boxplot(x=S2_w0_1_difusion_ajustado_binding, ax=ax2, color='b')
#
#
#
#ax1.set_xlim([min(S2_w0_1_difusion)*0.85,max(S2_w0_1_difusion)*1.15])
#ax2.set_xlim([min(S2_w0_1_difusion_ajustado_binding)*0.85,max(S2_w0_1_difusion_ajustado_binding)*1.15])
#ax1.set_xlabel('S2')
#ax2.set_xlabel('S2')
#
#ax1.set_ylabel('Ajuste: Binding')
#ax2.set_ylabel('Ajuste: Difusion')
#
#
##ax1.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=f.transFigure)    #bbox_transform=f.transFigure le está diciendo que la legenda la ubique usando la figura grande, y NO el subplot
#                                                                          #bbox_to_anchor=(1, 1) ubica en la posicion arriba a la derecha a la leyenda
##ax.get_xticklabels()  
#                                                                       #loc=1 se refiere a la esquina derecha superior DE LA LEYENDA! NO DE LA FIGURA. ie: para ubicar la leyenda en el bbox_to_anchor utiliza el punto de arriba a la derecha de la leyenda
#for ax in axes:
##    ax.set_xticks([]) 
#    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
#    
#f.suptitle('Proceso Difusion - N en PSF=3.18 \n')
#plt.show()
##==============================================================================
##
##                               Boxplot ASOC- DISOC
##
##============================================================================== 
#
#f, axes = f, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'wspace':0.35})
#colores_difusion=sns.color_palette("Blues")
#sns.set_style("white")
#
#sns.stripplot(x=S2_w0_1_binding, ax=ax1, color='lightsalmon', size=10)
#sns.boxplot(x=S2_w0_1_binding, ax=ax1, color='tomato',)
#
#
#sns.stripplot(x=S2_w0_1_binding_ajustado_por_difusion, ax=ax2, color='indianred',size=10)
#sns.boxplot(x=S2_w0_1_binding_ajustado_por_difusion, ax=ax2, color='#8f1402')
#
#
#
#ax1.set_xlim([min(S2_w0_1_binding)*0.85,max(S2_w0_1_binding)*1.15])
#ax2.set_xlim([min(S2_w0_1_binding_ajustado_por_difusion)*0.85,max(S2_w0_1_binding_ajustado_por_difusion)*1.15])
#ax1.set_xlabel('S2')
#ax2.set_xlabel('S2')
#
#ax1.set_ylabel('Ajuste: Binding')
#ax2.set_ylabel('Ajuste: Difusion')
#
##ax1.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=f.transFigure)    #bbox_transform=f.transFigure le está diciendo que la legenda la ubique usando la figura grande, y NO el subplot
#                                                                          #bbox_to_anchor=(1, 1) ubica en la posicion arriba a la derecha a la leyenda
#ax.get_xticklabels()                                                                         #loc=1 se refiere a la esquina derecha superior DE LA LEYENDA! NO DE LA FIGURA. ie: para ubicar la leyenda en el bbox_to_anchor utiliza el punto de arriba a la derecha de la leyenda
#for ax in axes:
##    ax.set_xticks([])
#    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
#    
#    
#f.suptitle('Proceso Difusion - N en PSF=3.18 \n')
#plt.show()
#
#
#
##==============================================================================   
##                                w0= 0.5 - wz= 3 - N en PSF = 0.39
##==============================================================================
##==============================================================================
##
##                               Boxplot DIFUSION
##
##============================================================================== 
#  
#f, axes = f, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'wspace':0.35})
#colores_difusion=sns.color_palette("Blues")
#sns.set_style("white")
#
#sns.stripplot(x=S2_w0_0_5_difusion, ax=ax1, color='mediumaquamarine', size=10)
#sns.boxplot(x=S2_w0_0_5_difusion, ax=ax1, color='forestgreen')
#
#sns.stripplot(x=S2_w0_0_5_difusion_ajustado_binding, ax=ax2, color='steelblue', size=10)
#sns.boxplot(x=S2_w0_0_5_difusion_ajustado_binding, ax=ax2, color='b')
#
#
#
#ax1.set_xlim([min(S2_w0_0_5_difusion)*0.85,max(S2_w0_0_5_difusion)*1.15])
#ax2.set_xlim([min(S2_w0_0_5_difusion_ajustado_binding)*0.85,max(S2_w0_0_5_difusion_ajustado_binding)*1.15])
#ax1.set_xlabel('S2')
#ax2.set_xlabel('S2')
#
#ax1.set_ylabel('Ajuste: Binding')
#ax2.set_ylabel('Ajuste: Difusion')
#
#
##ax1.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=f.transFigure)    #bbox_transform=f.transFigure le está diciendo que la legenda la ubique usando la figura grande, y NO el subplot
#                                                                          #bbox_to_anchor=(1, 1) ubica en la posicion arriba a la derecha a la leyenda
##ax.get_xticklabels()  
#                                                                       #loc=1 se refiere a la esquina derecha superior DE LA LEYENDA! NO DE LA FIGURA. ie: para ubicar la leyenda en el bbox_to_anchor utiliza el punto de arriba a la derecha de la leyenda
#for ax in axes:
##    ax.set_xticks([]) 
#    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
#    
#f.suptitle('Proceso Difusion - N en PSF=0.39 \n')
#plt.show()
##==============================================================================
##
##                               Boxplot ASOC- DISOC
##
##============================================================================== 
#
#f, axes = f, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'wspace':0.35})
#colores_difusion=sns.color_palette("Blues")
#sns.set_style("white")
#
#sns.stripplot(x=S2_w0_0_5_binding, ax=ax1, color='lightsalmon', size=10)
#sns.boxplot(x=S2_w0_0_5_binding, ax=ax1, color='tomato',)
#
#
#sns.stripplot(x=S2_w0_0_5_binding_ajustado_por_difusion, ax=ax2, color='indianred',size=10)
#sns.boxplot(x=S2_w0_0_5_binding_ajustado_por_difusion, ax=ax2, color='#8f1402')
#
#
#
#ax1.set_xlim([min(S2_w0_0_5_binding)*0.85,max(S2_w0_0_5_binding)*1.15])
#ax2.set_xlim([min(S2_w0_0_5_binding_ajustado_por_difusion)*0.85,max(S2_w0_0_5_binding_ajustado_por_difusion)*1.15])
#ax1.set_xlabel('S2')
#ax2.set_xlabel('S2')
#
#ax1.set_ylabel('Ajuste: Binding')
#ax2.set_ylabel('Ajuste: Difusion')
#
##ax1.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=f.transFigure)    #bbox_transform=f.transFigure le está diciendo que la legenda la ubique usando la figura grande, y NO el subplot
#                                                                          #bbox_to_anchor=(1, 1) ubica en la posicion arriba a la derecha a la leyenda
#ax.get_xticklabels()                                                                         #loc=1 se refiere a la esquina derecha superior DE LA LEYENDA! NO DE LA FIGURA. ie: para ubicar la leyenda en el bbox_to_anchor utiliza el punto de arriba a la derecha de la leyenda
#for ax in axes:
##    ax.set_xticks([])
#    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
#    
#    
#f.suptitle('Proceso Difusion - N en PSF=0.39 \n')
#plt.show()
#
#
#
##==============================================================================   
##                                w0= 0.25 - wz= 1.5 - N en PSF = 0.05
##==============================================================================
##==============================================================================
##
##                               Boxplot DIFUSION
##
##============================================================================== 
#f, axes = f, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'wspace':0.35})
#colores_difusion=sns.color_palette("Blues")
#sns.set_style("white")
#
#sns.stripplot(x=S2_w0_0_25_difusion, ax=ax1, color='mediumaquamarine', size=10)
#sns.boxplot(x=S2_w0_0_25_difusion, ax=ax1, color='forestgreen')
#
#sns.stripplot(x=S2_w0_0_25_difusion_ajustado_binding, ax=ax2, color='steelblue', size=10)
#sns.boxplot(x=S2_w0_0_25_difusion_ajustado_binding, ax=ax2, color='b')
#
#
#
#ax1.set_xlim([min(S2_w0_0_25_difusion)*0.85,max(S2_w0_0_25_difusion)*1.15])
#ax2.set_xlim([min(S2_w0_0_25_difusion_ajustado_binding)*0.85,max(S2_w0_0_25_difusion_ajustado_binding)*1.15])
#ax1.set_xlabel('S2')
#ax2.set_xlabel('S2')
#
#ax1.set_ylabel('Ajuste: Binding')
#ax2.set_ylabel('Ajuste: Difusion')
#
#
##ax1.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=f.transFigure)    #bbox_transform=f.transFigure le está diciendo que la legenda la ubique usando la figura grande, y NO el subplot
#                                                                          #bbox_to_anchor=(1, 1) ubica en la posicion arriba a la derecha a la leyenda
##ax.get_xticklabels()  
#                                                                       #loc=1 se refiere a la esquina derecha superior DE LA LEYENDA! NO DE LA FIGURA. ie: para ubicar la leyenda en el bbox_to_anchor utiliza el punto de arriba a la derecha de la leyenda
#for ax in axes:
##    ax.set_xticks([]) 
#    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
#    
#f.suptitle('Proceso Difusion - N en PSF=0.05 \n')
#plt.show()
##==============================================================================
##
##                               Boxplot ASOC- DISOC
##
##============================================================================== 
#
#f, axes = f, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'wspace':0.35})
#colores_difusion=sns.color_palette("Blues")
#sns.set_style("white")
#
#sns.stripplot(x=S2_w0_0_25_binding, ax=ax1, color='lightsalmon', size=10)
#sns.boxplot(x=S2_w0_0_25_binding, ax=ax1, color='tomato',)
#
#
#sns.stripplot(x=S2_w0_0_25_binding_ajustado_por_difusion, ax=ax2, color='indianred',size=10)
#sns.boxplot(x=S2_w0_0_25_binding_ajustado_por_difusion, ax=ax2, color='#8f1402')
#
#
#
#ax1.set_xlim([min(S2_w0_0_25_binding)*0.85,max(S2_w0_0_25_binding)*1.15])
#ax2.set_xlim([min(S2_w0_0_25_binding_ajustado_por_difusion)*0.85,max(S2_w0_0_25_binding_ajustado_por_difusion)*1.15])
#ax1.set_xlabel('S2')
#ax2.set_xlabel('S2')
#
#ax1.set_ylabel('Ajuste: Binding')
#ax2.set_ylabel('Ajuste: Difusion')
#
##ax1.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=f.transFigure)    #bbox_transform=f.transFigure le está diciendo que la legenda la ubique usando la figura grande, y NO el subplot
#                                                                          #bbox_to_anchor=(1, 1) ubica en la posicion arriba a la derecha a la leyenda
#ax.get_xticklabels()                                                                         #loc=1 se refiere a la esquina derecha superior DE LA LEYENDA! NO DE LA FIGURA. ie: para ubicar la leyenda en el bbox_to_anchor utiliza el punto de arriba a la derecha de la leyenda
#for ax in axes:
##    ax.set_xticks([])
#    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
#    
#    
#f.suptitle('Proceso Difusion - N en PSF=0.39 \n')
#plt.show()
#
#
#























#==============================================================================
#
#                               Boxplot en 2x2
#
#============================================================================== 
fig = plt.figure(figsize=(11,10))
colores_difusion=sns.color_palette("Blues")
fig.suptitle('N en PSF=3.18', bbox={'facecolor': 'chocolate', 'alpha': 0.5, 'pad': 1})
#==============================================================================
#
#                               Boxplot ASOC- DISOC
#
#============================================================================== 
ax1 =fig.add_subplot(2,2,1)
colores_difusion=sns.color_palette("Blues")
sns.set_style("white")

sns.stripplot(x=S2_w0_1_binding, ax=ax1, color='lightsalmon', size=10)
sns.boxplot(x=S2_w0_1_binding, ax=ax1, color='tomato',)
ax1.set_xlim([min(S2_w0_1_binding)*0.95,max(S2_w0_1_binding)*1.05])
ax1.set_xlabel('S2')
ax1.set_title('Ajuste: \n Asociacion-Disociacion')
ax1.set_ylabel('Proceso \n Asociacion-Disociacion')
ax1.get_xticklabels()                                                                         #loc=1 se refiere a la esquina derecha superior DE LA LEYENDA! NO DE LA FIGURA. ie: para ubicar la leyenda en el bbox_to_anchor utiliza el punto de arriba a la derecha de la leyenda
ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))


ax1 =fig.add_subplot(2,2,2)
sns.stripplot(x=S2_w0_1_binding_ajustado_por_difusion, ax=ax1, color='indianred',size=10)
sns.boxplot(x=S2_w0_1_binding_ajustado_por_difusion, ax=ax1, color='#8f1402')
ax1.set_xlim([min(S2_w0_1_binding_ajustado_por_difusion)*0.95,max(S2_w0_1_binding_ajustado_por_difusion)*1.05])
ax1.set_xlabel('S2')
ax1.set_ylabel('Proceso \n Asociacion-Disociacion')
ax1.set_title('Ajuste: \n Difusion')
ax1.get_xticklabels()                                                                         #loc=1 se refiere a la esquina derecha superior DE LA LEYENDA! NO DE LA FIGURA. ie: para ubicar la leyenda en el bbox_to_anchor utiliza el punto de arriba a la derecha de la leyenda
ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))


ax1 =fig.add_subplot(2,2,3)
sns.stripplot(x=S2_w0_1_difusion_ajustado_binding, ax=ax1, color='steelblue', size=10)
sns.boxplot(x=S2_w0_1_difusion_ajustado_binding, ax=ax1, color='b')
ax1.set_xlim([min(S2_w0_1_difusion_ajustado_binding)*0.95,max(S2_w0_1_difusion_ajustado_binding)*1.05])
ax1.set_xlabel('S2')
ax1.set_ylabel('Proceso \n Difusion')
ax1.set_title('Ajuste: \n Asociacion-Disociacion')
ax1.get_xticklabels()  
ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))    
    

ax1 =fig.add_subplot(2,2,4)
sns.set_style("white")
sns.stripplot(x=S2_w0_1_difusion, ax=ax1, color='mediumaquamarine', size=10)
sns.boxplot(x=S2_w0_1_difusion, ax=ax1, color='forestgreen')
ax1.set_xlim([min(S2_w0_1_difusion)*0.95,max(S2_w0_1_difusion)*1.05])
ax1.set_xlabel('S2')
ax1.set_ylabel('Proceso \n Difusion')
ax1.set_title('Ajuste: \n Difusion')
ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    
plt.gcf().subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.show()


#==============================================================================
#
#                               Boxplot en 2x2    Con   w0= 0.5 - wz= 3 - N en PSF = 0.39
#
#============================================================================== 
fig = plt.figure(figsize=(11,10))
colores_difusion=sns.color_palette("Blues")
fig.suptitle('N en PSF=0.39', bbox={'facecolor': 'peru', 'alpha': 0.5, 'pad': 1})
#==============================================================================
#
#                               Boxplot ASOC- DISOC
#
#============================================================================== 
ax1 =fig.add_subplot(2,2,1)
colores_difusion=sns.color_palette("Blues")
sns.set_style("white")

sns.stripplot(x=S2_w0_0_5_binding, ax=ax1, color='lightsalmon', size=10)
sns.boxplot(x=S2_w0_0_5_binding, ax=ax1, color='tomato',)
ax1.set_xlim([min(S2_w0_0_5_binding)*0.95,max(S2_w0_0_5_binding)*1.05])
ax1.set_xlabel('S2')
ax1.set_title('Ajuste: \n Asociacion-Disociacion')
ax1.set_ylabel('Proceso \n Asociacion-Disociacion')
ax1.get_xticklabels()                                                                         #loc=1 se refiere a la esquina derecha superior DE LA LEYENDA! NO DE LA FIGURA. ie: para ubicar la leyenda en el bbox_to_anchor utiliza el punto de arriba a la derecha de la leyenda
ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
   

ax1 =fig.add_subplot(2,2,2)
sns.stripplot(x=S2_w0_0_5_binding_ajustado_por_difusion, ax=ax1, color='indianred',size=10)
sns.boxplot(x=S2_w0_0_5_binding_ajustado_por_difusion, ax=ax1, color='#8f1402')
ax1.set_xlim([min(S2_w0_0_5_binding_ajustado_por_difusion)*0.95,max(S2_w0_0_5_binding_ajustado_por_difusion)*1.05])
ax1.set_xlabel('S2')
ax1.set_ylabel('Proceso \n Asociacion-Disociacion')
ax1.set_title('Ajuste: \n Difusion')
ax1.get_xticklabels()                                                                         #loc=1 se refiere a la esquina derecha superior DE LA LEYENDA! NO DE LA FIGURA. ie: para ubicar la leyenda en el bbox_to_anchor utiliza el punto de arriba a la derecha de la leyenda
ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))


ax1 =fig.add_subplot(2,2,3)
sns.stripplot(x=S2_w0_0_5_difusion_ajustado_binding, ax=ax1, color='steelblue', size=10)
sns.boxplot(x=S2_w0_0_5_difusion_ajustado_binding, ax=ax1, color='b')
ax1.set_xlim([min(S2_w0_0_5_difusion_ajustado_binding)*0.95,max(S2_w0_0_5_difusion_ajustado_binding)*1.05])
ax1.set_xlabel('S2')
ax1.set_ylabel('Proceso \n Difusion')
ax1.set_title('Ajuste: \n Asociacion-Disociacion')
ax1.get_xticklabels()  
ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))    
    

ax1 =fig.add_subplot(2,2,4)
sns.set_style("white")
sns.stripplot(x=S2_w0_0_5_difusion, ax=ax1, color='mediumaquamarine', size=10)
sns.boxplot(x=S2_w0_0_5_difusion, ax=ax1, color='forestgreen')
ax1.set_xlim([min(S2_w0_0_5_difusion)*0.95,max(S2_w0_0_5_difusion)*1.05])
ax1.set_xlabel('S2')
ax1.set_ylabel('Proceso \n Difusion')
ax1.set_title('Ajuste: \n Difusion')
ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    
plt.gcf().subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.show()




#==============================================================================
#
#                               Boxplot en 2x2    Con   w0= 0.25 - wz= 1.5 - N en PSF = 0.5
#
#============================================================================== 
fig = plt.figure(figsize=(11,10))
colores_difusion=sns.color_palette("Blues")
fig.suptitle('N en PSF=0.05', bbox={'facecolor': 'burlywood', 'alpha': 0.5, 'pad': 1})
#==============================================================================
#
#                               Boxplot ASOC- DISOC
#
#============================================================================== 
ax1 =fig.add_subplot(2,2,1)
colores_difusion=sns.color_palette("Blues")
sns.set_style("white")

sns.stripplot(x=S2_w0_0_25_binding, ax=ax1, color='lightsalmon', size=10)
sns.boxplot(x=S2_w0_0_25_binding, ax=ax1, color='tomato',)
ax1.set_xlim([min(S2_w0_0_25_binding)*0.95,max(S2_w0_0_25_binding)*1.05])
ax1.set_xlabel('S2')
ax1.set_title('Ajuste: \n Asociacion-Disociacion')
ax1.set_ylabel('Proceso \n Asociacion-Disociacion')
ax1.get_xticklabels()                                                                         #loc=1 se refiere a la esquina derecha superior DE LA LEYENDA! NO DE LA FIGURA. ie: para ubicar la leyenda en el bbox_to_anchor utiliza el punto de arriba a la derecha de la leyenda
ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
   

ax1 =fig.add_subplot(2,2,2)
sns.stripplot(x=S2_w0_0_25_binding_ajustado_por_difusion, ax=ax1, color='indianred',size=10)
sns.boxplot(x=S2_w0_0_25_binding_ajustado_por_difusion, ax=ax1, color='#8f1402')
ax1.set_xlim([min(S2_w0_0_25_binding_ajustado_por_difusion)*0.95,max(S2_w0_0_25_binding_ajustado_por_difusion)*1.05])
ax1.set_xlabel('S2')
ax1.set_ylabel('Proceso \n Asociacion-Disociacion')
ax1.set_title('Ajuste: \n Difusion')
ax1.get_xticklabels()                                                                         #loc=1 se refiere a la esquina derecha superior DE LA LEYENDA! NO DE LA FIGURA. ie: para ubicar la leyenda en el bbox_to_anchor utiliza el punto de arriba a la derecha de la leyenda
ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))


ax1 =fig.add_subplot(2,2,3)
sns.stripplot(x=S2_w0_0_25_difusion_ajustado_binding, ax=ax1, color='steelblue', size=10)
sns.boxplot(x=S2_w0_0_25_difusion_ajustado_binding, ax=ax1, color='b')
ax1.set_xlim([min(S2_w0_0_25_difusion_ajustado_binding)*0.95,max(S2_w0_0_25_difusion_ajustado_binding)*1.05])
ax1.set_xlabel('S2')
ax1.set_ylabel('Proceso \n Difusion')
ax1.set_title('Ajuste: \n Asociacion-Disociacion')
ax1.get_xticklabels()  
ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))    
    

ax1 =fig.add_subplot(2,2,4)
sns.set_style("white")
sns.stripplot(x=S2_w0_0_25_difusion, ax=ax1, color='mediumaquamarine', size=10)
sns.boxplot(x=S2_w0_0_25_difusion, ax=ax1, color='forestgreen')
ax1.set_xlim([min(S2_w0_0_25_difusion)*0.85,max(S2_w0_0_25_difusion)*1.15])
ax1.set_xlabel('S2')
ax1.set_ylabel('Proceso \n Difusion')
ax1.set_title('Ajuste: \n Difusion')
ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    
plt.gcf().subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.show()




#
#
#
#
##==============================================================================
##
##                                 RELACION S2 Y VARIACON DE N
##
##==============================================================================  
with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando N - w0 y wz fijos - DIFUSION\\N - S2 y N - DIFUSION.txt') as fobj:
    N_difusion = fobj.read()
N_difusion = re.split('\n', N_difusion)
N_difusion = [float(i) for i in N_difusion]

with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando N - w0 y wz fijos - DIFUSION\\S2 - S2 y N - DIFUSION.txt') as fobj:
    S_difusion = fobj.read()
S_difusion = re.split('\n', S_difusion)
S_difusion = [float(i) for i in S_difusion]



with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando N - w0 y wz fijos - BINDING\\N - S2 y N - BINDING.txt') as fobj:
    N_binding = fobj.read()
N_binding = re.split('\n', N_binding)
N_binding = [float(i) for i in N_binding]



with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando N - w0 y wz fijos - BINDING\\S2 - S2 y N - BINDING.txt') as fobj:
    S_binding = fobj.read()
S_binding = re.split('\n', S_binding)
S_binding = [float(i) for i in S_binding]




plt.figure()
#fig.suptitle('S2 vs # de molec en PSF')
#fig, axs = plt.subplots(2)
plt.semilogx(N_difusion,S_difusion, 'b-*', label='S2 difusion')
plt.semilogx(N_binding,S_binding, 'r-*', label='S2 binding')

plt.xlabel('log(molec en PSF)')
plt.ylabel('S2')
#axs.set_xlabel('N en PSF')
#axs.set_ylabel('S2')
plt.legend()
plt.show()





