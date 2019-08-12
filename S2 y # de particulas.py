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
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#==============================================================================
#
#                               DIFUSION
#
#==============================================================================  
with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando w0 y wz y dejando fijo N - DIFUSION\\S2_w0=1.txt') as fobj:
    S2_w0_1_difusion = fobj.read()
S2_w0_1_difusion = re.split('\n', S2_w0_1_difusion)
S2_w0_1_difusion = [float(i) for i in S2_w0_1_difusion]


with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando w0 y wz y dejando fijo N - DIFUSION\\S2_w0=0_5.txt') as fobj:
    S2_w0_0_5_difusion = fobj.read()
S2_w0_0_5_difusion = re.split('\n', S2_w0_0_5_difusion)
S2_w0_0_5_difusion = [float(i) for i in S2_w0_0_5_difusion]


with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando w0 y wz y dejando fijo N - DIFUSION\\S2_w0=0_25.txt') as fobj:
    S2_w0_0_25_difusion = fobj.read()
S2_w0_0_25_difusion = re.split('\n', S2_w0_0_25_difusion)
#S2_binding_puro_ajustado_por_dif.remove('')
S2_w0_0_25_difusion = [float(i) for i in S2_w0_0_25_difusion]


#==============================================================================
#                                USANDO    SEABORN
#==============================================================================   

#                                BOXPLOTS JUNTOS
#==============================================================================   
plt.figure()
colores_difusion=sns.color_palette("Blues")
sns.set_style("white")
X_Label = ['N en PSF = 0.05', 'N en PSF = 0.4', 'N en PSF = 3.18']
sns.stripplot(data=[S2_w0_0_25_difusion,S2_w0_0_5_difusion,S2_w0_1_difusion])
ax=sns.boxplot(data=[S2_w0_0_25_difusion,S2_w0_0_5_difusion,S2_w0_1_difusion])
ax.artists[0].set_facecolor(colores_difusion[0])
ax.artists[1].set_facecolor(colores_difusion[1])
ax.artists[2].set_facecolor(colores_difusion[2])

ax.get_xticklabels()
ax.set_xticklabels(X_Label)
#ax.set_ylim(-min(S2_w0_1_difusion),max(S2_w0_0_25_difusion)*1.01)


plt.title('DIFUSION \n', y=1)
plt.show()



#                                BOXPLOTS SEPARADOS
#==============================================================================   

f, axes = f, (ax1, ax2, ax3) = plt.subplots(1,3, gridspec_kw={'wspace':0.35})
colores_difusion=sns.color_palette("Blues")
sns.set_style("white")

sns.stripplot(data=S2_w0_0_25_difusion, ax=ax1, color='b')
sns.stripplot(data=S2_w0_0_5_difusion, ax=ax2, color='g')
sns.stripplot(data=S2_w0_1_difusion, ax=ax3, color='r')


sns.boxplot(data=S2_w0_0_25_difusion, ax=ax1, color=colores_difusion[0])
sns.boxplot(data=S2_w0_0_5_difusion,ax=ax2, color=colores_difusion[1])
sns.boxplot(data=S2_w0_1_difusion,ax=ax3, color=colores_difusion[2])

ax2.plot(0.0000502642,'m^', label='valor de S2 variando N', zorder=1)
ax1.plot(0.0022476617,'m^', label='valor de S2 variando N', zorder=1)


ax1.set_ylim([min(S2_w0_0_25_difusion)*0.1,max(S2_w0_0_25_difusion)*1.1])
ax2.set_ylim([min(S2_w0_0_5_difusion)*0.1,max(S2_w0_0_5_difusion)*1.1])
ax3.set_ylim([min(S2_w0_1_difusion)*0.1,max(S2_w0_1_difusion)*1.1])

ax1.set_ylabel('S2')

ax1.set_xlabel('0.05')
ax2.set_xlabel('0.4\nN en PSF')
ax3.set_xlabel('3.18')

ax1.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=f.transFigure)    #bbox_transform=f.transFigure le está diciendo que la legenda la ubique usando la figura grande, y NO el subplot
                                                                          #bbox_to_anchor=(1, 1) ubica en la posicion arriba a la derecha a la leyenda
                                                                          #loc=1 se refiere a la esquina derecha superior DE LA LEYENDA! NO DE LA FIGURA. ie: para ubicar la leyenda en el bbox_to_anchor utiliza el punto de arriba a la derecha de la leyenda
for ax in axes:
    ax.set_xticks([])
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

f.suptitle('DIFUSION \n')
plt.show()





#==============================================================================
#
#                               BINDING
#
#==============================================================================  


with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando w0 y wz y dejando fijo N - BINDING\\S2_w0=1_binding.txt') as fobj:
    S2_w0_1_binding = fobj.read()
S2_w0_1_binding = re.split('\n', S2_w0_1_binding )
S2_w0_1_binding = [float(i) for i in S2_w0_1_binding ]


with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando w0 y wz y dejando fijo N - BINDING\\S2_w0=0_5_binding.txt') as fobj:
    S2_w0_0_5_binding = fobj.read()
S2_w0_0_5_binding = re.split('\n', S2_w0_0_5_binding )
S2_w0_0_5_binding = [float(i) for i in S2_w0_0_5_binding ]


with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\Variando w0 y wz y dejando fijo N - BINDING\\S2_w0=0_25_binding.txt') as fobj:
    S2_w0_0_25_binding = fobj.read()
S2_w0_0_25_binding = re.split('\n', S2_w0_0_25_binding )
#S2_binding_puro_ajustado_por_dif.remove('')
S2_w0_0_25_binding  = [float(i) for i in S2_w0_0_25_binding ]



#                                BOXPLOTS JUNTOS
#==============================================================================   
plt.figure()
colores_binding=sns.cubehelix_palette(8)
sns.set_style("white")
sns.stripplot(data=[S2_w0_0_25_binding,S2_w0_0_5_binding,S2_w0_1_binding])
ax=sns.boxplot(data=[S2_w0_0_25_binding,S2_w0_0_5_binding,S2_w0_1_binding])
ax.artists[0].set_facecolor(colores_binding[0])
ax.artists[1].set_facecolor(colores_binding[1])
ax.artists[2].set_facecolor(colores_binding[2])

ax.get_xticklabels()
ax.set_xticklabels(X_Label)
plt.title('BINDING \n', y=1)
plt.show()

#                                BOXPLOTS SEPARADOS
#==============================================================================   
f, axes = f, (ax1, ax2, ax3) = plt.subplots(1,3, gridspec_kw={'wspace':0.35})
colores_binding=sns.cubehelix_palette(8)
sns.set_style("white")

sns.stripplot(data=S2_w0_0_25_binding, ax=ax1, color='b')
sns.stripplot(data=S2_w0_0_5_binding, ax=ax2, color='g')
sns.stripplot(data=S2_w0_1_binding, ax=ax3, color='r')

ax2.plot(0.0000775362,'m^', label='valor de S2 variando N',markersize=15,zorder=1)
ax1.plot(0.0052024122,'m^', label='valor de S2 variando N', zorder=1)


sns.boxplot(data=S2_w0_0_25_binding, ax=ax1, color=colores_binding[0])
sns.boxplot(data=S2_w0_0_5_binding,ax=ax2, color=colores_binding[1])
sns.boxplot(data=S2_w0_1_binding,ax=ax3, color=colores_binding[2])



ax1.set_ylim([min(S2_w0_0_25_binding)*0.1,max(S2_w0_0_25_binding)*1.1])
ax2.set_ylim([min(S2_w0_0_5_binding)*0.1,max(S2_w0_0_5_binding)*1.1])
ax3.set_ylim([min(S2_w0_1_binding)*0.1,max(S2_w0_1_binding)*1.1])

ax1.set_ylabel('S2')

ax1.set_xlabel('0.05')
ax2.set_xlabel('0.4\nN en PSF')
ax3.set_xlabel('3.18')

ax1.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=f.transFigure) 

for ax in axes:
    ax.set_xticks([])
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

f.suptitle('BINDING \n', y=1)
plt.show()



#==============================================================================
#
#                                 RELACION S2 Y VARIACON DE N
#
#==============================================================================  
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

plt.xlabel('molec en PSF')
plt.ylabel('S2')
#axs.set_xlabel('N en PSF')
#axs.set_ylabel('S2')
plt.legend()
plt.show()






