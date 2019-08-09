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





f, (ax1, ax2, ax3) = plt.subplots(1,3)
colores_difusion=sns.color_palette("Blues")
sns.set_style("white")


ax1.set_xlabel('N en PSF = 0.05')
ax2.set_xlabel('N en PSF = 0.4')
ax3.set_xlabel('N en PSF = 3.18')

ax1.set_ylabel('S2')
ax2.set_ylabel('S2')
ax3.set_ylabel('S2')

sns.stripplot(data=S2_w0_0_25_difusion, ax=ax1, color='b')
sns.stripplot(data=S2_w0_0_5_difusion, ax=ax2, color='g')
sns.stripplot(data=S2_w0_1_difusion, ax=ax3, color='r')


sns.boxplot(data=S2_w0_0_25_difusion, ax=ax1, color=colores_difusion[0])
sns.boxplot(data=S2_w0_0_5_difusion,ax=ax2, color=colores_difusion[1])
sns.boxplot(data=S2_w0_1_difusion,ax=ax3, color=colores_difusion[2])



ax1.set_ylim([min(S2_w0_0_25_difusion)*0.1,max(S2_w0_0_25_difusion)*1.1])
ax2.set_ylim([min(S2_w0_0_5_difusion)*0.1,max(S2_w0_0_5_difusion)*1.1])
ax3.set_ylim([min(S2_w0_1_difusion)*0.1,max(S2_w0_1_difusion)*1.1])

f.suptitle('DIFUSION \n', y=1)
plt.tight_layout()
plt.show()




#plt.figure()
#w0_1_mediana = np.median(S2_w0_1_difusion)
#W0_1_promedio = np.mean(S2_w0_1_difusion)
#w0_1_moda = mode (S2_w0_1_difusion)
#
#plt.plot([3.18]*10,S2_w0_1_difusion,'r*',label='N = 3.18')
#plt.ticklabel_format(useOffset=False)

#plt.plot([0.39]*10,S2_w0_0_5_difusion,'g*',label='N = 0.39')
#plt.plot([0.05]*10,S2_w0_0_25_difusion,'b*',label='N = 0.05')
#plt.legend()
#plt.show()


#plt.figure()
#labels = ['N = 3.18', 'N = 0.39', 'N = 0.05']
#plt.boxplot(S2_w0_1_difusion, labels=labels[0])
#plt.boxplot(S2_w0_0_5_difusion, labels=labels[1])
#plt.boxplot(S2_w0_0_25_difusion, labels=labels[2])
#
##plt.boxplot(S2_w0_1_difusion, S2_w0_0_5_difusion, S2_w0_0_25_difusion, labels=labels)
#
#plt.tight_layout() #hace que no me corte los márgenes
#plt.show()


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


f, (ax1, ax2, ax3) = plt.subplots(1,3)
colores_binding=sns.cubehelix_palette(8)
sns.set_style("white")


ax1.set_xlabel('N en PSF = 0.05')
ax2.set_xlabel('N en PSF = 0.4')
ax3.set_xlabel('N en PSF = 3.18')

ax1.set_ylabel('S2')
ax2.set_ylabel('S2')
ax3.set_ylabel('S2')

sns.stripplot(data=S2_w0_0_25_binding, ax=ax1, color='b')
sns.stripplot(data=S2_w0_0_5_binding, ax=ax2, color='g')
sns.stripplot(data=S2_w0_1_binding, ax=ax3, color='r')


sns.boxplot(data=S2_w0_0_25_binding, ax=ax1, color=colores_binding[0])
sns.boxplot(data=S2_w0_0_5_binding,ax=ax2, color=colores_binding[1])
sns.boxplot(data=S2_w0_1_binding,ax=ax3, color=colores_binding[2])



ax1.set_ylim([min(S2_w0_0_25_binding)*0.1,max(S2_w0_0_25_binding)*1.1])
ax2.set_ylim([min(S2_w0_0_5_binding)*0.1,max(S2_w0_0_5_binding)*1.1])
ax3.set_ylim([min(S2_w0_1_binding)*0.1,max(S2_w0_1_binding)*1.1])

f.suptitle('BINDING \n', y=1)
plt.tight_layout()
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
plt.plot(N_difusion,S_difusion, 'b-*', label='S2 difusion')
plt.plot(N_binding,S_binding, 'r-*', label='S2 binding')

plt.xlabel('molec en PSF')
plt.ylabel('S2')
#axs.set_xlabel('N en PSF')
#axs.set_ylabel('S2')
plt.legend()
plt.show()






