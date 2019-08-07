# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 11:10:12 2019

@author: ETCasa
"""

import matplotlib.pyplot as plt
import numpy as np
import re
from statistics import mode
#import seaborn as sns
#import pandas as pd

plt.close('all') 

with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\S2_w0=1.txt') as fobj:
    S2_w0_1 = fobj.read()
S2_w0_1 = re.split('\n', S2_w0_1)
S2_w0_1 = [float(i) for i in S2_w0_1]


with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\S2_w0=0_5.txt') as fobj:
    S2_w0_0_5 = fobj.read()
S2_w0_0_5 = re.split('\n', S2_w0_0_5)
S2_w0_0_5 = [float(i) for i in S2_w0_0_5]


with open('C:\\Users\\LEC\\Desktop\\S2 y # de partic\\S2_w0=0_25.txt') as fobj:
    S2_w0_0_25 = fobj.read()
S2_w0_0_25 = re.split('\n', S2_w0_0_25)
#S2_binding_puro_ajustado_por_dif.remove('')
S2_w0_0_25 = [float(i) for i in S2_w0_0_25]




plt.figure()
#==============================================================================
#                                w0 = 1
#==============================================================================   
#w0_1_mediana = np.median(S2_w0_1)
#W0_1_promedio = np.mean(S2_w0_1)
#w0_1_moda = mode (S2_w0_1)

plt.plot([1]*10,S2_w0_1,'r*',label='w0 = 1')
#plt.plot(1,w0_1_mediana,'rt',label='w0 = 1 - mediana')
#plt.plot(1,w0_1_mediana,'rs',label='w0 = 1 - mean')

#==============================================================================
#                                w0 = 0.5
#==============================================================================   




#==============================================================================
#                                w0 = 0.25
#==============================================================================   


## Dataset:
#a = pd.DataFrame({ 'group' : '1', 'value': S2_w0_1 })
#b = pd.DataFrame({ 'group' : '0.5', 'value': S2_w0_0_5 })
#
#df=a.append(b)
# 
## Usual boxplot
#sns.boxplot(x='group', y='value', data=df)
#


plt.plot([0.5]*10,S2_w0_0_5,'g*',label='w0 = 0.5')
plt.plot([0.25]*10,S2_w0_0_25,'b*',label='w0 = 0.25')
plt.legend()
plt.show()
