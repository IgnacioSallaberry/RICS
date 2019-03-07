

import numpy as np
import matplotlib.pyplot as plt
from acf_fitting import Acf_Fit

'''Por ahora, asume que ya se tiene la función de autocorrelacion como la exporta el SimFCS (tau vs G(tau))'''

#Inicia objeto, en la forma Acf_Fit( 'PATH', microscope= '2-photon'(default)-'1-photon'-'confocal', simulation=False(default), binding=None(default)
difusion_test = Acf_Fit('difusion_test3', '1-photon')

#Grafico de los datos experimentales obtenidos
#difusion_test.raw_graph()


'''Fitting''' # Por ahora se pueden proponer hasta 2 modelos en simultáneo
'Fit1'
#Default fittings are: diffusive,  2-diffusive, binding, diffusive-binding
f = lambda x, D, H: x**2+8*D+H


difusion_test.perform_fit([0.28, 0.25], [5,1.0])
print(difusion_test.popt1)
difusion_test.perform_fit2([0.1,0.1],[0,0])
print(difusion_test.popt2)
difusion_test.graph_fit()
