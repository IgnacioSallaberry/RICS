# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:02:07 2018

@author: FER
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
sns.set(style='darkgrid', palette='muted')


class Acf_Fit():

    def __init__(self, path_data, size=128, microscope='2-photon', type='FCS', simulation=False):
        '''
        init function contains information on the data collected from experiment
        :param data: <type: string> input path to data, stores the experiments data in the form of numpy array
        :param size: <type: int> image size
        :param microscope: <type: string> what microscope was used during experiment, this affects later formulas used for fitting.\n
        Supported microscopes are 1-photon, confocal and 2-photon (default)
        :param type: <type: string> type of experiment where measurements were taken. Supported types are FCS (default) - will imporove
        :param simulation: <type: boolean> whether data comes from simulation or real experiment
        '''
        self.raw_data = np.loadtxt(path_data, skiprows=6)
        self.size = size
        self.data = self.raw_data
        self.microscope = microscope
        self.simulation = simulation
        self.type = type
        self.tau = self.data[:,0]
        self.gtau = self.data[:,1]
        self.fit1 = self.fit2 = False

    def extract_acf(self, data, size=128):
        data_pixel = data[(size*size/2) + size/2 - 1:(size*size/2) + size-1]   #toma un pedazo de los datos crudos
        data_line = [data[size*i + size/2 - 1] for i in range(size)]           #crea una lista seleccionando los valores de datos crudos desde data [63](aca i=0) hasta data[16319](aca i=128) suponiendo size = 128 
        return data_line, data_pixel                                           #data_line = [data[63],data[64],...,data[16319]]
   
    # grafico de datos crudos
    def raw_graph(self):
        '''
        Plots graph of data (raw) before analysis or any fitting function
        '''
        self.raw = plt.subplot()
        self.raw.plot(self.tau, self.gtau, 'b-')
        self.raw.set_xscale("log")   #setea scala logartimica en x
        self.raw.grid(True)
        self.raw.set(xlabel='tau', ylabel='G(tau)', title='Datos experimentales ACF')
        plt.show()

    def set_origin(self, lower_bound):
        points = [self.gtau[i] if self.tau[i] > lower_bound else 0 for i in range(len(self.tau))]   #esto es lo mismo que escribir un if... else... en vertical???
        points = np.ma.masked_equal(points,0)
        avg = np.average(points)
        self.gtau = [i-avg for i in self.gtau]


    def fit_function(self, params, fit_func='diffusive'):
        '''
        Defines the fitting function to be used if it consists of a pre-existing model. Custom functions not included here.
        :param params: Fixed parameters to be passed to function. Check each model to see which are required ones.
        :param fit_func: Type of fitting model to be used
        :return: returns a <type: function> with the fitting function to be used, fixed parameters already replaced.
        '''
        if self.microscope == '2-photon':
            self.f = 2
        elif self.microscope == 'confocal' or self.microscope == '1-photon':
            self.f = 1
        #one species diffusing model
        if fit_func =='diffusive':
            w0, wz = params
            fitting = lambda x, D, G0: G0 * ( 1 + 4*D*self.f*x / w0**2 )**(-1) * ( 1 + 4*D*self.f*x / wz**2 )**(-1/2)
            return fitting
        #two species diffusing model
        if fit_func =='2-diffusive': 
            w0, wz, f1 = params
            fitting = lambda x, D1, D2, G01, G02: f1* f1 * G01 * (1 + 4 * D1 * self.f * x / w0 ** 2) ** (-1) * (1 + 4 * D1 * self.f * x / wz ** 2) ** (-1 / 2) + \
                                       (1 - f1) * (1 - f1) * G02 * (1 + 4 * D2 * self.f * x / w0 ** 2) ** (-1) * (1 + 4 * D2 * self.f * x / wz ** 2) ** (-1 / 2)
            return fitting
        #one species of diffusing and binding model
        if fit_func =='diffusive-binding':
            w0, wz, fa, fb = params
            fitting = lambda x, K, l, G0, D:  G0 * ( 1 + 4*D*self.f*x / w0**2 )**(-1) * ( 1 + 4*D*self.f*x / wz**2 )**(-1/2) * ( 1 + K*(fa - fb/K)**2 *np.exp(-l*x))
            # Go = G(tau=o)
            # K=Kf/Kb
            # l = landa = Kf+Kb = Ka+Kb
            # x = tau = tiempo
            # fa= fluorescent fraction of particle in state a
            return  fitting
        
        #one species binding model
        if fit_func =='binding':
            fitting = lambda x, K, l: ( 1 + K*(fa - fb/K)**2 *np.exp(-l*x))
            #Digman: Paxillin Dynamics Measured during Adhesion Assembly and Disassembly by Correlation Spectroscopy
            return fitting

    def perform_fit(self, params, initial, fit_func='diffusive', custom_func=None, limit=0):
        '''
        Performs least-squares fit on data using a fitting function defined by a pre-existing model or custom.
        :param params: <type: list> Fixed parameters to be used in model, args must be passed in order (check fit_func() for info)
        :param initial: <type: list> Initial guess for parameters to be estimated
        :param fit_func: <type: string> Model to be used for fitting, set to 'custom' for custom model. Default is 'diffusive'
        :param custom_func: <type: function> Function to be passed if custom model used.
        :param limit: <type: float> Use if only want to fit data up to a certain value, ignores data above limit
        '''
        self.fit1 = True
        self.fit1_initial = initial
        self.fit1_func_name = fit_func
        self.fit_tau = self.tau
        self.fit_gtau = self.gtau
        if self.fit1_func_name != 'custom':
            self.fit1_func = self.fit_function(params, self.fit1_func_name)
        elif self.fit1_func_name == 'custom':
            self.fit1_func = custom_func
        if limit>0:
            self.fit_tau = [i for i in self.tau if i < limit]
            self.fit_gtau = self.gtau[:len(self.fit_tau)]
        self.popt1, self.pcov1 = curve_fit(self.fit1_func, self.fit_tau, self.fit_gtau, p0 = self.fit1_initial, maxfev=10**7)

    def perform_fit2(self, params, initial, fit2_func='diffusive', custom_func=None, limit=0):
        '''
        The idea behind this function is to store a second fitting model for comparison
        :param params: <type: list> Fixed parameters to be used in model, args must be passed in order (check fit_func() for info)
        :param initial: <type: list> Initial guess for parameters to be estimated
        :param fit_func: <type: string> Model to be used for fitting, set to 'custom' for custom model. Default is 'diffusive'
        :param custom_func: <type: function> Function to be passed if custom model used.
        :param limit: <type: float> Use if only want to fit data up to a certain value, ignores data above limit
        '''
        self.fit2 = True
        self.fit2_initial = initial
        self.fit2_func_name = fit2_func
        self.fit2_tau = self.tau
        self.fit2_gtau = self.gtau
        if self.fit2_func_name != 'custom':
            self.fit2_func = self.fit_function(params, self.fit2_func_name)
        elif self.fit2_func_name == 'custom':
            self.fit2_func = custom_func
        if limit > 0:
            self.fit2_tau = [i for i in self.tau if i < limit]
            self.fit2_gtau = self.gtau[:len(self.fit2_tau)]
        self.popt2, self.pcov2 = curve_fit(self.fit2_func, self.fit2_tau, self.fit2_gtau, p0=self.fit2_initial, maxfev=10**7)

    def graph_fit(self):
        '''
        plots a graph of different models used to perform fit
        '''
        self.graph_fit = plt.subplot()
        self.graph_fit.plot(self.tau, self.gtau, 'k.')

        self.x = np.logspace(np.log(min(self.tau)), np.log(max(self.tau))+10, 200)
        if self.fit1 == True:
            self.graph_fit.plot(self.x, self.fit1_func(self.x, *self.popt1), 'b-', linewidth=1, label='Process: ' + self.fit1_func_name.upper())# + ', D=28.1579')# + str(round(self.popt1[0],4)))
        if self.fit2 == True:
            self.graph_fit.plot(self.x, self.fit2_func(self.x, *self.popt2), 'r-', linewidth=0.5, label='Process: ' + self.fit2_func_name.upper())
        if self.fit1 == True or self.fit2 == True:
            self.graph_fit.legend(loc='upper right')

        self.graph_fit.set_xscale("log")
        self.graph_fit.grid(True)
        self.graph_fit.set_xlim(min(self.tau)*0.9, max(self.tau)*1.1)
        self.graph_fit.set(xlabel='Tau ($s$)', ylabel='G(Tau)', title='Datos experimentales ACF')
        plt.tight_layout()
        plt.show()

    def calc_quadratic_dev(self, fit_func, popt):
        return np.average([(self.gtau[i] - fit_func(self.tau[i], *popt))**2 for i in range(len(self.tau))])

    def calculate_aic(self, fit_func, popt):
        quad_dev = self.calc_quadratic_dev(fit_func, popt)
        return np.log(quad_dev) + 2*len(popt)/len(self.tau)

    def calculate_bic(self, fit_func, popt):
        quad_dev = self.calc_quadratic_dev(fit_func, popt)
        return np.log(quad_dev) + len(popt)*np.log(len(self.tau)) / len(self.tau)

    def model_test(self):
        '''
        Calculates AIC and BIC for up to 2 different fits, prints results.
        '''
        if self.fit1 == True:
            print('____________________________________')
            print('Fitting Model: ' + self.fit1_func_name)
            print('AIC= ', self.calculate_aic(self.fit1_func, self.popt1), '  |  BIC= ', self.calculate_bic(self.fit1_func, self.popt1))
        if self.fit2 == True:
            print('Fitting Model: ' + self.fit2_func_name)
            print('AIC= ', self.calculate_aic(self.fit2_func, self.popt2), '  |  BIC= ',
                  self.calculate_bic(self.fit2_func, self.popt2))

if __name__ == '__main__':
