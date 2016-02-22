# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:44:11 2016

@author: clyde
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.integrate import odeint
from lmfit import minimize, Parameters

class UltraFast_TB(object):
    def __init__(self, times=None,traces=None,wavelengths=None, 
                 reaction_matrix=None,guess_ks=None,c0=None):
        self.times = times
        self.traces = traces
        self.wavelengths = wavelengths
        self.reaction_matrix = reaction_matrix
        self.guess_ks = guess_ks
        self.c0 = c0
        
        self.last_residuals = None
        self.fitted_ks = None
        self.fitted_C = None
        self.fitted_traces = None
        self.fitted_spectra = None
    
        self.resampled_times = None
        self.output = None
        
    def apply_svd(self, n):
        """
        Replaces spectral traces with their SVD transformed equivalents
        truncating at the nth component
        """
    
        ## need to handle svd sensibly if we have multiple traces
        ## fitting multiple traces simultaneously requires they all have the
        ## same basis so we pick the first trace to define the basis
        #svd_trace, s, self.rs_vectors = np.linalg.svd(self.traces[0], full_matrices=True)
        #transformed_traces = [svd_trace[:,:n]]
        #if len(self.traces > 1):
        #    # haven't tested this at all it's probably a bug filled mess
        #    # idea is to represent all the traces with the principle components
        #    # defined by the first set of traces
        #    transformed_traces += [self.rs_vectors.dot(t)[:,:n] for t in self.traces[1:]] 

        # Maybe it's ok to transform every trace individually -the fact that the resultant
        # traces will have different bases shouldn't affect the fitting process

        transformed_traces = []
        for trace in self.traces:
            U,s,V = np.linalg.svd(trace, full_matrices=True)
            transformed_traces.append(U[:,:n])
        self.traces = transformed_traces
        
    def get_spectra(self, conc_traces,spectral_trace):
        """Extraction of predicted spectra given concentration traces and spectral_traces"""
        # linear fit of the fitted_concs to the spectra CANNOT fit intercept here!
        l=LinearRegression(fit_intercept=False)
        l.fit(conc_traces,spectral_trace)
        fitted_spectra = l.coef_
        return fitted_spectra
    
    
    def get_traces(self, conc_traces, spectral_trace):
        """Extraction of fitted spectral traces given concentration traces and spectral traces"""
        # linear fit of the fitted_concs to the spectra CANNOT fit intercept here!
        l=LinearRegression(fit_intercept=False)
        l.fit(conc_traces,spectral_trace)
        fitted_spectral_traces = l.predict(conc_traces)
        return fitted_spectral_traces
     
        
    def dc_dt(self,C,t,K):
        """
        Rate function for the given reaction matrix.
        
        Columns of the reaction matrix correspond reactant species
        Rows of the reaction correspond to product species
        
        e.g. reaction_matrix = [[0, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0]]
              
        Corresponds to the reaction scheme A->B->C.
        
        The generated rate function has three arguments:
            C an array of floats giving the concentration of each species
            t a float giving the current time (not used but necessary for ODEs)
            K an lmfit Parameters object defining with float values 
            representing the rate constants.
        
        And returns:
            dc/dt an array floats corresponding to the derivative of the concentration
            of each species with time at time=t
            
        The above example reaction matrix would give rise to dc/dt = [-k1[A], k1[A]-k2[B], k2[B]]
        """
        
        # dc/dt built up by separately computing the positive and negative contributions.
        # In our example positive_dcdt = [0, k1[A], k2[B]]  and negative_dcdt = [-k1[A],-k2[B],0]
        reaction_matrix = np.array(self.reaction_matrix,dtype=np.int)
        C = np.array(C)
        K = np.array(K.valuesdict().values())
        
        # need to be careful about dtypes here:
        # reaction matrix dtype is int, rate matrix must be dtype float
        rate_matrix = reaction_matrix.copy()
        rate_matrix.dtype=np.float64
        rate_matrix[reaction_matrix==1] = K
        
        positive_dcdt = rate_matrix.dot(C)
        negative_dcdt = rate_matrix.sum(axis=0)*C
            
        return positive_dcdt - negative_dcdt
       
       
   # define a function that will measure the error between the fit and the real data:
    def errfunc(self, K):
        """
        Master error function
        
        Computes residuals for a given rate function, rate constants and initial concentrations
        by linearly fitting the integrated concentrations to the provided spectra.
        
        As we wish to simultaneously fit multiple data sets T and S contain multiple
        arrays of times and spectral traces respectively.
        
        K an lmfit Parameters object representing the rate constants.
        
        implit dependence on:
        self.c0 - an array of initial conditions, so that c(t0) = c0
        self.dc_dt - function to be integrated by odeint to get the concentrations
        
        self.times - the array over which integration occurs
                     since we have several data sets and each one has its own 
                     array of time points, self.times is an array of arrays.
        self.traces - the spectral data, it is an array of arrays
        """
        self.residuals = []    
    
        # compute the residuals for each array of times/spectral traces
        for times,spectral_trace in zip(self.times,self.traces):
            fitted_conc_traces = odeint(self.dc_dt, self.c0, times, args=(K,))
            fitted_spectral_traces = self.get_traces(fitted_conc_traces, spectral_trace)
            
            #residuals for the time traces of each of the species
            self.residuals += [fitted_spectral_traces[:,i] - spectral_trace[:,i] for i in range(spectral_trace.shape[1])]

        all_residuals = np.hstack(self.residuals).ravel()
        
        return all_residuals
        
        
    def fit(self):
        """Master fitting function"""
                       
        self.output = minimize(self.errfunc, self.guess_ks)
        fitted_k = self.output.params
        
        fitted_C = [odeint(self.dc_dt,self.c0,t,
                           args=(fitted_k,)) for t in self.times]
       
        self.fitted_traces = [self.get_traces(c, t) for c,t in zip(fitted_C,
                                                                  self.traces)]
        self.fitted_spectra = [self.get_spectra(c, t) for c,t in zip(fitted_C,
                                                                  self.traces)]
                
        self.fitted_ks = fitted_k
        self.fitted_C = fitted_C        
        
    def fit_sequential(self, no_species):
        """
        Utility function to fit assuming a sequential reaction model
        
        Sets the reaction matrix up for a sequential model then calls the 
        master fit() method
        """
        
        self.reaction_matrix = np.zeros([no_species, no_species])
        
        for i in range(no_species-1):       
            self.reaction_matrix[i,i+1] = 1
            
        if self.c0 is None:
            # if init concentrations not specified assume first component 
            # starts at concentration 1 and all others start at concentration 0
            self.c0 = np.zeros(no_species)
            self.c0[0] = 1
        
        self.fit()
    
    def fit_parallel(self, no_species):
        """
        Utility function to fit assuming a parallel reaction model
        
        Sets the reaction matrix up for a parallel model then calls the 
        master fit() method
        """
        
        self.reaction_matrix = np.zeros([no_species, no_species])
        
        for i in range(0,no_species,2):
            self.reaction_matrix[i,i+1] = 1
        
        if self.c0 is None:
            self.c0 = np.zeros(no_species)
            
            for i in range(0,no_species,2):
                self.c0[i] = 1
            
        self.fit()        
    
        
    def plot_traces(self, ind=None, semilog=False):
        """Utility plotting function"""
        
        if self.wavelengths is None:
            self.wavelengths = [range(s.shape[0]) for s in self.fitted_spectra]
        
        if ind is None:
            fitted_traces_to_plot = self.fitted_traces
            orig_traces_to_plot = self.traces
            times_to_plot = self.times
            wavelengths_to_plot = self.wavelengths
        else:
            fitted_traces_to_plot = self.fitted_traces[ind:ind+1]
            orig_traces_to_plot = self.traces[ind:ind+1]
            times_to_plot = self.times[ind:ind+1]
            wavelengths_to_plot = self.wavelengths[ind:ind+1]
         
        if semilog:
            plot_f = plt.semilogx
        else:
            plot_f = plt.plot
            
        for orig_traces,fit_traces,times,wlengths in zip(orig_traces_to_plot,
                                                         fitted_traces_to_plot,
                                                         times_to_plot,
                                                         wavelengths_to_plot):
                                                     
            for i,wavelength_traces in enumerate(zip(orig_traces.T,
                                                     fit_traces.T)):
                if self.wavelengths is not None:
                    wavelength = '{:.2f}'.format(wlengths[i])
                else:
                    wavelength = ''
                
                plt.figure(figsize=(3,3))
                plot_f(times,wavelength_traces[0],marker='o',linestyle='')
                plot_f(times,wavelength_traces[1],label=wavelength)
                
                plt.legend()
            plt.show()

    def plot_spectra(self, ind=None):
        """Utility plotting function"""
        
        # hopefully we need less than 26 species!
        species = iter('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        if self.wavelengths is None:
            self.wavelengths = [range(s.shape[0]) for s in self.fitted_spectra]

        if ind is None:
            spectra_to_plot = self.fitted_spectra
            wavelengths_to_plot = self.wavelengths
        else:
            spectra_to_plot = self.fitted_spectra[ind:ind+1]
            wavelengths_to_plot = self.wavelengths[ind:ind+1]
        
        for wlengths, spectra in zip(wavelengths_to_plot, spectra_to_plot):
            
            plt.figure()
            for spectrum in spectra.T:           
                plt.plot(wlengths, spectrum,label=species.next())
            plt.legend()
            plt.show()
        
    def plot_concentrations(self):
        """Utility plotting function"""
        
        # hopefully we need less than 26 species!
        species = iter('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        # resample times so that we get a good curve
        no_points = max([len(t) for t in self.times])
        max_time = max(np.ravel(self.times))
        resampled_times = np.linspace(0, max_time, no_points*5)        
        
        resampled_C = odeint(self.dc_dt, self.c0, resampled_times, 
                               args=(self.fitted_ks,))
        
        for c in resampled_C.T:
            plt.plot(resampled_times, c, label=species.next())
            
        plt.legend()
        plt.show()
        
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    test_spectra = np.loadtxt('test_spectra.csv',delimiter=',')
    test_times = [np.loadtxt('test_time.csv',delimiter=',')]
    test_traces = [np.loadtxt('test_trace.csv',delimiter=',')]
    test_wavelengths = [np.loadtxt('test_wavelengths.csv',delimiter=',')]

    reaction_matrix = [[0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0]]
    c0 = [1,0,0]
    
    k_guess = Parameters()
    #                (Name, Value, Vary, Min, Max,  Expr)
    k_guess.add_many(('k1', 0.1,   True, None,   None, None),
                     ('k2', 0.1,   True, None,   None, None))
                     
    tb = UltraFast_TB(test_times, test_traces, test_wavelengths, 
                      reaction_matrix, k_guess, c0)
    tb.apply_svd(n=3)    
    tb.fit()
    
    k_fit = tb.fitted_ks
    k_fit['k1'].vary = False
    k_fit['k2'].vary = False
    
    tb = UltraFast_TB(test_times, test_traces, test_wavelengths, 
                      reaction_matrix, k_fit, c0)
    tb.fit()

    tb.plot_spectra()
    tb.plot_traces()
    tb.plot_concentrations()
    
    print([k.value for k in k_fit.values()])