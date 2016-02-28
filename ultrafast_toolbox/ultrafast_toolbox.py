# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:44:11 2016

@author: clyde
"""

import numpy as np
import warnings
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import Imputer
from scipy.integrate import odeint
from lmfit import minimize, Parameters
       
class UltraFast_TB(object):
    def __init__(self, times=None,traces=None,wavelengths=None, 
                 guess_ks=None,reaction_matrix=None,c0=None, alpha=0):
                             
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
    
        self.resampled_C = None
        self.resampled_times = None
        self.output = None
    
        self.extras = {}
        
        if alpha:
            self.regressor = Ridge(fit_intercept=False,alpha=alpha) 
        else:
            self.regressor = LinearRegression(fit_intercept=False)
            
        
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
        # wavelengths now correspond to principle components
        transformed_wavelengths = []
        
        for trace in self.traces:
            U,s,V = np.linalg.svd(trace, full_matrices=True)
            transformed_traces.append(U[:,:n])
            transformed_wavelengths.append(np.arange(n))
        
        self.traces = transformed_traces
        self.wavelengths = transformed_wavelengths        
        
    def get_spectra(self, conc_traces,spectral_trace):
        """Extraction of predicted spectra given concentration traces and spectral_traces"""
        # linear fit of the fitted_concs to the spectra CANNOT fit intercept here!
        self.regressor.fit(conc_traces,spectral_trace)
        fitted_spectra = self.regressor.coef_
        return fitted_spectra
    
    
    def get_traces(self, conc_traces, spectral_trace):
        """Extraction of fitted spectral traces given concentration traces and spectral traces"""
        # linear fit of the fitted_concs to the spectra CANNOT fit intercept here!
        self.regressor.fit(conc_traces,spectral_trace)
        fitted_spectral_traces = self.regressor.predict(conc_traces)
        return fitted_spectral_traces
     
    
    def dc_dt(self,C,t,K):
        """
        Rate function for the given reaction matrix.
        
        Rows of the reaction matrix correspond reactant species
        Columns of the reaction correspond to product species
        
        e.g. reaction_matrix = [[0, 1, 0],
                                [0, 0, 1],
                                [0, 0, 0]]
              
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

        # need to have the same number of rate parameters in K
        # as indicated in reaction_matrix!
        assert len(K) == np.sum(reaction_matrix)
        
        # need to be careful about dtypes:
        # reaction matrix dtype is int, rate matrix must be dtype float
        rate_matrix = reaction_matrix.copy()
        rate_matrix.dtype=np.float64
        rate_matrix[reaction_matrix==1] = K
        
        positive_dcdt = rate_matrix.T.dot(C)
        negative_dcdt = rate_matrix.T.sum(axis=0)*C
            
        return positive_dcdt - negative_dcdt
       
        
    def C(self,t,K):
        """
        Concentration function returns concentrations at the times given in t
        Uses odeint to integrate dc/dt using rate constants k over times t.
        Implicitly uses self.c0 and self.dc_dt
        """
        #ode(self.dc_dt,self.c0,t,args=(k,)).set_integrator('lsoda')
        #ode(self.dc_dt,self.c0,t,args=(k,)).set_integrator('vode', method='bdf', order=15)
        
        # if we have any negative times we assume they occur before the 
        # reaction starts hence all negative times are assigned concentration 
        # c0
        
        static_C = np.array([self.c0 for _ in t[t<0]])
        dynamic_C = odeint(self.dc_dt,self.c0,t[t>=0],args=(K,))
        
        if static_C.any():
            return np.vstack([static_C,dynamic_C])
        else:
            return dynamic_C
   
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
        self.traces - the spectral data, it is an array of array of arrays
        """
        
        self.residuals = []    
        # compute the residuals for each dataset of times/traces
        for times,traces in zip(self.times,self.traces):
            fitted_conc_traces = self.C(times, K)
            
            if np.isnan(fitted_conc_traces).any():
                fix = Imputer(missing_values='NaN', strategy='median',axis=0) 
                fitted_conc_traces  = fix.fit_transform(fitted_conc_traces )
                warnings.warn('Nan found in predicted concentrations')
                
            fitted_spectral_traces = self.get_traces(fitted_conc_traces, traces)
            current_residuals = fitted_spectral_traces - traces
            self.residuals.append(current_residuals)
 
        # do we need to worry about the order of the flattened residuals? 
        # e.g. if switch current_residuals for current_residuals.T 
        # would it matter?
 
        # combine residuals for each data set and flatten
        all_residuals = np.hstack(self.residuals).ravel()
        
        return all_residuals
    
    def debug_fit(self, params,iter,resid,*args,**kwargs):
        """Method passed to minimize if we are debugging"""

        print(iter)  
        print(params.valuesdict())
                    
    def fit(self, debug=False):
        """Master fitting function"""
                    
        if debug:
            self.output = minimize(self.errfunc, self.guess_ks, 
                                 iter_cb=self.debug_fit)
        else:
            self.output = minimize(self.errfunc, self.guess_ks)

        fitted_k = self.output.params
        
        fitted_C = [self.C(t,fitted_k) for t in self.times]
       
        self.fitted_traces = [self.get_traces(c, t) for c,t in zip(fitted_C,
                                                                  self.traces)]
        self.fitted_spectra = [self.get_spectra(c, t) for c,t in zip(fitted_C,
                                                                  self.traces)]
                
        self.fitted_ks = fitted_k
        self.fitted_C = fitted_C  
        
        
        # create master resampled concentration data
        no_points = max([len(t) for t in self.times])
        max_time = max(np.ravel(self.times))
        self.resampled_times = np.linspace(0, max_time, no_points*5)        
        
        self.resampled_C = self.C(self.resampled_times,self.fitted_ks)

        
    def fit_sequential(self, no_species, debug=False):
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
        
        if self.guess_ks is None:
            guess_ks = Parameters()
            guess_ks.add_many(*[('k{i}'.format(i=n),0.1,True,0,None,None) 
                               for n in range(1,no_species)])
            self.guess_ks = guess_ks
            
        self.fit(debug)
    
    def fit_parallel(self, no_species,debug=False):
        """
        Utility function to fit assuming a parallel reaction model
        
        Sets the reaction matrix up for a parallel model then calls the 
        master fit() method
        """
        
        self.reaction_matrix = np.zeros([no_species, no_species])
        
        for i in range(0,no_species-1,2):
            self.reaction_matrix[i,i+1] = 1
        
        if self.c0 is None:
            self.c0 = np.zeros(no_species)
            
            for i in range(0,no_species-1,2):
                self.c0[i] = 1
        
        if self.guess_ks is None:
            guess_ks = Parameters()
            guess_ks.add_many(*[('k{i}'.format(i=n),0.1,True,0,None,None) 
                               for n in range(1,no_species,2)])
            self.guess_ks = guess_ks
        self.fit(debug)        
    
    def tex_reaction_scheme(self):
        """Returns a Latex representation of the current reaction scheme"""
        
        if self.reaction_matrix is None or self.guess_ks is None:
            return 'undefined'
            
        species = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        eqn = []
        
        reactants, products = self.reaction_matrix.nonzero()
        for r,p,k in zip(reactants, products,self.guess_ks.keys()):
            eqn.append( species[r] + r'\xrightarrow{{' + k + '}}' + species[p])
        return '$' + ','.join(eqn) + '$'
        
        
if __name__ == '__main__':
    
    from plot_utils import plot_spectra, plot_traces, plot_concentrations
    
    test_spectra = np.loadtxt('test_spectra.csv',delimiter=',')
    test_times = [np.loadtxt('test_time.csv',delimiter=',')]
    test_traces = [np.loadtxt('test_trace.csv',delimiter=',')]
    test_wavelengths = [np.loadtxt('test_wavelengths.csv',delimiter=',')]

#    reaction_matrix = [[0, 1, 0],
#                       [0, 0, 1],
#                       [0, 0, 0]]
#    c0 = [1,0,0]
#    
    #k_guess = Parameters()
    #                (Name, Value, Vary, Min, Max,  Expr)
    #k_guess.add_many(('k1', 0.1,   True, None,   None, None),
    #                 ('k2', 0.1,   True, None,   None, None))
                     
    tb = UltraFast_TB(test_times, test_traces, test_wavelengths, 
                     )#k_guess , reaction_matrix, c0)
    tb.apply_svd(n=3)    
    tb.fit_sequential(3,debug=True)
    
    k_fit = tb.fitted_ks
    for k in k_fit.values():
        k.vary=False
    
    tb = UltraFast_TB(test_times, test_traces, test_wavelengths, 
                      k_fit)#, reaction_matrix, c0)
    tb.fit_sequential(3)

    plot_spectra(tb)
    #plot_traces(tb)
    plot_concentrations(tb)
    
    print([k.value for k in k_fit.values()])