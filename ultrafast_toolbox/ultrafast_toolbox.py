# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:44:11 2016

@author: clyde
"""

#import scipy
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import Imputer
from scipy.integrate import odeint
from lmfit import minimize, Parameters
       
class UltraFast_TB(object):
    def __init__(self, times=None,traces=None,wavelengths=None, 
                 input_params=None,reaction_matrix=None,c0=None, 
                 method='leastsq',alpha=0,gamma=0):
                             
        self.times = times
        self.traces = traces
        self.wavelengths = wavelengths
        self.reaction_matrix = reaction_matrix
        self.input_params = input_params
        self.c0 = c0
         
        self.last_residuals = None
        self.fitted_ks = None
        self.fitted_C = None
        self.fitted_traces = None
        self.fitted_spectra = None
    
        self.resampled_C = None
        self.resampled_times = None
        self.output = None
    
        self.method = method
        if alpha:
            self.regressor = Ridge(fit_intercept=False,alpha=alpha)
        elif gamma:
            self.regressor = Lasso(fit_intercept=False,alpha=gamma)
        else:
            self.regressor = LinearRegression(fit_intercept=False)
            
        # if we are fitting against multiple traces they must be measured at
        # the same wavelengths. Note if we happen to be measuring at the same
        # number of wavelengths but the wavelengths being measured are 
        # different we wil pass this test but the results will still be 
        # meaningless
        no_wavlengths_measured = [st.shape[1] for st in self.traces]
        
        assert len(set(no_wavlengths_measured)) == 1
        
    def apply_svd(self, n):
        """
        Replaces spectral traces with their SVD transformed equivalents
        truncating at the nth component
        """
    
        ## should really handle svd sensibly if we have multiple traces
        ## fitting multiple traces simultaneously requires they all have the
        ## same basis. Could pick the first trace to define the basis
        #svd_trace, s, self.rs_vectors = np.linalg.svd(self.traces[0], full_matrices=True)
        #transformed_traces = [svd_trace[:,:n]]
        #if len(self.traces > 1):
        #    # haven't tested this at all it's probably a bug filled mess
        #    # idea is to represent all the traces with the principle components
        #    # defined by the first set of traces
        #    transformed_traces += [self.rs_vectors.dot(t)[:,:n] for t in self.traces[1:]] 

        # or look for svd like transformation to apply the the entire block of traces?

        # either way current approach is totally dodgey if fitting against 
        # multiple svd transformed traces

        transformed_traces = []
        # wavelengths now correspond to principle components
        
        for trace in self.traces:
            U,s,V = np.linalg.svd(trace, full_matrices=True)
            transformed_traces.append(U[:,:n])
        
        self.traces = transformed_traces
        self.wavelengths = np.arange(n)
        
    def get_spectra(self, conc_traces,spectral_trace):
        """Extraction of predicted spectra given concentration traces and spectral_traces"""
        # linear fit of the fitted_concs to the spectra CANNOT fit intercept here!
        self.regressor.fit(conc_traces,spectral_trace)
        fitted_spectra = self.regressor.coef_
        return fitted_spectra
    
    
    def get_traces(self, conc_traces, spectra):
        """Extraction of fitted spectral traces given concentration traces and spectral traces"""
        # linear fit of the fitted_concs to the spectra CANNOT fit intercept here!
        #self.regressor.fit(conc_traces,spectral_trace)
        #fitted_spectral_traces = self.regressor.predict(conc_traces)
        fitted_spectral_traces = spectra.dot(conc_traces.T)        
        return fitted_spectral_traces.T
     
    
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
        #K = np.array(K.valuesdict().values())

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
        
        ## could switch to something like ode15s that the oiginal matlab code 
        ## uses - can odeint cope with equations as stiff as we need?
        ## to use integrrate.ode need order of arguments in dc_dt to switch
        
        #r = scipy.integrate.ode(self.dc_dt)
        #r = r.set_integrator('vode', method='bdf', order=15,nsteps=3000)
        #r = r.set_initial_value(self.c0)
        #r = r.set_f_params((K,))
        #r.integrate(t)
        
        static_times = t[t<0]
        dynamic_times = t[t>=0]

        static_C = np.array([self.c0 for _ in static_times])

        # odeint always takes the first time point as t0
        # our t0 is always 0 (removing t0 occures before we integrate)
        # so if the first time point is not 0 we add it 
                
        if dynamic_times[0]:
            #fancy indexing returns a copy so we can do this
            dynamic_times = np.hstack([[0],dynamic_times])            
            dynamic_C = odeint(self.dc_dt,self.c0,dynamic_times,args=(K,))[1:]
        else:
            dynamic_C = odeint(self.dc_dt,self.c0,dynamic_times,args=(K,))
            
        if static_C.any():
            return np.vstack([static_C,dynamic_C])
        else:
            return dynamic_C
   
   # define a function that will measure the error between the fit and the real data:
    def errfunc(self, params):
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
        
        # rate constants must be in order i.e. k1 must be added before k2
        rate_constant_strs = [key for key in params if 'k' in key]
        K_params = [params[k] for k in rate_constant_strs]
        K = np.array([k_param.value for k_param in K_params])
        
        # t0 values must be in order i.e. t01 before t02
        T0_strs = [key for key in params if 't0' in key]
        T0_params = [params[key] for key in T0_strs]
        T0 = np.array([t0_param.value for t0_param in T0_params])
        
        # OD_offset values must be in order i.e. OD_offset1 before OD_offset2
        OD_strs = [key for key in params if 'OD_offset' in key]
        OD_params = [params[key] for key in OD_strs]
        OD_offset = np.array([od_param.value for od_param in OD_params])
        
        offset_times = [t-t0 for t,t0 in zip(self.times,T0)]
        offset_traces = [st - od for st,od in zip(self.traces,OD_offset)]

        # calculated concentrations for the different time sets
        fitted_conc_traces = []   
        for t in offset_times:
            conc_traces = self.C(t,K)

            if np.isnan(conc_traces).any():
                fix = Imputer(missing_values='NaN', strategy='median',axis=0) 
                conc_traces  = fix.fit_transform(conc_traces )
                warnings.warn('Nan found in predicted concentrations')

            fitted_conc_traces.append(conc_traces)
        
        # spectra fitted against all data sets
        # REQUIRES spectral traces to be measured at the SAME WAVELENGTHS!
        
        fitted_spectra = self.get_spectra(np.hstack(fitted_conc_traces),
                                          np.hstack(offset_traces))
                                          
        fitted_spectral_traces = [self.get_traces(c, fitted_spectra) for c in
                                        fitted_conc_traces]
            
        self.residuals = [fst -t for fst,t in zip(fitted_spectral_traces,
                                                  offset_traces)]
            
        all_residuals = np.hstack(self.residuals).ravel()
        
        return all_residuals
        
#        # compute the residuals for each dataset of times/traces
#        for times,traces in zip(self.times,self.traces):
#            offset_times = times-t0
#            fitted_conc_traces = self.C(offset_times, K)
#            
#            # handle case where we have poor parameters causing concentrations
#            # that are higher than floating point allows by replacing them with
#            # the  median concentration for that species.
#            # We expect these instances to be a very poor fit and hence that 
#            # this procedure will not affect the final fitted rate constants
#            if np.isnan(fitted_conc_traces).any():
#                fix = Imputer(missing_values='NaN', strategy='median',axis=0) 
#                fitted_conc_traces  = fix.fit_transform(fitted_conc_traces )
#                warnings.warn('Nan found in predicted concentrations')
#                
#            offset_traces = traces - OD_offset
#            fitted_spectra = self.get_spectra(fitted_conc_traces,
#                                              offset_traces)
#            fitted_spectral_traces = self.get_traces(fitted_conc_traces, 
#                                                     fitted_spectra)
#            current_residuals = fitted_spectral_traces - traces
#            self.residuals.append(current_residuals)
# 
#        # do we need to worry about the order of the flattened residuals? 
#        # e.g. if switch current_residuals for current_residuals.T 
#        # would it matter?
# 
#        # combine residuals for each data set and flatten
#
#        all_residuals = np.hstack(self.residuals).ravel()
#        
#        return all_residuals
    
    def printfunc(self, params, iter, resid, *args, **kwargs):
        """
        Method passed to minimize if we are debugging to print out the
        values of the parameters as minimisation is occuring
        """

        print(iter)  
        print(params.valuesdict())
                    
    def fit(self, debug=False):
        """Master fitting function"""
                    
        if debug:
            self.output = minimize(self.errfunc, self.input_params, 
                                   method=self.method,iter_cb=self.printfunc)
        else:
            self.output = minimize(self.errfunc, self.input_params,
                                   method=self.method)

        fitted_params = self.output.params

        rate_constant_strs = [key for key in fitted_params if 'k' in key]        
        fitted_k_params = [fitted_params[key] for key in rate_constant_strs]
        fitted_k = np.array([k_param.value for k_param in fitted_k_params])
              
        T0_strs = [key for key in fitted_params if 't0' in key]
        fitted_T0_params = [fitted_params[key] for key in T0_strs]
        fitted_T0 = np.array([t0_param.value for t0_param in fitted_T0_params])
        
        OD_strs = [key for key in fitted_params if 'OD_offset' in key]
        fitted_OD_params = [fitted_params[key] for key in OD_strs]
        fitted_OD = np.array([od_param.value for od_param in fitted_OD_params])

        offset_traces = [traces - od for traces,od in zip(self.traces,
                                                          fitted_OD)]
                                                        
        offset_times = [times - t0 for times,t0 in zip(self.times, 
                                                       fitted_T0)]
                                                       
        fitted_C = [self.C(t, fitted_k) for t in offset_times]
       
        fitted_spectra = self.get_spectra(np.hstack(fitted_C),
                                          np.hstack(offset_traces))
                                          
        
        fitted_traces = [self.get_traces(c, fitted_spectra) for c in fitted_C]
         
        self.fitted_spectra = fitted_spectra
        self.fitted_traces = fitted_traces
        self.fitted_ks = fitted_k
        self.fitted_t0 = fitted_T0
        self.fitted_OD_offset = fitted_OD
        self.fitted_C = fitted_C  
        
        # create master resampled data
        no_points = max([len(t) for t in offset_times])
        max_time = max(np.ravel(offset_times))
        min_time = min(np.ravel(offset_times))
        
        if min_time > 0:
            min_time = 0
        
        resampled_offset_times = np.linspace(min_time, max_time, no_points*5)
        
        self.resampled_C = self.C(resampled_offset_times,self.fitted_ks)
        self.resampled_traces = self.get_traces(self.resampled_C, 
                                                self.fitted_spectra)
                                                
        # for resampled times we arbitrarily choose to align with the
        # first dataset
        self.resampled_times = resampled_offset_times + self.fitted_t0[0]
        
    def fit_sequential(self, no_species, debug=False):
        """
        Utility function to fit assuming a sequential reaction model
        
        Sets the reaction matrix up for a sequential model then calls the 
        master fit() method
        """
        
        self.reaction_matrix = np.zeros([no_species, no_species])
        
        no_datasets = len(self.traces)
        
        for i in range(no_species-1):       
            self.reaction_matrix[i,i+1] = 1
            
        if self.c0 is None:
            # if init concentrations not specified assume first component 
            # starts at concentration 1 and all others start at concentration 0
            self.c0 = np.zeros(no_species)
            self.c0[0] = 1
        
        if self.input_params is None:
            self.input_params = Parameters()
            
        if not any(['k' in key for key in self.input_params.valuesdict()]):
            rate_constants = [('k{i}'.format(i=n),0.1,True,0,None,None) 
                               for n in range(1,no_species)]
            self.input_params.add_many(*rate_constants)
          
        if not  any(['t0' in key for key in self.input_params.valuesdict()]):
            t0 = [('t0{i}'.format(i=n),0,False,None,None,None)
                  for n in range(no_datasets)]
            self.input_params.add_many(*t0)
                   
        if not  any(['OD' in key for key in self.input_params.valuesdict()]):
            OD_offset = [('OD_offset{i}'.format(i=n),0,False,None,None,None)
                         for n in range(no_datasets)]
            self.input_params.add_many(*OD_offset)
            
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
        
        if self.input_params is None:
            self.input_params = Parameters()
        # if no rate constants set, assign n-1 rate constants to a default
        # of 0.1
        if not any(['k' in key for key in self.input_params.valuesdict()]):
            rate_constants = [('k{i}'.format(i=n),0.1,True,0,None,None) 
                              for n in range(1,no_species,2)]
            self.input_params.add_many(*rate_constants)

        if not  any(['t0' in key for key in self.input_params.valuesdict()]):
            t0 = [('t0{i}'.format(i=n),0,False,None,None,None)
                  for n in range(no_datasets)]
            self.input_params.add_many(*t0)
                    
        if not  any(['OD' in key for key in self.input_params.valuesdict()]):
            OD_offset = [('OD_offset{i}'.format(i=n),0,False,None,None,None)
                         for n in range(no_datasets)]
            self.input_para,s.add_many(*OD_offset)

            
        self.fit(debug)        
    
    def tex_reaction_scheme(self):
        """Returns a Latex representation of the current reaction scheme"""
        
        if self.reaction_matrix is None or self.input_params is None:
            return 'undefined'
            
        species = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        eqn = []
        
        reactants, products = self.reaction_matrix.nonzero()
        for r,p,k in zip(reactants, products,self.input_params.keys()):
            eqn.append( species[r] + r'\xrightarrow{{' + k + '}}' + species[p])
        return '$' + ','.join(eqn) + '$'
        
        
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from plot_utils import *
    
    test_spectra = np.loadtxt('test_spectra.csv',delimiter=',')
    test_times = [np.loadtxt('test_time.csv',delimiter=',')]
    test_traces = [np.loadtxt('test_trace.csv',delimiter=',')]
    test_wavelengths = np.loadtxt('test_wavelengths.csv',delimiter=',')

#    reaction_matrix = [[0, 1, 0],
#                       [0, 0, 1],
#                       [0, 0, 0]]
#    c0 = [1,0,0]
#    
    #k_guess = Parameters()
    #                 (Name,        Value, Vary,  Min,    Max,  Expr)
    #k_guess.add_many(('k1',        0.1,   True,  None,   None, None),
    #                 ('k2',        0.1,   True,  None,   None, None),
    #                 ('t0',        0,     False, None,   None, None),
    #                 ('OD_offset', 0,     False, None,   None, None))
                     
    tb = UltraFast_TB(test_times, test_traces, test_wavelengths, alpha=1e-5,
                     )#k_guess , reaction_matrix, c0)
    tb.apply_svd(n=3)    
    tb.fit_sequential(3,debug=True)
    
    fitted_params = tb.output.params
    for k in fitted_params.values():
        k.vary=False
    
    tb = UltraFast_TB(test_times, test_traces, test_wavelengths, 
                      fitted_params)#, reaction_matrix, c0)
    tb.fit_sequential(3)

    #plot_spectra(tb)
    #plot_traces(tb)
    #plot_concentrations(tb)
    
    plot_master(tb,trace_wavelengths=(125,150,175,230,275))
    
    print([k.value for k in fitted_params.values()])
    
    plt.plot(test_spectra)
    plt.show()
