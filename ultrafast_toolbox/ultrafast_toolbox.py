# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:44:11 2016

@author: clyde
"""

#import scipy
import numpy as np
import warnings
import copy
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import Imputer
from scipy.integrate import odeint
from lmfit import minimize, Parameters
       
class UltraFast_TB(object):
    def __init__(self, times=None,traces=None,wavelengths=None, 
                 input_params=None,reaction_matrix=None,
                 method='leastsq',alpha=0,gamma=0):
                             
        self.times = times
        self.traces = traces
        self.wavelengths = wavelengths
        self.reaction_matrix = reaction_matrix
        self.input_params = input_params

        try:
            self.no_species = self.reaction_matrix.shape[0]
        except AttributeError:
            self.no_species = None

        self.last_residuals = None
        self.fitted_ks = None
        self.fitted_c0 = None
        self.fitted_C = None
        self.fitted_traces = None
        self.fitted_spectra = None

        self.no_resampled_points = None    
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
       
        
    def C(self,t,K,c0):
        """
        Concentration function returns concentrations at the times given in t
        Uses odeint to integrate dc/dt using rate constants k over times t at
        initial concentrations c0
        Implicitly uses self.dc_dt
        """
        #ode(self.dc_dt,c0,t,args=(k,)).set_integrator('lsoda')
        #ode(self.dc_dt,c0,t,args=(k,)).set_integrator('vode', method='bdf', order=15)
        
        # if we have any negative times we assume they occur before the 
        # reaction starts hence all negative times are assigned concentration 
        # c0
        
        ## could switch to something like ode15s that the oiginal matlab code 
        ## uses - can odeint cope with equations as stiff as we need?
        ## to use integrate.ode need order of arguments in dc_dt to switch
        
        #r = scipy.integrate.ode(self.dc_dt)
        #r = r.set_integrator('vode', method='bdf', order=15,nsteps=3000)
        #r = r.set_initial_value(c0)
        #r = r.set_f_params((K,))
        #r.integrate(t)
        
        static_times = t[t<0]
        dynamic_times = t[t>=0]

        static_C = np.array([c0 for _ in static_times])

        # odeint always takes the first time point as t0
        # our t0 is always 0 (removing t0 occures before we integrate)
        # so if the first time point is not 0 we add it 
                
        if not dynamic_times.any() or dynamic_times[0]:
            #fancy indexing returns a copy so we can do this
            dynamic_times = np.hstack([[0],dynamic_times])            
            dynamic_C = odeint(self.dc_dt,c0,dynamic_times,args=(K,))[1:]
        else:
            dynamic_C = odeint(self.dc_dt,c0,dynamic_times,args=(K,))
            
        if static_C.any():
            return np.vstack([static_C,dynamic_C])
        else:
            return dynamic_C
   
    def _get_K(self, params):
        no_datasets = len(self.traces)
        n_Ks = int(np.sum(self.reaction_matrix))
       
        K=[]
        for d in range(1,no_datasets+1):
            k_keys = ['k{i}{d}'.format(i=i,d=d) for i in range(1,n_Ks+1)]
            dataset_K = np.array([params[key].value for key in k_keys])
            K.append(dataset_K)
        return K
    
    def _get_C0(self, params):
        no_datasets = len(self.traces)
        n_c0s = self.no_species
        
        C0 = []
        for d in range(1,no_datasets+1):
            c0_keys = ['c0{i}{d}'.format(i=i,d=d) for i in range(1,n_c0s+1)]
            dataset_c0 = np.array([params[key].value for key in c0_keys])
            C0.append(dataset_c0)
            
        return C0
        
    def _get_T0(self, params):
        no_datasets = len(self.traces)
        
        T0_keys = ['t0{d}'.format(d=d) for d in range(1,no_datasets+1)]
        T0 = np.array([params[key].value for key in T0_keys])
        return T0
        
    def _get_OD_offset(self, params):        
        no_datasets = len(self.traces)
        
        OD_keys = ['OD_offset{d}'.format(d=d) for d in range(1,no_datasets+1)]
        OD = np.array([params[key].value for key in OD_keys])
        return OD
        
   # define a function that will measure the error between the fit and the real data:
    def errfunc(self, params):
        """
        Master error function
        
        Computes residuals for a given rate function, rate constants and initial concentrations
        by linearly fitting the integrated concentrations to the provided spectra.
        
        As we wish to simultaneously fit multiple data sets T and S contain multiple
        arrays of times and spectral traces respectively.
        
        params an lmfit Parameters object representing the rate constants, 
        initial concentrations,initial times, and OD offsets.
        
        implit dependence on:
        dc_dt - function to be integrated by odeint to get the concentrations
        
        self.times - the array over which integration occurs
                     since we have several data sets and each one has its own 
                     array of time points, self.times is an array of arrays.
        self.traces - the spectral data, it is an array of array of arrays
        """

        K = self._get_K(params)
        T0 = self._get_T0(params)
        C0 = self._get_C0(params)        
        OD_offset = self._get_OD_offset(params)
        
        offset_times = [t-t0 for t,t0 in zip(self.times,T0)]
        offset_traces = [st - od for st,od in zip(self.traces,OD_offset)]

        # calculated concentrations for the different time sets
        fitted_conc_traces = []   
        for t ,k, c0 in zip(offset_times,K,C0):
            conc_traces = self.C(t,k,c0)

            if np.isnan(conc_traces).any():
                fix = Imputer(missing_values='NaN', strategy='median',axis=0) 
                conc_traces  = fix.fit_transform(conc_traces )
                warnings.warn('Nan found in predicted concentrations')

            fitted_conc_traces.append(conc_traces)
        
        # spectra fitted against all data sets
        # REQUIRES spectral traces to be measured at the SAME WAVELENGTHS!
        
        fitted_spectra = self.get_spectra(np.vstack(fitted_conc_traces),
                                          np.vstack(offset_traces))
                                                
        fitted_spectral_traces = [self.get_traces(c, fitted_spectra) for c in
                                        fitted_conc_traces]
            
        self.residuals = [fst -t for fst,t in zip(fitted_spectral_traces,
                                                  offset_traces)]
            
        all_residuals = np.vstack(self.residuals).ravel()
        
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
        
        self.expand_params()

        if debug:
            self.output = minimize(self.errfunc, self.input_params, 
                                   method=self.method,iter_cb=self.printfunc)
        else:
            self.output = minimize(self.errfunc, self.input_params,
                                   method=self.method)

        fitted_params = self.output.params
      
        fitted_K = self._get_K(fitted_params)
        fitted_T0 = self._get_T0(fitted_params)
        fitted_OD_offset = self._get_OD_offset(fitted_params)
        fitted_C0 = self._get_C0(fitted_params)
        
        offset_traces = [traces - od for traces,od in zip(self.traces,
                                                          fitted_OD_offset)]
                                                        
        offset_times = [times - t0 for times,t0 in zip(self.times, 
                                                       fitted_T0)]
                                                       
        fitted_C = [self.C(t, fitted_k, c0) for t,fitted_k,c0 in zip(offset_times,
                                                                  fitted_K,
                                                                  fitted_C0)]
       
        fitted_spectra = self.get_spectra(np.vstack(fitted_C),
                                          np.vstack(offset_traces))
                                          
        
        fitted_traces = [self.get_traces(c, fitted_spectra) for c in fitted_C]
         
        self.fitted_spectra = fitted_spectra
        self.fitted_traces = fitted_traces
        self.fitted_ks = fitted_K
        self.fitted_t0 = fitted_T0
        self.fitted_c0 = fitted_C0        
        self.fitted_OD_offset = fitted_OD_offset
        self.fitted_C = fitted_C  
        
        # create master resampled data
        if self.no_resampled_points:
            no_points = self.no_resampled_points
        else:
            no_points = max([len(t) for t in offset_times])*5

        max_time = max(np.hstack(offset_times))
        min_time = min(np.hstack(offset_times))
        
        if min_time > 0:
            min_time = 0
        
        resampled_times = np.linspace(min_time, max_time, no_points)
        
        self.resampled_C = [self.C(resampled_times,k,c0) for k,c0 in zip(fitted_K,
                                                                         fitted_C0)]
        self.resampled_traces = [self.get_traces(c,self.fitted_spectra) for c in
                                        self.resampled_C]
          
        self.resampled_times = [resampled_times + t0 for t0 in fitted_T0]
     
    def expand_params(self):
        """
        If only a single set of parameters has been provided then we expand 
        the parameters by constructing a set for each dataset
        """
        
        no_datasets = len(self.traces)
        no_species = self.reaction_matrix.shape[0]
        
        t0_keys = [key for key in self.input_params.keys() if 't0' in key]
        od_keys = [key for key in self.input_params.keys() if 'OD' in key]
        k_keys = [key for key in self.input_params.keys() if 'k' in key]
        c0_keys = [key for key in self.input_params.keys() if 'c0' in key]
       
        enum_keys = list(enumerate(self.input_params.keys()))
        first_t0 = next(i for i,key in enum_keys if 't0' in key)
        first_od = next(i for i,key in  enum_keys if 'OD' in key)
        first_k = next(i for i,key in enum_keys if 'k' in key)
        first_c0 = next(i for i,key in enum_keys if 'c0' in key)
        
        t0_params = [self.input_params.pop(k) for k in t0_keys]
        od_params = [self.input_params.pop(k) for k in od_keys]
        k_params = [self.input_params.pop(k) for k in k_keys]
        c0_params = [self.input_params.pop(k) for k in c0_keys]
        
        if len(t0_keys) == 1 and t0_keys[0] == 't0':            
            p = t0_params[0]
            new_t0_params = []            
            for d in range(1,no_datasets+1):
                new_p = copy.deepcopy(p)
                new_p.name += str(d)
                new_t0_params.append(new_p)
            t0_params = new_t0_params
            
        if len(od_keys) == 1 and od_keys[0] == 'OD_offset':             
            p = od_params[0]
            new_od_params = []
            for d in range(1,no_datasets+1):
                new_p = copy.deepcopy(p)
                new_p.name += str(d)
                new_od_params.append(new_p)
            od_params = new_od_params
            
        # TODO - this is not adequate - what if the first rate parameter 
        # isn't k1?
        if len(k_keys) == self.reaction_matrix.sum() and k_keys[0] == 'k1':
            new_k_params = []
            for p in k_params:
                for d in range(1,no_datasets+1):
                    new_p = copy.deepcopy(p)                    
                    new_p.name += str(d)
                    new_k_params.append(new_p)
            k_params = new_k_params
            
        if len(c0_keys) == no_species and c0_keys[0] == 'c01':
            new_c0_params = []
            for p in c0_params:
                for d in range(1,no_datasets+1):
                    new_p = copy.deepcopy(p)
                    new_p.name += str(d)
                    new_c0_params.append(new_p)
            c0_params = new_c0_params
            
        # as lmfit parameters objects are ordered dictionaries the order
        # that we do this actually matters and will influence the fitting
        # we would like to allow the used to specify the order and respect the 
        # order they choose.
        
        # NB The ideal order is to have the parameters whos initial values are 
        # better optimised after the parameters whos initial values are worse       
           
        expanded_params = sorted([(t0_params,first_t0),
                                  (od_params,first_od),
                                  (k_params,first_k),
                                  (c0_params,first_c0)], key=lambda e:e[1])
        expanded_params, loc = zip(*expanded_params)
                       
        for ep in expanded_params:
            self.input_params.add_many(*ep)
    
    # TODO order is not yet ideal - would like explicitly given parameters
    # to be optimised last        
    def init_sequential(self, no_species):
        """Initialises parameters for a sequential fit"""        
        
        if not self.no_species is None and self.no_species != no_species:
            raise UserWarning('Inconsistent number of species')
        
        if not self.reaction_matrix is None:
            raise UserWarning('Reaction matrix already specified')

        self.reaction_matrix = np.zeros([no_species, no_species])
        self.no_species = no_species
        
        no_datasets = len(self.traces)
        
        for i in range(no_species-1):       
            self.reaction_matrix[i,i+1] = 1
        
        if self.input_params is None:
            self.input_params = Parameters()
        
        # if no rate constants set, assign n-1 rate constants to a default
        # of 0.1 for each dataset
        if not any(['k' in key for key in self.input_params.valuesdict()]):
            rate_constants = [('k{i}{d}'.format(i=n,d=d),0.1,True,0,None,None) 
                               for n in range(1,no_species)
                               for d in range(1,no_datasets+1)]                  
            self.input_params.add_many(*rate_constants)
          
        # if no t0s assign t0 to a default of 0 and flag them not to be
        # optimised for each dataset
        if not any(['t0' in key for key in self.input_params.valuesdict()]):
            t0 = [('t0{d}'.format(d=d),0,False,None,None,None)
                  for d in range(1,no_datasets+1)]
            self.input_params.add_many(*t0)
        
        # if no OD_offsets assign OD_offset to a default of 0 and flag them
        # not to be optimised for each dataset          
        if not any(['OD' in key for key in self.input_params.valuesdict()]):
            OD_offset = [('OD_offset{d}'.format(d=d),0,False,None,None,None)
                         for d in range(1,no_datasets+1)]
            self.input_params.add_many(*OD_offset)
 
       # if no c0s assign c0 to a default of [1,0,0,...] and flag them
        # not to be optimised  for each dataset        
        if not any(['c0' in key for key in self.input_params.valuesdict()]):
            C0 = [('c0{i}{d}'.format(i=n,d=d),0,False,0,1,None)
                         for n in range(1,no_species+1)
                         for d in range(1,no_datasets+1)]
        
            self.input_params.add_many(*C0)
            
            for d in range(1,no_datasets+1):
                self.input_params['c01{d}'.format(d=d)].value = 1
                
    def fit_sequential(self, no_species, debug=False):
        """
        Utility function to fit assuming a sequential reaction model
        
        Sets the reaction matrix up for a sequential model then calls the 
        master fit() method
        """
 
        self.init_sequential(no_species)
        self.fit(debug)
            
    # TODO order is not yet ideal - would like explicitly given parameters
    # to be optimised last
    def init_parallel(self, no_species):
        """Initialises parameters for a parallel fit"""
        if not self.no_species is None and self.no_species != no_species:
            raise UserWarning('Inconsistent number of species')
         
        if not self.reaction_matrix is None:
            raise UserWarning('Reaction matrix already specified')

        self.reaction_matrix = np.zeros([no_species, no_species])
        self.no_species = no_species
                
        no_datasets = len(self.traces)
        
        for i in range(0,no_species-1,2):
            self.reaction_matrix[i,i+1] = 1
        
        
        if self.input_params is None:
            self.input_params = Parameters()
        
        # if no rate constants set, assign n-1 rate constants to a default
        # of 0.1 for each dataset
        if not any(['k' in key for key in self.input_params.valuesdict()]):
            rate_constants = [('k{i}{d}'.format(i=n,d=d),0.1,True,0,None,None) 
                              for n in range(1,no_species,2)
                              for d in range(1,no_datasets+1)]
            self.input_params.add_many(*rate_constants)

        # if no t0s assign n t0s to a default of 0 and flat them not to be
        # optimised
        if not  any(['t0' in key for key in self.input_params.valuesdict()]):
            t0 = [('t0{i}'.format(i=n),0,False,None,None,None)
                  for n in range(1,no_datasets+1)]
            self.input_params.add_many(*t0)
        
        # if no OD_offsets assign OD_offset to a default of 0 and flag them
        # not to be optimised for each dataset
        if not  any(['OD' in key for key in self.input_params.valuesdict()]):
            OD_offset = [('OD_offset{i}'.format(i=n),0,False,None,None,None)
                         for n in range(1,no_datasets+1)]
            self.input_para,s.add_many(*OD_offset)

        # if no c0s assign c0 to a default of 1 and flag them
        # not to be optimised for each dataset
       # if no c0s assign c0 to a default of [1,0,0,...] and flag them
        # not to be optimised  for each dataset        
        if not any(['c0' in key for key in self.input_params.valuesdict()]):
            C0 = [('c0{i}{d}'.format(i=n,d=d),0,False,0,1,None)
                         for n in range(1,no_species+1)
                         for d in range(1,no_datasets+1)]
        
            self.input_params.add_many(*C0)
            
            for n in range(1,no_species,2):
                for d in range(1,no_datasets+1):
                    self.input_params['c0{i}{d}'.format(i=n,d=d)].value = 1
                    
    def fit_parallel(self, no_species,debug=False):
        """
        Utility function to fit assuming a parallel reaction model
        
        Sets the reaction matrix up for a parallel model then calls the 
        master fit() method
        """
        
        self.init_parallel(no_species)
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
        
        latex_eqn = r'$' + ','.join(eqn) + r'$'
        return latex_eqn        
        
if __name__ == '__main__':
    
    from lmfit import fit_report
    import matplotlib.pyplot as plt
    from plot_utils import *
    
    test_spectra = np.loadtxt('test_spectra.csv',delimiter=',')
    test_times = [np.loadtxt('test_time.csv',delimiter=',')+0,
                  np.loadtxt('test_time.csv',delimiter=',')+1]
    test_traces = [np.loadtxt('test_trace.csv',delimiter=','),
                   np.loadtxt('test_trace.csv',delimiter=',')]
    test_wavelengths = np.loadtxt('test_wavelengths.csv',delimiter=',')

#    reaction_matrix = [[0, 1, 0],
#                       [0, 0, 1],
#                       [0, 0, 0]]
#    
    input_params = Parameters()
    #                     (Name,        Value,  Vary,  Min,   Max,  Expr)
    input_params.add_many(('k1',        0.1,    True,  0,     None, None),
                          ('k2',        0.1,    True,  0,     None, None),
                          ('t01',         0,     False, None,  None, None),
                          ('t02',        1.,     False, None,  None, None),
                          ('OD_offset1', 0,      False, None,  None, None),
                          ('OD_offset2', 0,      False, None,  None, None),
                          ('c01',        1,      False, None,  None, None),
                          ('c02',        0,      False, None,  None, None),
                          ('c03',        0,      False, None,  None, None))

    tb = UltraFast_TB(test_times, test_traces, test_wavelengths, alpha=1e-5,
                      input_params=input_params) #, reaction_matrix)
    tb.apply_svd(n=3)    
    tb.fit_sequential(3,debug=True)
    
    fitted_params = tb.output.params
    for k in fitted_params.values():
        k.vary=False
    
    tb = UltraFast_TB(test_times, test_traces, test_wavelengths, 
                      input_params=fitted_params)#, reaction_matrix)
    tb.fit_sequential(3)

    #plot_spectra(tb)
    #plot_traces(tb)
    #plot_concentrations(tb)
    
    plot_master(tb,trace_wavelengths=(125,150,175,230,275),ind=0)
    
    print('Residuals: ', np.abs(tb.residuals).sum())
    print(fit_report(fitted_params))
    
    plt.plot(test_spectra)
    plt.show()
