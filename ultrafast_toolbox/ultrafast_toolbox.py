# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:44:11 2016

@author: clyde
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import optimize
from scipy.integrate import odeint

# define a function that will measure the error between the fit and the real data:
def errfunc(K,rate_function,c0,T,S):
    """
    Master error function
    
    Computes residuals for a given rate function, rate constants and initial concentrations
    by linearly fitting the integrated concentrations to the provided spectra.
    
    As we wish to simultaneously fit multiple data sets T and S contain multiple
    arrays of times and spectral traces respectively.
    
    K is an array of rate constants for the rate equation
    rate_function is the function to be integrated by odeint
    c0 is an array of initial conditions, so that c(t0) = c0
    T is the array over which integration occurs, since we have several data sets and each 
    one has its own array of time points, T is an array of arrays.
    S is the spectral data, it is an array of arrays
    """
    res = [] 

    # compute the residuals for each array of times/spectral traces
    for times,spectral_trace in zip(T,S):
        fitted_conc_traces = odeint(rate, c0, times, args=(K,))
        fitted_spectral_traces = get_traces(fitted_conc_traces, spectral_trace)
        
        #residuals for the time traces of each of the species
        res += [fitted_spectral_traces[:,i] - spectral_trace[:,i] for i in range(spectral_trace.shape[1])]

    all_residuals = np.hstack(res).ravel()
    return all_residuals


def get_spectra(conc_traces,spectral_trace):
    """Extraction of predicted spectra given concentration traces and spectral_traces"""
    # linear fit of the fitted_concs to the spectra CANNOT fit intercept here!
    l=LinearRegression(fit_intercept=False)
    l.fit(conc_traces,spectral_trace)
    fitted_spectra = l.coef_
    return fitted_spectra


def get_traces(conc_traces, spectral_trace):
    """Extraction of fitted spectral traces given concentration traces and spectral traces"""
    # linear fit of the fitted_concs to the spectra CANNOT fit intercept here!
    l=LinearRegression(fit_intercept=False)
    l.fit(conc_traces,spectral_trace)
    fitted_spectral_traces = l.predict(conc_traces)
    return fitted_spectral_traces


def plot_traces(fitted_traces,time, wavelengths=None):
    """Utility plotting function"""
    for i,wavelength_trace in enumerate(fitted_traces.T):
        if wavelengths is not None:
            wavelength = '{:.2f}'.format(wavelengths[i])
        else:
            wavelength = ''
        
        plt.figure(figsize=(3,3))
        plt.plot(time,wavelength_trace,label=wavelength)
        plt.legend()
    plt.show()
   
   
def get_rate_function(reaction_matrix):
    """
    Construction of a rate function (returning dc/dt) for a given reaction matrix.
    
    Columns of the reaction matrix correspond reactant species
    Rows of the reaction correspond to product species
    
    e.g. reaction_matrix = [[0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0]]
          
    Corresponds to the reaction scheme A->B->C.
    
    The generated rate function has three arguments:
        C an array of floats giving the concentration of each species
        t a float giving the current time (not used but necessary for ODEs)
        K an array of floats giving the rate constants.
    
    And returns:
        dc/dt an array floats corresponding to the derivative of the concentration
        of each species with time at time=t
        
    The above example reaction matrix would give rise to dc/dt = [-k1[A], k1[A]-k2[B], k2[B]]
    """
    
    # dc/dt built up by separately computing the positive and negative contributions.
    # In our example positive_dcdt = [0, k1[A], k2[B]]  and negative_dcdt = [-k1[A],-k2[B],0]
 
    reaction_matrix = np.array(reaction_matrix,dtype=np.int)

    # no_species = reaction_matrix.shape[1]
    # no_ks = reaction_matrix.ravel().sum()
    
    def rate(C,t,K):
        C = np.array(C)
        K = np.array(K)
        
        # need to be careful about dtypes here:
        # reaction matrix dtype is int, rate matrix must be dtype float
        rate_matrix = reaction_matrix.copy()
        rate_matrix.dtype=np.float64
        rate_matrix[reaction_matrix==1] = K
        
        positive_dcdt = rate_matrix.dot(C)
        negative_dcdt = rate_matrix.sum(axis=0)*C
            
        return tuple(positive_dcdt - negative_dcdt)
    
    return rate


def fit_kinetics(times,traces,wavelengths, rate_function, k_guess, svd_comps=0, c0=None):
    """Master fitting function"""
    if c0 is None:
        # if init concentrations not specified assume first component starts at concentration 1 and 
        # all other start a concentration 0
        c0 = np.zeros(len(k_guess)+1)
        c0[0] = 1
    
    if svd_comps:
        svd_trace, s, V = np.linalg.svd(traces, full_matrices=True)
        fitted_k,success = optimize.leastsq(errfunc, k_guess, args=(rate_function,c0,[times],[svd_trace[:,:svd_comps]])) 
    else:
        fitted_k,success = optimize.leastsq(errfunc, k_guess, args=(rate_function,c0,[times],[traces])) 

    fitted_concs = odeint(rate_function, c0, times, args=(fitted_k,))
    
    return fitted_k, fitted_concs


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    reaction_matrix = [[0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0]]
    
    rate = get_rate_function(reaction_matrix)
    
    test_spectra = np.loadtxt('test_spectra.csv',delimiter=',')
    test_time = np.loadtxt('test_time.csv',delimiter=',')
    test_traces = np.loadtxt('test_trace.csv',delimiter=',')
    test_wavelengths = np.loadtxt('test_wavelengths.csv',delimiter=',')
    
    c0 = np.array([1,0,0])
    k_guess = np.array([0.1,0.1],dtype=np.float64)
    
    fitted_k,fitted_concs = fit_kinetics(test_time, test_traces, test_wavelengths, rate, 
                                         k_guess=k_guess, svd_comps=0)
    fitted_traces = get_traces(fitted_concs, test_traces)
    fitted_spectra = get_spectra(fitted_concs, test_traces)
    
    print(fitted_k)
    plt.subplot(211)
    plt.plot(fitted_spectra)
    plt.subplot(212)
    plt.plot(test_spectra)
    plt.show()