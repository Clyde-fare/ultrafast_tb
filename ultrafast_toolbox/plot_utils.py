# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 19:06:29 2016

@author: clyde
"""

import numpy as np
from lmfit import fit_report
from matplotlib import pyplot as plt
from matplotlib import gridspec
from itertools import cycle


def plot_svd(uf_tb, n,log=True):
    """Utility plotting function"""
    
    if log:
        plot_f = plt.semilogx
    else:
        plot_f = plt.plot
        
    for times,traces in zip(uf_tb.times,uf_tb.traces):
        U,s,V = np.linalg.svd(traces, full_matrices=True)
        
        plt.figure(figsize=(12,8))

        for i in range(n):
            plt.subplot(2,n,1+i)
            plot_f(times, U[:,i])
        for j in range(n,2*n):
            plt.subplot(2,n,j+1)
            plot_f(uf_tb.wavelengths, V[j,:])
        plt.show()

def plot_all_traces(uf_tb, ind=None, semilog=False):
    """Utility plotting function"""
    
    if uf_tb.wavelengths is None:
        wavelengths_to_plot = range(uf_tb.fitted_spectra[0].shape[0])
    else:
        wavelengths_to_plot = uf_tb.wavelengths        
    if ind is None:
        fitted_traces_to_plot = uf_tb.fitted_traces
        orig_traces_to_plot = uf_tb.traces
        times_to_plot = uf_tb.times
    else:
        fitted_traces_to_plot = uf_tb.fitted_traces[ind:ind+1]
        orig_traces_to_plot = uf_tb.traces[ind:ind+1]
        times_to_plot = uf_tb.times[ind:ind+1]
     
    get_plot_f = lambda ax: ax.semilog if semilog else ax.plot
        

    for orig_traces,fit_traces,times in zip(orig_traces_to_plot,
                                                     fitted_traces_to_plot,
                                                     times_to_plot):
                   
        n_traces = len(wavelengths_to_plot)
        n_columns=5
        n_rows = int(np.ceil(n_traces/n_columns))
        plt.figure(figsize=(n_columns*3,n_rows*3))
        grid=gridspec.GridSpec(n_rows,n_columns,wspace=0.0, hspace=0.0)
        
        for i,wavelength_traces in enumerate(zip(orig_traces.T,
                                                 fit_traces.T)):
            wavelength = '{:.2f}'.format(wavelengths_to_plot[i])
            
            ax = plt.subplot(grid[i//n_columns,i%n_columns])
            plot_f = get_plot_f(ax)
            plot_f(times,wavelength_traces[0],marker='o',linestyle='')
            plot_f(times,wavelength_traces[1],label=wavelength)
            ax.legend()
            
        plt.show()

def plot_traces(uf_tb, trace_wavelengths=None, ind=0, ax=None):
    """Utility plotting function"""

    if not ax:   
        fig=plt.figure()
        ax = fig.gca()

    trace_colors = cycle(['teal','orange','fuchsia','cornflowerblue','darkviolet','black','cyan'])
    if trace_wavelengths is None:
        trace_wavelengths = []       
    trace_inds = [np.abs(uf_tb.wavelengths-w).argmin() for w in trace_wavelengths]
    
    for wvl,trace_ind in zip(trace_wavelengths, trace_inds):
        c=trace_colors.next()
        ax.plot(uf_tb.resampled_times[ind],
                 uf_tb.resampled_traces[ind].T[trace_ind], marker='',
                 linestyle='-',label=wvl,c=c)
        ax.plot(uf_tb.times[ind],uf_tb.traces[ind].T[trace_ind],marker='o',
                 linestyle='',c=c)
    return ax
    
def plot_spectra(uf_tb, ind=0):
    """Utility plotting function"""
    
    # hopefully we need less than 26 species!
    species = iter('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    if uf_tb.wavelengths is None:
        uf_tb.wavelengths = [range(s.shape[0]) for s in uf_tb.fitted_spectra]

    if ind is None:
        spectra_to_plot = uf_tb.fitted_spectra
        wavelengths_to_plot = uf_tb.wavelengths
    else:
        spectra_to_plot = uf_tb.fitted_spectra[ind:ind+1]
        wavelengths_to_plot = uf_tb.wavelengths[ind:ind+1]
    
    for wlengths, spectra in zip(wavelengths_to_plot, spectra_to_plot):
        
        plt.figure()
        for spectrum in spectra.T:           
            plt.plot(wlengths, spectrum,label=species.next())
        plt.legend()
        plt.show()
    
def plot_concentrations(uf_tb):
    """Utility plotting function"""
    
    # hopefully we need less than 26 species!
    species = iter('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    for c in uf_tb.resampled_C.T:
        plt.plot(uf_tb.resampled_times, c, label=species.next())
        
    plt.legend()
    plt.show()

def plot_master(uf_tb, trace_wavelengths=None, inc_details=False, ind=0):
    
    if trace_wavelengths is None:
        trace_wavelengths = []
       
    trace_colors = cycle(['teal','orange','fuchsia','cornflowerblue','darkviolet','black','cyan'])
    
    #    time_offset = uf_tb.extras['time_offset']
    #    u_times = uf_tb.times[0] - time_offset
    #    u_times = u_times[u_times>0]
    #    u_times += time_offset
    
    trace_inds = [np.abs(uf_tb.wavelengths-w).argmin() for w in trace_wavelengths]
    
    #labels=['{:.2f}'.format(k) for k in uf_tb.fitted_ks.valuesdict().values()]
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:uf_tb.no_species]
    
    plt.figure(figsize=(12,12))
    plt.subplot(2,3,1)
    plt.plot(uf_tb.times[ind],np.abs(uf_tb.residuals[ind]).sum(axis=1))
    plt.xlabel('Time /s')
    plt.ylabel('Residuals')
    plt.subplot(2,3,2)
    plt.plot(uf_tb.wavelengths,np.abs(uf_tb.residuals[ind]).sum(axis=0))
    plt.xlabel('Wavelengths')
    plt.ylabel('Residuals')
    plt.subplot(2,3,3)
    
    for sp,l in zip(uf_tb.fitted_spectra.T,labels):
        plt.plot(uf_tb.wavelengths,sp,label=l)
    plt.xlabel('Wavelengths')
    plt.ylabel('Absorbance')
    plt.legend()
    plt.subplot(2,3,4)
    for wvl,trace_ind in zip(trace_wavelengths, trace_inds):
        c=trace_colors.next()
        plt.plot(uf_tb.resampled_times[ind],
                 uf_tb.resampled_traces[ind].T[trace_ind], marker='',
                 linestyle='-',label=wvl,c=c)
        plt.plot(uf_tb.times[ind],uf_tb.traces[ind].T[trace_ind],marker='o',
                 linestyle='',c=c)

    plt.axvline(uf_tb.fitted_t0[ind])
    plt.xlabel('Time /s')
    plt.ylabel('Absorbance')
    plt.legend()
    plt.subplot(2,3,5)
    plt.plot(uf_tb.wavelengths,uf_tb.traces[ind].T)
    plt.subplot(2,3,6)
    for c, l in zip(uf_tb.resampled_C[ind].T, labels):
        plt.plot(uf_tb.resampled_times[ind],c,label=l)
    
    plt.xlabel('Time /s')
    plt.ylabel('Concentration')
    plt.legend()
    plt.show()
    
    if inc_details:
        print(fit_report(uf_tb.output))
