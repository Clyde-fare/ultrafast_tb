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
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator

sns.set_style("white")
sns.set_style("ticks")

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
            plot_f(times, U[:,i],label='{:.5f}'.format(s[i]))
            ax = plt.gca() 
            ax.yaxis.tick_left()
            ax.xaxis.tick_bottom()
            plt.legend()

        for j in range(n,2*n):
            plt.subplot(2,n,j+1)
            plot_f(uf_tb.wavelengths, V[j,:])
            ax = plt.gca()
            ax.yaxis.tick_left()
            ax.xaxis.tick_bottom()
   
        plt.tight_layout()
        plt.show()

def plot_all_traces_v2(uf_tb, ind=None, semilog=False, xticks=None, xlim=None, save='', invert_wavs=False):
    sns.set(style="ticks")
    
    if uf_tb.wavelengths is None:
        wavelengths_to_plot = range(uf_tb.fitted_spectra[0].shape[0])
    else:
        wavelengths_to_plot = list(uf_tb.wavelengths)
    
    #this is deliberate but it's a weird way of doing it
    if not invert_wavs:
        wavelengths_to_plot.reverse()
        
    no_wavelengths = len(wavelengths_to_plot)
    
    if ind is None:
        orig_traces_to_plot = uf_tb.traces[0]
        times_to_plot = list(uf_tb.times[0])
        
        fitted_traces_to_plot = uf_tb.resampled_traces[0].T
        resampled_times_to_plot = uf_tb.resampled_times[0]
    else:
        orig_traces_to_plot = uf_tb.traces[ind]
        times_to_plot = list(uf_tb.times[ind])
        
        fitted_traces_to_plot = uf_tb.resampled_traces[ind].T
        resampled_times_to_plot = uf_tb.resampled_times[ind]
    
    fitted_traces_to_plot = np.flipud(fitted_traces_to_plot)
    
    no_points = orig_traces_to_plot.shape[0]
    
    wavelength = np.array(wavelengths_to_plot*no_points)
    wavelength = wavelength.reshape(no_points,no_wavelengths).T
    
    time = np.array(times_to_plot * no_wavelengths)
    time = time.reshape(no_points,no_wavelengths)
    
    traces = orig_traces_to_plot.T

    df = pd.DataFrame(np.c_[traces.ravel(),
                            time.ravel(), 
                            wavelength.ravel()],
                      columns=["absorption", "time", "wavelength"]).round({'wavelength': 1})
    
    # # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(df, col="wavelength", hue="wavelength", col_wrap=5, size=3,sharey=False)
    
    # # Draw a horizontal line to show the starting point
    grid.map(plt.axhline, y=0, ls=":", c=".5")
    
    # # Draw points showing the absorption vs. time for each wavelength
    grid.map(plt.plot, "time", "absorption", marker="o", linestyle='', ms=6)
    
    # # Adjust the tick positions and labels
    if xticks:
        grid.set(xticks=xticks)
    if xlim:
        grid.set(xlim=xlim)
    
    #for each plot adjust the ylimits and yticks and add the fitted trace
    for i,ax in enumerate(grid.axes):
        l = ax.get_lines()[1]
        c = plt.getp(l, 'color')
        ax.plot(resampled_times_to_plot, fitted_traces_to_plot[i], 
                linestyle='--', color=c, label='fit')
        
        ymin, ymax = ax.get_ylim()
        y_range = ymax-ymin
        
        mod_ymax = ymin + y_range*1.1
        mod_ymin = ymax - y_range*1.1
        
        ax.set_ylim(mod_ymin,mod_ymax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='lower'))
    
    # # Adjust the arrangement of the plots
    #grid.fig.tight_layout(w_pad=1)
    if save:
        grid.savefig(save)

def plot_all_traces(uf_tb, ind=None, semilog=False):
    """Utility plotting function"""
    
    if uf_tb.wavelengths is None:
        wavelengths_to_plot = range(uf_tb.fitted_spectra[0].shape[0])
    else:
        wavelengths_to_plot = uf_tb.wavelengths        
    if ind is None:
        fitted_traces_to_plot = uf_tb.resampled_traces
        orig_traces_to_plot = uf_tb.traces
        times_to_plot = uf_tb.times
        resampled_times_to_plot = uf_tb.resampled_times
    else:
        fitted_traces_to_plot = uf_tb.resampled_traces[ind:ind+1]
        orig_traces_to_plot = uf_tb.traces[ind:ind+1]
        times_to_plot = uf_tb.times[ind:ind+1]
        resampled_times_to_plot = uf_tb.resampled_times[ind:ind+1]
     
    get_plot_f = lambda ax: ax.semilog if semilog else ax.plot
        

    for orig_traces,fit_traces,times,rtimes in zip(orig_traces_to_plot,
                                                   fitted_traces_to_plot,
                                                   times_to_plot,
                                                   resampled_times_to_plot):
                   
        n_traces = len(wavelengths_to_plot)
        n_columns=5
        n_rows = int(np.ceil(n_traces/float(n_columns)))
        plt.figure(figsize=(n_columns*3,n_rows*3))
        grid=gridspec.GridSpec(n_rows,n_columns,wspace=0.0, hspace=0.0)
        
        for i,wavelength_traces in enumerate(zip(orig_traces.T,
                                                  fit_traces.T)):
            wavelength = '{:.2f}'.format(wavelengths_to_plot[i])
            
            ax = plt.subplot(grid[i//n_columns,i%n_columns])
            plot_f = get_plot_f(ax)
            plot_f(times,wavelength_traces[0],marker='o',linestyle='')
            plot_f(rtimes,wavelength_traces[1],label=wavelength)
            ax.legend()
            ax.yaxis.tick_left()
            ax.xaxis.tick_bottom()
           
        plt.tight_layout()
        plt.show()

def plot_traces(uf_tb, trace_wavelengths=None, linestyle='-',xlim=None,ind=0, ax=None):
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
                 linestyle=linestyle,label=wvl,c=c)
        ax.plot(uf_tb.times[ind],uf_tb.traces[ind].T[trace_ind],marker='o',
                 linestyle='',c=c)
    if xlim:
        ax.set_xlim(xlim)
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

def plot_master(uf_tb, trace_wavelengths=None, norm_spec=None, inc_details=False, ind=0, invert_xaxis=False):
    
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
    ax1=plt.subplot(2,3,1)
    ax1.yaxis.tick_left()
    ax1.xaxis.tick_bottom()
    ax1.plot(uf_tb.times[ind],np.abs(uf_tb.residuals[ind]).sum(axis=1))
    ax1.set_xlabel('Time /s')
    ax1.set_ylabel('Residuals')
    ax2=plt.subplot(2,3,2)
    ax2.plot(uf_tb.wavelengths,np.abs(uf_tb.residuals[ind]).sum(axis=0))
    ax2.set_xlabel('Wavelengths')
    ax2.set_ylabel('Residuals')
    ax2.yaxis.tick_left()
    ax2.xaxis.tick_bottom()
    
    ax3=plt.subplot(2,3,3)
    
    if norm_spec is None:
        mults = [1 for _ in range(uf_tb.fitted_spectra.shape[1])]
    else:
        maxima = np.max(uf_tb.fitted_spectra,axis=0)[norm_spec:]
        minima = np.min(uf_tb.fitted_spectra,axis=0)[norm_spec:]
        
        s0_max = maxima[0]
        s0_min = minima[0]

        unity_spec = uf_tb.fitted_spectra.shape[1] - (len(maxima)-1)
        
        mults = [np.max([s0_max/s_max, s0_min,s_min]) for s_max,s_min in zip(maxima[1:],
                                                                             minima[1:])]    
        mults = [1 for _ in range(unity_spec)] + mults

    for m,sp,l in zip(mults,uf_tb.fitted_spectra.T,labels):
        ax3.plot(uf_tb.wavelengths,m*sp,label=l)

    ax3.set_xlabel('Wavelengths')
    ax3.set_ylabel('Absorbance')
    ax3.legend()
    ax3.yaxis.tick_left()
    ax3.xaxis.tick_bottom()
    ax4=plt.subplot(2,3,4)
    for wvl,trace_ind in zip(trace_wavelengths, trace_inds):
        c=trace_colors.next()
        ax4.plot(uf_tb.resampled_times[ind],
                 uf_tb.resampled_traces[ind].T[trace_ind], marker='',
                 linestyle='-',label=wvl,c=c)
        ax4.plot(uf_tb.times[ind],uf_tb.traces[ind].T[trace_ind],marker='o',
                 linestyle='',c=c)

    ax4.axvline(uf_tb.fitted_t0[ind])
    ax4.set_xlabel('Time /s')
    ax4.set_ylabel('Absorbance')
    ax4.legend()
    ax4.yaxis.tick_left()
    ax4.xaxis.tick_bottom()
    ax5=plt.subplot(2,3,5)
    ax5.plot(uf_tb.wavelengths,uf_tb.traces[ind].T)
    ax5.yaxis.tick_left()
    ax5.xaxis.tick_bottom()
    ax5.set_xlabel('Wavelengths')
    ax5.set_ylabel('Absorbance')
    ax6=plt.subplot(2,3,6)
    for c, l in zip(uf_tb.resampled_C[ind].T, labels):
        ax6.plot(uf_tb.resampled_times[ind],c,label=l)
    
    ax6.set_xlabel('Time /s')
    ax6.set_ylabel('Concentration')
    ax6.legend()
    ax6.yaxis.tick_left()
    ax6.xaxis.tick_bottom()

    if invert_xaxis:
        ax2.invert_xaxis()
        ax3.invert_xaxis()
        ax5.invert_xaxis()

    plt.tight_layout()
    plt.show()
    
    if inc_details:
        print(fit_report(uf_tb.output))
 
