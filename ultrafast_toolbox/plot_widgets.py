from ipywidgets import fixed, interactive
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt

def raw_plot_spec(data, times, i=0, sigma=0, w1=405,w2=506):
    wavelengths = data[:,0]
    traces = data[:,1:]
    
    proc_traces = gaussian_filter(traces[:,i],sigma)
    plt.figure(figsize=(10,10))
    plt.plot(wavelengths,proc_traces)
    plt.title('UV_Vis spectra time {t}s'.format(t=times[i]))
    plt.xlabel('Wavelength /nm')
    plt.ylabel('Absorbance')
    plt.ylim(-0.2,1.4)
        
    plt.axvline(w1, linestyle='--')
    plt.axvline(w2, linestyle='--')

    print('step=',i,'time=',times[i])

def plot_spec(data, times): 
    return interactive(raw_plot_spec, data=fixed(data), times=fixed(times), i=[0,100,1],sigma=[0,50,0.5],w1=[0,1200,1],w2=[0,1200,1])
