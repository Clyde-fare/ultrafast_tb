from scipy.optimize import curve_fit

def dX(X,t,k):
    return k.dot(X)

def X(t,k,X0):
    """
    Solution to the ODE X'(t) = f(X,k,A0) with initial condition A(0) = A0
    """
    y = odeint(dX, X0, t, args=(k,))
    return y

def K(ks, connectivity):
    """
    Converts a matrix of connectivities defining the reactions and a list of rate constants 
    to the k matrix.
    
    Connectivity matrix:
           rows represent reactants
           columns represent products
    
    k matrix:
            rows represent rate of change of species
            columns represent concentrations of species 
            
                        k1   k2
    Example reaction: A -> B -> C
    
            d[A]/dt = -k1[A]
            d[B]/dt =  k1[A] - k2[B]
            d[C]/dt =  k2[B]

    
    connectivity = np.array([[0, 1, 0],
                             [0, 0, 1],
                             [0, 0, 0]])
   
    ks = np.array([1, 1])
    k1,k2 = ks

    k = np.array([[-k1,  0,  0],
                  [ k1, -k2, 0],
                  [ 0,   k2, 0 ]])

    """
    
    connectivity = connectivity.astype(np.float64)
    rxn_locs = np.nonzero(connectivity)
    assert len(ks) == len(rxn_locs)


    connectivity[rxn_locs] = ks

    #negative diagonal
    k_diag = -connectivity.sum(axis=1)
    #positive off-diagonal
    k =  connectivity.T.copy()

    #building the final k matrix
    k[np.diag_indices_from(k)] = k_diag
    return k

class Kinetics_Fit(object):
    def __init__(self, connectivity, t, traces, wavelengths=None, species=None, 
                       guess_ks=None, guess_x0=None, guess_abs=None):

        self.connectivity = connectivity.astype(np.float64) 
        self.t = t
        self.traces = traces

        self.no_traces = self.traces.shape[1]
        self.no_species = self.connectivity.shape[0]
        self.no_rate_constants = len(np.nonzero(self.connectivity))
        self.no_time_steps = len(self.t)
    
        #variables used to fit multiple traces
        
        self.stacked_times = np.hstack([self.t for i in range(self.no_traces)]) #not actually used
        self.stacked_traces = self.traces.T.reshape(self.no_traces * self.no_time_steps)
    
        if wavelengths is None:
            self.wavelengths = range(self.no_traces)
        else:
            assert len(wavelengths) == self.no_traces
            self.wavelengths = wavelengths

        if species is None:
            self.species = range(self.no_species)
        else:
            assert len(species) == self.no_species
            self.species = species
            
        if guess_ks is None:
            self.guess_ks = np.ones(self.no_rate_constants).astype(np.float64)
        else:
            self.guess_ks = guess_ks
            
        if guess_x0 is None:
            self.guess_x0 = np.zeros(self.no_species).astype(np.float64)
            self.guess_x0[0] = 1
        else:
            self.guess_x0 = guess_x0
            
        if guess_abs is None:
            self.guess_abs = np.zeros([self.no_traces,self.no_species]).astype(np.float64)
        else:
            self.guess_abs = guess_abs
            
    def sim_single_abs(self, t, ks, abs_factor):
        """Simulation of a single spectral trace"""
        X0 = self.guess_x0
        
        k=K(ks,connectivity)
        concs = X(t,k,X0)
        
        spectral_contribs = concs * abs_factor
        return np.sum(spectral_contribs, axis=1)
    
    def sim_multi_abs(self, t, ks, abs_factors):
        """Simulate multiple spectral traces"""

        sim_traces = []
        for i in range(self.no_traces):
            trace_params = list(ks)+list(abs_factors[i])
            sim_traces.append( self.sim_single_abs(t, ks, abs_factors[i]) )

        return np.array(sim_traces)
    
    def sim_stacked_abs(self,stacked_t, *params):
        """Master fitting function

        Stacked_t is a dummy variables that we aren't actually using during the fit
        """
        
        assert len(params) == self.no_rate_constants + self.no_traces * self.no_species
        
        #print(params)
        ks = list(params[:self.no_rate_constants])
        r_abs_factors = params[self.no_rate_constants:self.no_rate_constants+self.no_traces*self.no_species]
        r_abs_factors = np.array(r_abs_factors)
        abs_factors = r_abs_factors.reshape([self.no_traces,self.no_species])

        t = self.t
        return self.sim_multi_abs(t, ks, abs_factors).ravel()
        
    def fit_data(self):
        p0 = list(self.guess_ks) + list(self.guess_abs.ravel())
        p,conv = curve_fit(self.sim_stacked_abs, self.stacked_times , self.stacked_traces, p0=p0)
        
        self.final_ks = p[:self.no_rate_constants]
        self.final_abs_factors = np.array(p[self.no_rate_constants:])
        self.final_abs_factors = self.final_abs_factors.reshape([self.no_traces,self.no_species])
        
        temp_r = (self.sim_stacked_abs(self.stacked_times, *p) - self.stacked_traces)**2
        self.final_resids = temp_r.sum()
        
    def plot_fit(self):
        sim_traces = self.sim_multi_abs(self.t,self.final_ks,self.final_abs_factors)
        for sim_trace in sim_traces:
            plt.plot(self.t, sim_trace, label='simulated')
           
        for trace in self.traces.T:
            plt.plot(self.t, trace, label='data')

        plt.legend()
        plt.show()
        
    def plot_spectra(self):
        for species,species_abs in zip(self.species,self.final_abs_factors.T):
            plt.plot(self.wavelengths, species_abs,label='species {s}'.format(s=species))
        plt.legend()
        plt.show()
