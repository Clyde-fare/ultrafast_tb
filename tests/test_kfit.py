import numpy as np
from ultrafast_toolbox.kfit import X,K, Kinetics_Fit

def sim_concs(t, n, *params):
    params = np.array(params)
    connectivity = params[:n**2].reshape([n,n])

    n_ks = len(np.nonzero(connectivity))
    ks = params[n**2:n**2+n_ks]

    k = K(ks,connectivity)
    no_components = k.shape[0]
    
    X0 = np.zeros(no_components)
    X0[0] = 1

    return X(t,k,X0).T

connectivity = np.array([[0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0]])
ks = np.array([1,1])
params = list(connectivity.ravel()) + list(ks)

t = np.linspace(0,10)
temp_test_abs = sim_concs(t,3,*params) * np.array([[1,0,0]]).T
temp_test_abs2 = sim_concs(t,3,*params) * np.array([[0,1,0]]).T
test_abs = np.sum(temp_test_abs.T,axis=1)
test_abs2 = np.sum(temp_test_abs2.T,axis=1)
traces = np.array([test_abs,test_abs2]).T

kf = Kinetics_Fit(connectivity, t, traces, guess_ks=np.array([10,4]) )
kf.fit_data()

tol=0.0000001
assert (kf.final_ks - np.array([ 1.00004232,  0.99995769]) < tol).all()

rec_abs_factors = np.array([[  1.00000000e+00,   4.23120092e-05,  -6.20073604e-11],
                                 [ -5.53728942e-10,   9.99957689e-01,   4.22350517e-12]])

assert (kf.final_abs_factors - rec_abs_factors  < tol).all()

assert kf.final_resids - 8.9281410093166487e-19 < tol
