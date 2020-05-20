"""
Asymmetric Least Squares fitter implementations.

For a description of each method, check the links provided with some functions or google it.

The default values for each function call are just a starting point and are not optimized!
There is a way to optimize the values as described in the original paper, but the method is not yet implemented.

The source codes are collected from various sources on the internet. Some links are provided.
"""

from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky
from scipy.stats import norm
import numpy as np



def als(y, lam=1e6, p=0.05, niter=10):
    '''
    y is a 1-dim numpy array, which should only consist of finite values
    If nan or infinite value present, use nanals.
    
    SOURCE
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library/29185844
    
    two parameters: p for asymmetry and λ for smoothness
    0.001 ≤ p ≤ 0.1 is a good choice (for a signal with positive peaks) and 
    10^2 ≤ λ ≤ 10^9
    
    vary λ on a grid that is approximately linear for log λ
    '''
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    
    return z



def nanals(iy, lam=1e6, p=0.05, niter=10):
    '''
    SOURCE
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library/29185844
    
    two parameters: p for asymmetry and λ for smoothness
    0.001 ≤ p ≤ 0.1 is a good choice (for a signal with positive peaks) and 
    10^2 ≤ λ ≤ 10^9
    
    vary λ on a grid that is approximately linear for log λ
    
    this fitter first removes nan's from the data, but returns an array of same size as input array
    '''
    nsel = np.isfinite(iy)
    y = iy[nsel]
    
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    
    zr = iy.copy()
    zr[nsel] = z
    
    return zr



def airpls(y, lam=1e4, niter=10):
    '''
    SOURCE
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library/29185844
    '''
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        #######################################################################################
        # Modification
        target = 1e-3*np.sum(np.abs(y))
        if np.sum(np.abs(y-z)*(y-z < 0)) < target:
            break
        w = 0 * (y > z) + np.exp(-i * (y - z) / np.sum(np.abs(y - z)*(y - z < 0))) * (y < z)
        #######################################################################################

    return z



def nanairpls(iy, lam=1e4, niter=10):
    '''
    SOURCE
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library/29185844
    '''
    nsel = np.isfinite(iy)
    y = iy[nsel]
    
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        #######################################################################################
        # Modification
        target = 1e-3*np.sum(np.abs(y))
        if np.sum(np.abs(y-z)*(y-z < 0)) < target:
            break
        w = 0 * (y > z) + np.exp(-i * (y - z) / np.sum(np.abs(y - z)*(y - z < 0))) * (y < z)
        #######################################################################################

    
    zr = iy.copy()
    zr[nsel] = z
    
    return zr



def psalsa(y, lam=1e6, p=0.1, k=1e5, niter=10):
    '''
    SOURCE
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library/29185844
    '''
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        ######################################################################################
        # Modification
        w = p * np.exp(-(y - z) / k) * (y > z) + (1 - p) * (y < z)
        ######################################################################################
    
    return z


def arpls(y, lam=1e4, ratio=0.05, itermax=100):
    """
    Inputs:
        y:  
            input data (1-dim spectrum)
        lam:
            parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting background, z
        ratio:
            wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
        itermax:
            number of iterations to perform
    Output:
        the fitted background vector

    """
    
    N = len(y)
#  D = sparse.csc_matrix(np.diff(np.eye(N), 2))
    D = sparse.eye(N, format='csc')
    D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    D = D[1:] - D[:-1]

    H = lam * D.T * D
    w = np.ones(N)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        WH = sparse.csc_matrix(W + H)
        C = sparse.csc_matrix(cholesky(WH.todense()))
        z = spsolve(C, spsolve(C.T, w * y))
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt
    return z



def nanarpls(iy, lam=1e4, ratio=0.05, itermax=100):
    """
    Inputs:
        y:
            input data (1-dim spectrum, can contain nans or infinite values, which are ignored)
        lam:
            parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting background, z
        ratio:
            wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
        itermax:
            number of iterations to perform
    Output:
        the fitted background vector

    """
    nsel = np.isfinite(iy)
    y = iy[nsel]

    N = len(y)
#  D = sparse.csc_matrix(np.diff(np.eye(N), 2))
    D = sparse.eye(N, format='csc')
    D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    D = D[1:] - D[:-1]

    H = lam * D.T * D
    w = np.ones(N)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        WH = sparse.csc_matrix(W + H)
        C = sparse.csc_matrix(cholesky(WH.todense()))
        z = spsolve(C, spsolve(C.T, w * y))
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt
    
    zr = iy.copy()
    zr[nsel] = z
    
    return zr





def arplsw(y, lam=1e4, ratio=0.05, itermax=100):
    """
    Inputs:
        y:
            input data (1-dim spectrum)
        lam:
            parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting background, z
        ratio:
            wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
        itermax:
            number of iterations to perform
    Output:
        z: the fitted background vector 
        
        w: return the weights vector 

    """
    N = len(y)
#  D = sparse.csc_matrix(np.diff(np.eye(N), 2))
    D = sparse.eye(N, format='csc')
    D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    D = D[1:] - D[:-1]

    H = lam * D.T * D
    w = np.ones(N)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        WH = sparse.csc_matrix(W + H)
        C = sparse.csc_matrix(cholesky(WH.todense()))
        z = spsolve(C, spsolve(C.T, w * y))
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt
    return z, w





                
if __name__ == '__main__':
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    
    # example data
    path = './test_data.txt'
    ydata = np.genfromtxt(path, skip_header=3, usecols=[4])
    
    # start the fitting
    als = als(-ydata)
    airpls = airpls(-ydata)
    arpls = arpls(-ydata, lam=1e6, ratio=0.02)
    psalsa = psalsa(-ydata)
    
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    
    ax[0].plot(-ydata, color='black', linewidth=1)
    ax[0].plot(als, color='orange', linewidth=3)
    ax[0].plot(airpls, color='blue', linewidth=1)
    ax[0].plot(arpls, color='green', linewidth=1)
    ax[0].plot(psalsa, '--', color='magenta', linewidth=1)
    
    ax[1].plot(-ydata - als, color='orange', linewidth=3, label='ALS')
    ax[1].plot(-ydata - airpls, color='blue', linewidth=1, label='airPLS')
    ax[1].plot(-ydata - arpls, color='green', linewidth=1, label='arPLS')
    ax[1].plot(-ydata - psalsa, color='magenta', linewidth=1, label='psalsa')
    
    plt.show()
    

