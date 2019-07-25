"""
Utilities to help Adaptive Bayesian Inference computations and tests on 
USID Datasets and visualize results
Created on Tue Jul 02 2019
@author: Alvin Tan, Emily Costa
"""
import os
import time
import math
import cupy as cp
import numpy as np
import scipy.linalg as spla
from matplotlib import pyplot as plt

# Libraries for USID database use
import h5py
import pycroscopy as px 
import pyUSID as usid 

# Library to speed up adaptive metropolis
#from numba import jit 

# Takes in the sine full_V wave and finds the index used to shift the waveform into
# a forward and reverse sweep. Then finds the index of the maximum value (i.e. the
# index used to split the waveform into forward and reverse sections). It returns
# both indices as well as a shifted full_V
def get_shift_and_split_indices(full_V):
    # Since we have a sine wave (which goes up, then down, then up),
    # we want to take the last bit that goes up and move it to the
    # front, such that the data first goes from -6 to 6 V, then
    # from 6 to -6 V.
    full_V = cp.asarray(full_V)
    shift_index = full_V.size - 1
    while(full_V[shift_index-1] < full_V[shift_index]):
        shift_index -= 1

    full_V = cp.concatenate((full_V[shift_index:], full_V[:shift_index]))

    # Then we do another walk to get the boundary between the
    # forward and reverse sections.
    split_index = 0
    while(full_V[split_index] < full_V[split_index + 1]):
        split_index += 1

    return full_V, shift_index, split_index


# Takes in a full_i_meas response and shifts it according to the given shift index
def get_shifted_response(full_i_meas, shift_index):
    full_i_meas = cp.asarray(full_i_meas)
    shift_index = cp.asarray(shift_index)
    return cp.concatenate((full_i_meas[shift_index:], full_i_meas[:shift_index]))


# Takes a shifted array and un-shifts it according to the given shift index
def get_unshifted_response(full_i_meas, shift_index):
    full_i_meas = cp.asarray(full_i_meas)
    shift_index = cp.asarray(shift_index)
    new_shift_index = full_i_meas.size - shift_index
    return cp.concatenate((full_i_meas[new_shift_index:], full_i_meas[:new_shift_index]))


# Takes in excitation wave amplitude and desired M value. Returns M, dx, and x.
def get_M_dx_x(V0=6, M=25):
    dx = 2*V0/(M-2)
    #x = cp.arange(-V0, V0+dx, dx)[cp.newaxis].T
    x = np.arange(-V0, V0+dx, dx)
    x = x.astype(float)
    x = cp.asarray(x)
    x = cp.expand_dims(x,0)
    x = cp.transpose(x)
    M = x.size # M may not be the desired value but it will be very close
    return M, dx, x


# Takes in a single period of a shifted excitation wave as full_V and the corresponding
# current response as full_i_meas. Returns either the estimated resistances and 
# reconstructed currents or a pyplot figure.
def process_pixel(full_i_meas, full_V, split_index, M, dx, x, shift_index, f, V0, Ns, dvdt, pix_ind=0, graph=False, verbose=False):
    # If verbose, check if full_V and full_i_meas exist and are actually 1D
    if verbose:
        if full_V is None:
            raise Exception("full_V is None")
        if full_i_meas is None:
            raise Exception("full_i_meas is None")
        if len(full_V.shape) != 1:
            raise Exception("full_V is not one-dimensional. Its shape is {}".format(full_V.shape))
        if len(full_i_meas.shape) != 1:
            raise Exception("full_i_meas is not one-dimensional. Its shape is {}".format(full_i_meas.shape))
        if full_V.size != full_i_meas.size:
            raise Exception("full_V and full_i_meas do not have the same length. full_V is of length {} while full_i_meas is of length {}".format(full_V.size, full_i_meas.size))

    # Split up our data into forward and reverse sections
    Vfor = full_V[:split_index]
    Vrev = full_V[split_index:]
    Ifor = full_i_meas[:split_index]
    Irev = full_i_meas[split_index:]
    dvdtFor = dvdt[:split_index]
    dvdtRev = dvdt[split_index:]

    # Run the adaptive metropolis on both halves and save the results
    forward_results = _run_bayesian_inference(Vfor, Ifor, M, dx, x, f, V0, Ns, dvdtFor, verbose=verbose)
    reverse_results = _run_bayesian_inference(Vrev, Irev, M, dx, x, f, V0, Ns, dvdtRev, verbose=verbose)

    '''
    # If we want a graph, we graph our data and return the figure
    #if(graph):
        
        R, R_sig, capacitance, i_recon, i_corrected = forward_results
        forward_graph = _get_simple_graph(x, R, R_sig, Vfor, Ifor, i_recon, i_corrected)

        R, R_sig, capacitance, i_recon, i_corrected = reverse_results
        reverse_graph = _get_simple_graph(x, R, R_sig, Vrev, Irev, i_recon, i_corrected)

        return forward_graph, reverse_graph
        

    else:'''
    # Concatenate the forward and reverse results together and return in a tuple
    # for easier parallel processing
    # Note, results are (R, R_sig, capacitance, i_recon, i_corrected)
    R = cp.concatenate((forward_results[0], reverse_results[0]), axis=0)
    R_sig = cp.concatenate((forward_results[1], reverse_results[1]), axis=0)
    capacitance = cp.array([forward_results[2], reverse_results[2]])
    i_recon = cp.concatenate((forward_results[3], reverse_results[3]), axis=0)
    i_corrected = cp.concatenate((forward_results[4], reverse_results[4]), axis=0)
    # Shift i_recon and i_corrected back to correspond to a sine excitation wave
    i_recon = get_unshifted_response(i_recon, shift_index)
    i_corrected = get_unshifted_response(i_corrected, shift_index)

    if(graph):
        # calls publicGetGraph(Ns, pix_ind, shift_index, split_index, x, R, R_sig, V, i_meas, i_recon, i_corrected)
        full_V = get_unshifted_response(full_V, shift_index)
        full_i_meas = get_unshifted_response(full_i_meas, shift_index)
        x = cp.concatenate((x, x))
        return publicGetGraph(Ns, pix_ind, shift_index, split_index, x, R, R_sig, full_V, full_i_meas, i_recon, i_corrected)
    else:
        return R, R_sig, capacitance, i_recon, i_corrected

# Helper function because Numba crashes when it isn't supposed to
#@jit(nopython=True)
def _logpo_R1_fast(pp, A, V, dV, y, gam, P0, mm):
    out = cp.linalg.norm(V*cp.exp(-A[:, :-1] @ pp[:-2]) +\
          pp[-2][0] * (dV + pp[-1][0]*V) - y)**2/2/gam/gam + (((pp[:-2]-mm[:-2]).T @ P0) @ (pp[:-2]-mm[:-2]))[0][0]/2
    return out

# Does math stuff and returns a number relevant to some probability distribution.
# Used only in the while loop of run_bayesian_inference() (and once before to initialize)
def _logpo_R1(pp, A, V, dV, y, gam, P0, mm, Rmax, Rmin, Cmax, Cmin):
    if pp[-1] > Rmax or pp[-1] < Rmin:
        return cp.inf
    if pp[-2] > Cmax or pp[-2] < Cmin:
        return cp.inf
    '''
    out = cp.linalg.norm(V*cp.exp(cp.matmul(-A[:, :-1], pp[:-2])) + \
                         pp[-2][0] * (dV + pp[-1][0]*V) - y)**2/2/gam/gam + \
          cp.matmul(cp.matmul((pp[:-2]-mm[:-2]).T, P0), pp[:-2]-mm[:-2])/2
    '''
    return _logpo_R1_fast(pp, A, V, dV, y, gam, P0, mm)


def _run_bayesian_inference(V, i_meas, M, dx, x, f, V0, Ns, dvdt, verbose=False):
    '''
    Takes in raw filtered data, parses it down and into forward and reverse sweeps,
    and runs an adaptive metropolis alglrithm on the data. Then calculates the
    projected resistances and variances for both the forward and reverse sweeps, as
    well as the reconstructed current.

    Parameters
    ----------
    V:          np.ndtype row vector of dimension 1xN
                the excitation waveform; assumed to be a forward or reverse sweep
    i_meas:     np.ndtype row vector of dimension 1xN
                the measured current resulting from the excitation waveform
    f:          int
                the frequency of the excitation waveform (Hz)
    V0:         int
                the amplitude of the excitation waveform (V)
    Ns:         int
                the number of iterations we want the adaptive metropolis to run
    verbose:    boolean
                prints debugging messages if True

    Returns
    -------
    R:              np.ndtype column vector
                    the estimated resistances
    R_sig:          np.ndtype column vector
                    the standard deviations of R resistances 
    capacitance:    float
                    the capacitance of the setup
    i_recon:        np.ndtype column vector
                    the reconstructed current
    i_corrected:    np.ndtype column vector
                    the measured current corrected for capacitance
    '''
    # Grab the start time so we can see how long this takes
    if(verbose):
        start_time = time.time()

    # Setup some constants that will be used throughout the code
    nt = 1000;
    nx = 32;

    r_extra = 110
    ff = 1e0

    gam = 0.01
    sigc = 10
    sigma = 1

    tmax = 1/f/2
    t = np.linspace(0, tmax, V.size)
    dt = t[1] - t[0]
    dV = cp.diff(V)
    dV = cp.divide(dV,dt)
    dV = cp.concatenate((dV,cp.asarray([dV[-1]], dtype='float64')))
    N = V.size
    M = int(M)
    #dx = 2*V0/(M-2)
    #x = cp.arange(-V0, V0+dx, dx)[cp.newaxis].T
    #M = x.size # M may not be the desired value but it will be very close

    # Change V and dV into column vectors for computations
    # Note: V has to be a row vector for cp.diff(V) and
    # max(V) to work properly
    dV = cp.transpose(cp.expand_dims(dV, axis=0))
    #dV = dV[cp.newaxis].T
    V = cp.transpose(cp.expand_dims(V, axis=0))
    # V = V[cp.newaxis].T
    i_meas = cp.transpose(cp.expand_dims(i_meas, axis=0))
    #i_meas = i_meas[cp.newaxis].T

    # Build A : the forward map
    #look to optimize for-loop
    A = cp.zeros((N, M + 1))
    for j in range(N):
        # Note: ix will be used to index into arrays, so it is one less
        # than the ix used in the Matlab code
        ix = math.floor((V[j] + V0)/dx) + 1
        ix = min(ix, x.size - 1)
        ix = max(ix, 1)
        A[j, ix] = int(cp.divide(cp.subtract(V[j],x[ix-1]),cp.subtract(x[ix],x[ix-1])))
        A[j, ix-1] = int(cp.subtract(1, cp.divide(cp.subtract(V[j],x[ix-1]),
                                                  cp.subtract(x[ix],x[ix-1]))))
        #A[j, ix-1] = (1 - (V[j] - x[ix-1])/(x[ix] - x[ix-1]));
    A[:, M] = cp.squeeze(cp.add(dV,ff*r_extra*V))
    
    # Similar to above, but used to simulate data and invert for E(s|y)
    # for initial condition
    A1 = cp.zeros((N, M + 1))
    for j in range(N):
        # Note: Again, ix is one less than it is in the Matlab code
        ix = math.floor((V[j] + V0)/dx)+1
        ix = min(ix, x.size - 1)
        ix = max(ix, 1)
        A[j, ix] = int(cp.divide(cp.subtract(V[j],x[ix-1]),cp.subtract(x[ix],x[ix-1])))
        A[j, ix-1] = int(cp.subtract(1, cp.divide(cp.subtract(V[j],x[ix-1]),
                                                  cp.subtract(x[ix],x[ix-1]))))
    A1[:, M] = cp.squeeze(cp.add(dV,ff*r_extra*V)) # transpose again here

    # A rough guess for the initial condition is a bunch of math stuff
    # This is an approximation of the Laplacian
    # Note: have to do inconvenient things with x because it is a column vector
    Lap = (-cp.diag(cp.power(x.T[0][:-1], 0), -1) - cp.diag(cp.power(x.T[0][:-1], 0), 1) 
        + 2*cp.diag(cp.power(x.T[0], 0), 0))/dx/dx
    Lap[0, 0] = 1/dx/dx
    Lap[-1, -1] = 1/dx/dx

    P0 = cp.zeros((M+1, M+1))
    P0[:M, :M] = (1/sigma/sigma)*(cp.eye(M) + cp.matmul(Lap, Lap))
    P0[M, M] = 1/sigc/sigc

    Sigma = cp.linalg.inv(cp.matmul(A1.T, A1)/gam/gam + P0)
    m = cp.matmul(Sigma, cp.matmul(A1.T, i_meas)/gam/gam)

    # Tuning parameters
    Mint = 1000
    Mb = 100
    r = 1.1
    beta = 1
    nacc = 0
    #P = cp.zeros((M+2, Ns))
    # Only store a million samples to save space
    num_samples = min(Ns, int(1e6))
    P = cp.zeros((M+2, num_samples))

    # Define prior
    SS = cp.matmul(cp.sqrt(Sigma), cp.random.randn(M+1, num_samples)) + cp.tile(m, (1, num_samples))
    RR = cp.concatenate((cp.log(1/cp.maximum(SS[:M, :], cp.full((SS[:M, :]).shape, 
        cp.finfo(float).eps))), SS[M, :][cp.newaxis]), axis=0)
    mr = 1/num_samples*cp.sum(RR, axis=1)[cp.newaxis].T
    SP = 1/num_samples*cp.matmul(RR - cp.tile(mr, (1, num_samples)), (RR - cp.tile(mr, (1, num_samples))).T)
    amp = 100
    C0 = amp**2 * SP[:M, :M]
    SR = cp.sqrt(C0)

    # Initial guess for Sigma from Eq 1.8 in the notes
    S = cp.concatenate((cp.concatenate((SR, cp.zeros((2, M))), axis=0), 
        cp.concatenate((cp.zeros((M, 2)), amp*cp.array([[1e-2, 0], [0, 1e-1]])), axis=0)), axis=1)
    S2 = cp.matmul(S, S.T)
    S1 = cp.zeros((M+2, 1))
    print(mr.shape)
    print(type(mr))
    print(cp.expand_dims(cp.expand_dims(cp.asarray(r_extra), axis=0), axis=1))
    mm = cp.concatenate((mr, cp.expand_dims
        (cp.expand_dims(cp.asarray(r_extra), axis=0).shape, axis=1)))
    print(mm.shape)
    mm = cp.transpose(cp.expand_dims(mm))
    #mm = cp.append(mr, r_extra)[cp.newaxis].T
    ppp = mm.astype(cp.float64)
    breakpoint()

    # for some reason, this may throw a cp.linalg.LinAlgError: Singular Matrix
    # when trying to process the 0th pixel. After exiting this try catch block,
    # the concatinations in process pixel fails.
    try:
        P0 = cp.linalg.inv(C0)
    except cp.linalg.LinAlgError:
        print("P0 failed to instantiate.")
        # return zero versions of R, R_sig, capacitance, i_recon, i_corrected
        return cp.zeros(x.shape), cp.zeros(x.shape), 0, cp.zeros(i_meas.shape), cp.zeros(i_meas.shape)
    

    # Now we are ready to start the active metropolis
    if verbose:
        print("Starting active metropolis...")
        met_start_time = time.time()

    i = 0
    j = 0
    Rmax = 120
    Rmin = 100
    Cmax = 10
    Cmin = 0
    logpold = _logpo_R1(ppp, A, V, dV, i_meas, gam, P0, mm, Rmax, Rmin, Cmax, Cmin)

    while i < Ns:
        pppp = ppp + beta*cp.matmul(S, cp.random.randn(M+2, 1)).astype(cp.float64) # using pp also makes gdb bug out
        logpnew = _logpo_R1(pppp, A, V, dV, i_meas, gam, P0, mm, Rmax, Rmin, Cmax, Cmin)
        
        # accept or reject
        # Note: unlike Matlab's rand, which is a uniformly distributed selection from (0, 1),
        # Python's cp.random.rand is a uniformly distributed selection from [0, 1), and can
        # therefore be 0 (by a very small probability, but still). Thus, a quick filter must
        # be made to prevent us from trying to evaluate log(0).
        randBoi = cp.random.rand()
        while randBoi == 0:
            randBoi = cp.random.rand()
        if cp.log(randBoi) < logpold - logpnew:
            ppp = pppp
            logpold = logpnew
            nacc += 1 # count accepted proposals

        # stepsize adaptation
        if (i+1) % Mb == 0:
            # estimate acceptance probability, and keep near 0.234
            rat = nacc/Mb

            if rat > 0.3:
                beta = beta*r
            elif rat < 0.1:
                beta = beta/r

            nacc = 0

        # Save only the last num_samples samples to save space
        P[:, i%num_samples] = ppp.T[0]

        # proposal covariance adaptation
        if (i+1) % Mint == 0:
            # Get index values to fit into the smaller matrix
            jMintStart = (j-Mint)%num_samples
            jMintEnd = j%num_samples
            iMintStart = (i-Mint+1)%num_samples
            iMintEnd = (i+1)%num_samples

            if (j+1) == Mint:
                S1lag = cp.sum(P[:, :j] * P[:, 1:j+1], axis=1)[cp.newaxis].T / j
            else:
                S1lag = (j - Mint)/j*S1lag + cp.sum(P[:, jMintStart:jMintEnd] * P[:, jMintStart+1:jMintEnd+1], axis=1)[cp.newaxis].T / j

            # Update meani based on Mint batch
            S1 = (j+1-Mint)/(j+1)*S1 + cp.sum(P[:, iMintStart:iMintEnd], axis=1)[cp.newaxis].T/(j+1)
            # Update Sigma based on Mint batch
            S2 = (j+1-Mint)/(j+1)*S2 + cp.matmul(P[:, iMintStart:iMintEnd], P[:, iMintStart:iMintEnd].T)/(j+1)
            #print("P's shape is {}".format(P.shape))
            # Approximate L such that L*L' = Sigma, where the second term is a
            # decaying regularization
            # for some reason this may throw a cp.linalg.LinAlgError: Matrix is not positive definite
            # If this fails, just return zeros
            try:
                S = cp.linalg.cholesky(S2 - cp.matmul(S1, S1.T) + (1e-3)*cp.eye(M+2)/(j+1))
            except cp.linalg.LinAlgError:
                print("Cholesky failed on iteration {}".format(i+1))
                return cp.zeros(x.shape), cp.zeros(x.shape), 0, cp.zeros(i_meas.shape), cp.zeros(i_meas.shape)

            if verbose and ((i+1)%1e5 == 0):
                print("i = {}".format(i+1))

        i += 1
        j += 1

    if verbose:
        print("Finished Adaptive Metropolis!\nAdaptive Metropolis took {}.\nTotal time taken so far is {}.".format(time.time() - met_start_time, time.time() - start_time))

    # m is a column vector, and the last element is the capacitance exponentiated
    #capacitance = math.log(m[-1][0])
    capacitance = pppp[-2][0]
    r_extra = pppp[-1][0]

    # Mean and variance of resistance
    meanr = cp.matmul(cp.exp(P[:M,:]), cp.ones((num_samples, 1))) / num_samples
    mom2r = cp.matmul(cp.exp(P[:M,:]), cp.exp(P[:M,:]).T) / num_samples
    varr = mom2r - cp.matmul(meanr, meanr.T)

    R = meanr.astype(cp.float)
    R_sig = cp.sqrt(cp.diag(varr[:M, :M]))[cp.newaxis].T.astype(cp.float)

    # Reconstruction of the current
    i_recon = V * cp.matmul(cp.exp(cp.matmul(-A[:, :M], P[:M, :])), cp.ones((num_samples, 1))) / (num_samples) + \
            cp.matmul(cp.tile(P[M,:], (N, 1))*(cp.tile(dV, (1, num_samples)) + P[M+1, :]*cp.tile(V, (1, num_samples))), \
                      cp.ones((num_samples, 1))) / num_samples

    # Adjusting for capacitance
    point_i_cap = capacitance * dvdt
    point_i_extra = r_extra * 2 * capacitance * V
    i_corrected = i_meas - point_i_cap - point_i_extra

    #breakpoint()

    return R, R_sig, capacitance, i_recon, i_corrected


def _get_simple_graph(x, R, R_sig, V, i_meas, i_recon, i_corrected):
    # Clean up R and R_sig for unsuccessfully predicted resistances
    for i in range(R_sig.size):
        if cp.isnan(R_sig[i]) or R_sig[i] > 100:
            R_sig[i] = cp.nan
            R[i] = cp.nan

    #breakpoint()

    # Create the figure to be returned
    result = plt.figure()

    # Plot the resistance estimation on the left subplot
    plt.subplot(121)
    plt.plot(x, R, "gx-", label="ER")
    plt.plot(x, R+R_sig, "rx:", label="ER+\u03C3_R")
    plt.plot(x, R-R_sig, "rx:", label="ER-\u03C3_R")
    plt.legend()

    # Plot the current data on the right subplot
    plt.subplot(122)
    plt.plot(V, i_meas, "ro", mfc="none", label="i_meas")
    plt.plot(V, i_recon, "gx-", label="i_recon")
    plt.plot(V, i_corrected, "bo", label="i_corrected")
    plt.legend()

    return result


def publicGetGraph(Ns, pix_ind, shift_index, split_index, x, R, R_sig, V, i_meas, i_recon, i_corrected):
    #return _get_simple_graph(x, R, R_sig, V, i_meas, i_recon, i_corrected)
    rLenHalf = R_sig.size//2

    shiftV = get_shifted_response(V, shift_index)
    i_corrected = get_shifted_response(i_corrected, shift_index)

    # Clean up R and R_sig for unsuccessfully predicted resistances
    for i in range(rLenHalf*2):
        if cp.isnan(R_sig[i]) or R_sig[i] > 100:
            R_sig[i] = cp.nan
            R[i] = cp.nan

    #breakpoint()

    # Create the figure to be returned
    result = plt.figure()

    plt.suptitle("Pixel {} after {} iterations".format(pix_ind, Ns))

    # Plot the forward resistance estimation on the left subplot
    plt.subplot(131)
    plt.title("Forward resistance")
    plt.plot(x[:rLenHalf], R[:rLenHalf], "gx-", label="ER")
    plt.plot(x[:rLenHalf], R[:rLenHalf]+R_sig[:rLenHalf], "rx:", label="ER+\u03C3_R")
    plt.plot(x[:rLenHalf], R[:rLenHalf]-R_sig[:rLenHalf], "rx:", label="ER-\u03C3_R")
    plt.legend()

    # Plot the reverse resistance estimation on the middle subplot
    plt.subplot(132)
    plt.title("Reverse resistance")
    plt.plot(x[rLenHalf:], R[rLenHalf:], "gx-", label="ER")
    plt.plot(x[rLenHalf:], R[rLenHalf:]+R_sig[rLenHalf:], "rx:", label="ER+\u03C3_R")
    plt.plot(x[rLenHalf:], R[rLenHalf:]-R_sig[rLenHalf:], "rx:", label="ER-\u03C3_R")
    plt.legend()

    # Plot the current data on the right subplot
    plt.subplot(133)
    plt.title("Current")
    plt.plot(V, i_meas, "ro", mfc="none", label="i_meas")
    plt.plot(V, i_recon, "gx-", label="i_recon")
    plt.plot(shiftV[:split_index], i_corrected[:split_index], "bx", label="forward i_corrected")
    plt.plot(shiftV[split_index:], i_corrected[split_index:], "yx", label="reverse i_corrected")
    plt.legend()

    return result

