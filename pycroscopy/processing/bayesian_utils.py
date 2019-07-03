"""
Utilities to help Adaptive Bayesian Inference computations and tests on 
USID Datasets and visualize results
Created on Tue Jul 02 2019
@author: Alvin Tan, Emily Costa
"""
import os
import time
import math
import numpy as np
import scipy.linalg as spla
from matplotlib import pyplot as plt

# Libraries for USID database use
import h5py
import pycroscopy as px 
import pyUSID as usid 

# Does math stuff and returns a number relevant to some probability distribution.
# Used only in the while loop of run_bayesian_inference() (and once before to initialize)
def _logpo_R1(pp, A, V, dV, y, gam, P0, mm, Rmax, Rmin, Cmax, Cmin):
    if pp[-1] > Rmax or pp[-1] < Rmin:
        return np.inf
    if pp[-2] > Cmax or pp[-2] < Cmin:
        return np.inf
    
    out = np.linalg.norm(V*np.exp(np.matmul(-A[:, :-1], pp[:-2])) + \
                         pp[-2][0] * (dV + pp[-1][0]*V) - y)**2/2/gam/gam + \
          np.matmul(np.matmul((pp[:-2]-mm[:-2]).T, P0), pp[:-2]-mm[:-2])/2

    return out

def run_bayesian_inference(V, i_meas, f=200, V0=6, Ns=int(1e6), verbose=False):
	'''
	Takes in raw filtered data, parses it down and into forward and reverse sweeps,
	and runs an adaptive metropolis alglrithm on the data. Then calculates the
	projected resistances and variances for both the forward and reverse sweeps, as
	well as the reconstructed current.

	Parameters
	----------
	V:			numpy.ndtype row vector of dimension 1xN
				the excitation waveform; assumed to be a single period of a sine wave
	i_meas:		numpy.ndtype row vector of dimension 1xN
				the measured current resulting from the excitation waveform
	f:			int
				the frequency of the excitation waveform (Hz)
	V0:			int
				the amplitude of the excitation waveform (V)
	Ns:			int
				the number of iterations we want the adaptive metropolis to run
	verbose:	boolean
				prints debugging messages if True

	Returns
	-------
	R_forward:		numpy.ndtype column vector
					the reconstructed resistances for the forward sweep
	R_reverse:		numpy.ndtype column vector
					the reconstructed resistances for the reverse sweep
	R_sig_forward:	numpy.ndtype column vector
					the standard deviations of R_forward resistances 
	R_sig_reverse:	numpy.ndtype column vector
					the standard deviations of R_reverse resistances
	i_recon:		numpy.ndtype column vector
					the reconstructed current
	'''
	# Grab the start time so we can see how long this takes
	if(verbose):
		startTime = time.time()

	# Setup some constants that will be used throughout the code
    Rmax = 100
    Rmin = -1e-6

    nt = 1000;
    nx = 32;

    r_extra = 0.110
    ff = 1e0

    gam = 0.01
    sigc = 10
    sigma = 1

    tmax = 1/f/2
    t = np.linspace(0, tmax, V.size)
    dt = t[1] - t[0]
    dV = np.diff(V)/dt
    dV = np.append(dV, dV[dV.size-1])
    N = V.size 
    x = np.arange(-V0, V0+dx, dx)[np.newaxis].T
    dx = x[1] - x[0]
    M = x.size

    # Change V and dV into column vectors for computations
    # Note: V has to be a row vector for np.diff(V) and
    # max(V) to work properly
    dV = dV[np.newaxis].T
    V = V[np.newaxis].T
    i_meas = i_meas[np.newaxis].T

    # Build A : the forward map
    A = np.zeros((N, M + 1)).astype(complex)
    for j in range(N):
        # Note: ix will be used to index into arrays, so it is one less
        # than the ix used in the Matlab code
        ix = math.floor((V[j] + V0)/dx) + 1
        ix = min(ix, x.size - 1)
        ix = max(ix, 1)
        A[j, ix] = (V[j] - x[ix-1])/(x[ix] - x[ix-1])
        A[j, ix-1] = (1 - (V[j] - x[ix-1])/(x[ix] - x[ix-1]));
    A[:, M] = (dV + ff*r_extra*V).T # take the transpose cuz python is dumb

    # Similar to above, but used to simulate data and invert for E(s|y)
    # for initial condition
    A1 = np.zeros((N, M + 1)).astype(complex)
    for j in range(N):
        # Note: Again, ix is one less than it is in the Matlab code
        ix = math.floor((V[j] + V0)/dx)+1
        ix = min(ix, x.size - 1)
        ix = max(ix, 1)
        A1[j, ix] = V[j]*(V[j] - x[ix-1])/(x[ix] - x[ix-1])
        A1[j, ix-1] = V[j]*(1 - (V[j] - x[ix-1])/(x[ix] - x[ix-1]))
    A1[:, M] = (dV + ff*r_extra*V).T # transpose again here

    # A rough guess for the initial condition is a bunch of math stuff
    # This is an approximation of the Laplacian
    # Note: have to do dumb things with x because it is a column vector
    Lap = (-np.diag(np.power(x.T[0][:-1], 0), -1) - np.diag(np.power(x.T[0][:-1], 0), 1) 
    	+ 2*np.diag(np.power(x.T[0], 0), 0))/dx/dx
    Lap[0, 0] = 1/dx/dx
    Lap[-1, -1] = 1/dx/dx

    P0 = np.zeros((M+1, M+1)).astype(complex)
    P0[:M, :M] = (1/sigma/sigma)*(np.eye(M) + np.matmul(Lap, Lap))
    P0[M, M] = 1/sigc/sigc

    Sigma = np.linalg.inv(np.matmul(A1.T, A1)/gam/gam + P0).astype(complex)
    m = np.matmul(Sigma, np.matmul(A1.T, i_meas)/gam/gam)

    # Tuning parameters
    Mint = 1000
    Mb = 100
    r = 1.1
    beta = 1
    P = np.zeros((M+2, Ns))

    # Define prior
    SS = np.matmul(spla.sqrtm(Sigma), np.random.randn(M+1, Ns)) + np.tile(m, (1, Ns))
    print("SS's shape is {}".format(SS.shape))
    RR = np.concatenate((np.log(1/np.maximum(SS[:M, :], np.full((SS[:M, :]).shape, 
    	np.finfo(float).eps))), SS[M, :][np.newaxis]), axis=0)
    mr = 1/Ns*np.sum(RR, axis=1)[np.newaxis].T
    SP = 1/Ns*np.matmul(RR - np.tile(mr, (1, Ns)), (RR - np.tile(mr, (1, Ns))).T)
    amp = 100
    C0 = amp**2 * SP[:M, :M]
    SR = spla.sqrtm(C0)

    # Initial guess for Sigma from Eq 1.8 in the notes
    S = np.concatenate((np.concatenate((SR, np.zeros((2, M))), axis=0), 
    	np.concatenate((np.zeros((M, 2)), amp*(1e-2)*np.eye(2)), axis=0)), axis=1).astype(complex)
    S2 = np.matmul(S, S.T)
    S1 = np.zeros((M+2, 1)).astype(complex)
    mm = np.append(mr, r_extra)[np.newaxis].T
    ppp = mm
    P0 = np.linalg.inv(C0)

    

def setup_initial(V, i_meas, f=200, V0=6):
    """
    Sets up a lot of constants, vectors, and matrices for the computations
    used in the adaptive metropolis algorithm.
    
    Parameters
    ----------
    V: dtype
        what it is, general description
    i_meas:
    f: (optional) dtype
        description
    V0: (optional)
        
    Returns
    -------
    Tuple for adaptive metropolis (ppp, A, V, dV, i_meas, gam, P0, 
        mm, Rmax, Rmin, Ns, beta, r, S, S1, S2, M, Mb, Mint, A1, m, x)
        
    Notes
    -----
    #add any notes needed

    """
    Ns = int(1e4)
    Rmax = 100
    Rmin = -1e-6

    nt = 1000;
    nx = 32;

    r_extra = 0.110
    ff = 1e0

    gam = 0.01
    sigc = 10
    sigma = 1

    #f = 200
    tmax = 1/f/2
    t = np.linspace(0, tmax, V.size)
    dt = t[1] - t[0]
    dV = np.diff(V)/dt
    dV = np.append(dV, dV[dV.size-1])
    N = V.size 
    dx = 0.1
    #V0 = max(V)
    x = np.arange(-V0, V0+dx, dx)[np.newaxis].T
    M = x.size

    # Change V and dV into column vectors for computations
    # Note: V has to be a row vector for np.diff(V) and
    # max(V) to work properly
    dV = dV[np.newaxis].T
    V = V[np.newaxis].T
    i_meas = i_meas[np.newaxis].T

    # Build A : the forward map
    A = np.zeros((N, M + 1)).astype(complex)
    for j in range(N):
        # Note: ix will be used to index into arrays, so it is one less
        # than the ix used in the Matlab code
        ix = math.floor((V[j] + V0)/dx) + 1
        ix = min(ix, x.size - 1)
        ix = max(ix, 1)
        A[j, ix] = (V[j] - x[ix-1])/(x[ix] - x[ix-1])
        A[j, ix-1] = (1 - (V[j] - x[ix-1])/(x[ix] - x[ix-1]));
    A[:, M] = (dV + ff*r_extra*V).T # take the transpose cuz python is dumb

    # Similar to above, but used to simulate data and invert for E(s|y)
    # for initial condition
    A1 = np.zeros((N, M + 1)).astype(complex)
    for j in range(N):
        # Note: Again, ix is one less than it is in the Matlab code
        ix = math.floor((V[j] + V0)/dx)+1
        ix = min(ix, x.size - 1)
        ix = max(ix, 1)
        A1[j, ix] = V[j]*(V[j] - x[ix-1])/(x[ix] - x[ix-1])
        A1[j, ix-1] = V[j]*(1 - (V[j] - x[ix-1])/(x[ix] - x[ix-1]))
    A1[:, M] = (dV + ff*r_extra*V).T # transpose again here

    # A rough guess for the initial condition is a bunch of math stuff
    # This is an approximation of the Laplacian
    # Note: have to do dumb things with x because it is a column vector
    Lap = (-np.diag(np.power(x.T[0][:-1], 0), -1) - np.diag(np.power(x.T[0][:-1], 0), 1) 
    	+ 2*np.diag(np.power(x.T[0], 0), 0))/dx/dx
    Lap[0, 0] = 1/dx/dx
    Lap[-1, -1] = 1/dx/dx

    P0 = np.zeros((M+1, M+1)).astype(complex)
    P0[:M, :M] = (1/sigma/sigma)*(np.eye(M) + np.matmul(Lap, Lap))
    P0[M, M] = 1/sigc/sigc

    Sigma = np.linalg.inv(np.matmul(A1.T, A1)/gam/gam + P0).astype(complex)
    m = np.matmul(Sigma, np.matmul(A1.T, i_meas)/gam/gam)

    # Tuning parameters
    Mint = 1000
    Mb = 100
    r = 1.1
    beta = 1
    P = np.zeros((M+2, Ns))

    # Define prior
    SS = np.matmul(spla.sqrtm(Sigma), np.random.randn(M+1, Ns)) + np.tile(m, (1, Ns))
    print("SS's shape is {}".format(SS.shape))
    RR = np.concatenate((np.log(1/np.maximum(SS[:M, :], np.full((SS[:M, :]).shape, 
    	np.finfo(float).eps))), SS[M, :][np.newaxis]), axis=0)
    mr = 1/Ns*np.sum(RR, axis=1)[np.newaxis].T
    SP = 1/Ns*np.matmul(RR - np.tile(mr, (1, Ns)), (RR - np.tile(mr, (1, Ns))).T)
    amp = 100
    C0 = amp**2 * SP[:M, :M]
    SR = spla.sqrtm(C0)

    # Initial guess for Sigma from Eq 1.8 in the notes
    S = np.concatenate((np.concatenate((SR, np.zeros((2, M))), axis=0), 
    	np.concatenate((np.zeros((M, 2)), amp*(1e-2)*np.eye(2)), axis=0)), axis=1).astype(complex)
    S2 = np.matmul(S, S.T)
    S1 = np.zeros((M+2, 1)).astype(complex)
    mm = np.append(mr, r_extra)[np.newaxis].T
    ppp = mm # using p as a variable makes pdb bug out
    P0 = np.linalg.inv(C0)

    return ppp, A, V, dV, i_meas, gam, P0, mm, Rmax, Rmin, Ns, beta, r, S, S1, S2, M, Mb, Mint, A1, m, x 
    #N, t, x (these last three are for simple graphs)

# Takes in the reshaped main USIDatabase and returns V and i_meas data
def get_data_from_reshaped_USID(h5_resh, pixelNumber):
    # Note: h5_resh.file returns h5File. h5_resh_grp does not hold any
    # of the attributes we want for ex_amp and whatnot.
    return [float(v) for v in h5_resh.h5_spec_vals[()][0]][::4], [float(v) for v in h5_resh[pixelNumber, ::4]]

# Takes float lists fullV and fulli_meas and splits them into forward
# and reverse directions. Assumes the fullV is a sine wave.
# Returns Vfor, Ifor, Vrev, Irev as 1D numpy arrays.
def get_forward_and_reverse(pixelData):
    # Parse our input into what we want.
    fullV, fulli_meas = pixelData

    # Since we have a sine wave (which goes up, then down, then up),
    # we want to take the last bit that goes up and move it to the
    # front, such that the data first goes from -6 to 6 V, then
    # from 6 to -6 V.
    lowestInd = len(fullV) - 1
    while(fullV[lowestInd-1] < fullV[lowestInd]):
        lowestInd -= 1

    fullV = fullV[lowestInd:] + fullV[:lowestInd]
    fulli_meas = fulli_meas[lowestInd:] + fulli_meas[:lowestInd]

    # Then we do another walk to get the boundary between the
    # forward and reverse sections.
    forwardEnd = 0
    while(fullV[forwardEnd] < fullV[forwardEnd + 1]):
        forwardEnd += 1

    # Finally, we split the data into forward and reverse pieces
    # and return them
    Vfor = np.array(fullV[:forwardEnd]) #+ 1])
    Ifor = np.array(fulli_meas[:forwardEnd])# + 1])
    Vrev = np.array(fullV[forwardEnd:])
    Irev = np.array(fulli_meas[forwardEnd:])

    return Vfor, Ifor, Vrev, Irev



# This runs through our adaptive metropolis algorithm.
# Takes in the output of setup_initial().
# Returns V, dV, i_meas, Ns, M, S, S1, S1lag, P, A for graphing.
def do_adaptive_metropolis(metroData):
    """
    What function does
    
    Parameters
    ----------
    metroData: dtype
        what it is, general description
        
    Returns
    -------
    Tuple (V, dV, i_meas, Ns, M, S, S1, S1lag, P, A)
    V: general description
    dV: 
    i_meas:
    Ns:
    M:
    S:
    S1:
    S1lag:
    P:
    A:
        
    Notes
    -----
    #add any notes needed

    """
    # Parse the input into what we want
    ppp, A, V, dV, i_meas, gam, P0, mm, Rmax, Rmin, Ns, beta, r, S, S1, S2, M, Mb, Mint = metroData

    # Quick setup for adaptive metropolis.
    i = 0
    j = 0
    nacc = 0 # number of accepted proposals
    P = np.zeros((M+2, Ns)).astype(complex) # an array to hold the sample values
    logpold = _logpo_R1(ppp, A, V, dV, i_meas, gam, P0, mm, Rmax, Rmin, Rmax, Rmin)

    print("Starting Active Metropolis...")
    metStartTime = time.time()
    while i < Ns:
        pppp = ppp + beta*np.matmul(S, np.random.randn(M+2, 1))
        logpnew = _logpo_R1(pppp, A, V, dV, i_meas, gam, P0, mm, Rmax, Rmin, Rmax, Rmin)

        # accept or reject
        # Note: unlike Matlab's rand, which is a uniformly distributed selection from (0, 1),
        # Python's np.random.rand is a uniformly distributed selection from [0, 1), and can
        # therefore be 0 (by a very small probability, but still). Thus, a quick filter must
        # be made to prevent us from trying to evaluate log(0).
        randBoi = np.random.rand()
        while randBoi == 0:
            randBoi = np.random.rand()
        if np.log(randBoi) < logpold - logpnew:
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

        # Save all samples because why not (this is not necessary)
        P[:, i] = ppp.T[0]

        # proposal covariance adaptation
        if (i+1) % Mint == 0:
            if (j+1) == Mint:
                S1lag = np.sum(P[:, :j] * P[:, 1:j+1], axis=1)[np.newaxis].T.astype(complex) / j
            else:
                S1lag = (j - Mint)/j*S1lag + np.sum(P[:, j-Mint:j] * P[:, j-Mint+1:j+1], axis=1)[np.newaxis].T / j

            # Update meani based on Mint batch
            S1 = (j+1-Mint)/(j+1)*S1 + np.sum(P[:, i-Mint+1:i+1], axis=1)[np.newaxis].T/(j+1)
            # Update Sigma based on Mint batch
            S2 = (j+1-Mint)/(j+1)*S2 + np.matmul(P[:, i-Mint+1:i+1], P[:, i-Mint+1:i+1].T)/(j+1)
            #print("P's shape is {}".format(P.shape))
            # Approximate L such that L*L' = Sigma, where the second term is a
            # decaying regularization
            S = np.linalg.cholesky(S2 - np.matmul(S1, S1.T) + (1e-3)*np.eye(M+2)/(j+1))

            print("i = {}".format(i+1))

        i += 1
        j += 1

    print("Finished Adaptive Metropolis!\nAdaptive Metropolis took {}.".format(time.time() - metStartTime))

    # Return statement for simple graphs
    return V, dV, i_meas, Ns, M, S, S1, S1lag, P, A

# Plots either forward or reverse graphs. Used mainly for testing.
# Takes in the output of doAdaptiveMetropolis(). Returns nothing.
def _plot_simple_graph(graphData, metroBits):
    """
    Helper function for plotting a simple graph
    
    Parameters
    ----------
    graph_data: Tuple

    metro_bits: Tuple
        
    Notes
    -----
    #add any notes needed
    """
    # Parse the input to what we want.
    V, dV, i_meas, Ns, M, S, S1, S1lag, P, A = graphData
    x, N, t = metroBits

    ACL1 = (S1lag - np.square(S1))/np.diag(np.matmul(S, S.T))[np.newaxis].T

    # Mean and variance log resistances
    meani = np.mean(P[:, int(Ns/2):Ns], axis=1)[np.newaxis].T
    mom2 = np.matmul(P[:, int(Ns/2):Ns], P[:, int(Ns/2):Ns].T)/(Ns/2)
    var = mom2 - np.matmul(meani, meani.T).astype(complex)

    # Mean and variance of resistance
    meanr = np.matmul(np.exp(P[:M, int(Ns/2):Ns]), np.ones((int(Ns/2), 1))) / (Ns/2)
    mom2r = np.matmul(np.exp(P[:M, int(Ns/2):Ns]), np.exp(P[:M, int(Ns/2):Ns]).T) / (Ns/2)
    varr = mom2r - np.matmul(meanr, meanr.T).astype(complex)

    # Mean of current
    meanI = np.tile(V, (1, int(Ns/2))) * np.matmul(np.exp(np.matmul(-A[:, :M], P[:M, int(Ns/2):Ns])), np.ones((int(Ns/2), 1))) / (Ns/2) + \
            np.matmul(np.tile(P[M, int(Ns/2):Ns], (N, 1))*(np.tile(dV, (1, int(Ns/2))) + P[M+1, int(Ns/2):Ns]*np.tile(V, (1, int(Ns/2)))), \
                      np.ones((int(Ns/2), 1))) / (Ns/2)

    # Calculate the standard deviation of R and U to avoid redundant calculations
    sigmaR = np.sqrt(np.diag(varr[:M, :M]))[np.newaxis].T
    sigmaU = np.sqrt(np.diag(var[:M, :M]))[np.newaxis].T

    # Now graph out our results.
    fig101 = plt.figure()
    plt.plot(V, i_meas, "ro", mfc="none")
    plt.plot(V, meanI, "gx")
    nn = min(700, max(t.shape))
    plt.plot(V[:nn], V[:nn]*np.exp(np.matmul(-A[:nn, :M], meani[:M])) + \
             np.matmul(dV[:nn] + meani[-1]*V[:nn], meani[M])[np.newaxis].T, "go", mfc="none")
    plt.plot(V[:nn], V[:nn]*np.exp(np.matmul(-A[:nn, :M], meani[:M]+sigmaU)) + \
             (dV[:nn] + (meani[-1] + np.sqrt(var[M+1][0])) * V[:nn]) * (meani[M] + np.sqrt(var[M][0])), "rx")
    plt.plot(V[:nn], V[:nn]*np.exp(np.matmul(-A[:nn, :M], meani[:M]-sigmaU)) + \
             (dV[:nn] + (meani[-1] - np.sqrt(var[M+1][0])) * V[:nn]) * (meani[M] - np.sqrt(var[M][0])), "ko-", mfc="none")
    fig101.show()

    fig1 = plt.figure()
    plt.plot(x, meanr[:M], label="ER")
    plt.plot(x, meanr[:M] + sigmaR, '--', label="ER+\u03C3_R")
    plt.plot(x, meanr[:M] - sigmaR, '--', label="ER-\u03C3_R")
    plt.plot(x, np.exp(meani[:M]), label="exp(AEu)")
    plt.plot(x, np.exp(meani[:M] + sigmaU), '--', label="exp\{A(Eu+\u03C3_u)\}")
    plt.plot(x, np.exp(meani[:M] - sigmaU), '--', label="exp\{A(Eu-\u03C3_u)\}")
    plt.legend()
    fig1.show()

    # Wait here to allow the user to view the graphs
    input("Press <Enter> to continue...")

 
def plot_simple_graphs(verbose=False):
    """
    Main function for plotting a simple graph
    
    Parameters
    ----------
    verbose: (optional) Boolean
        set True for verbose
        for the purpose of debugging
        
    Notes
    -----
    #add any notes needed
    """

    #pixelData = getDataFromTxt("i_meas_220.txt")
    h5Path = "pzt_nanocap_6_split_bayesian_compensation_R_correction.h5"
    pixelData = get_data_from_USID(h5Path, 32444)

    Vfor, Ifor, Vrev, Irev = get_forward_and_reverse(pixelData)

    # Do the forward direction
    if verbose: print("Going in the forward direction.")
    forwardMetroData = setup_initial(Vfor, Ifor)
    forwardGraphData = do_adaptive_metropolis(forwardMetroData[:-5])
    plot_simple_graphs(forwardGraphData, forwardMetroData[-3:])

    # Do the reverse direction
    if verbose: print("Going in the reverse direction.")
    reverseMetroData = setup_initial(Vrev, Irev)
    reverseGraphData = do_adaptive_metropolis(reverseMetroData[:-5])
    plot_simple_graphs(reverseGraphData, forwardMetroData[-3:])

    if verbose: print("Simple graph plotting complete.")

