# Oak Ridge National Lab
# Center for Nanophase Materials Sciences
# 7/1/2019
# written by Alvin Tan

import os
import time
import math
import numpy as np
import scipy.linalg as spla
from matplotlib import pyplot as plt

# import some more libraries for USID database use
import h5py
import pycroscopy as px 
import pyUSID as usid 

# Reads in V and Imeas data from file fileName. Used mainly for testing.
# Returns fullV and fullImeas as float lists.
def getDataFromTxt(fileName):
    # Load in the data
    with open(fileName) as dataFile:
        # Reads in the units of the lines
        firstLine = dataFile.readline().split()
        # Reads in the data in a Nx2 matrix
        data = np.array([line.split() for line in dataFile])

    # Parse down the data a bit to speed up calculations
    fullV = [float(v) for v in data[::4, 0]]
    fullImeas = [float(i) for i in data[::4, 1]]

    return fullV, fullImeas

# Reads in V and Imeas data from a USIDataset.
def getDataFromUSID(fileName, pixelNumber):
    # Open the file
    h5File = h5py.File(fileName)

    # Grab the main dataset
    h5Group = h5File["Measurement_000/Channel_000"]
    h5Main = usid.USIDataset(h5Group["Raw_Data"])

    # Get some more information out of this dataset
    samp_rate = usid.hdf_utils.get_attr(h5Group, 'IO_samp_rate_[Hz]')
    ex_freq = usid.hdf_utils.get_attr(h5Group, 'excitation_frequency_[Hz]')

    # Getting ancillary information and other parameters
    h5_spec_vals = h5Main.h5_spec_vals

    # Excitation waveform for a single line / row of data
    excit_wfm = h5_spec_vals[()]

    # We expect each pixel to have a single period of the sinusoidal excitation
    # Calculating the excitation waveform for a single pixel
    pts_per_cycle = int(np.round(samp_rate/ex_freq))
    single_AO = excit_wfm[0, :pts_per_cycle]

    # Get excitation amplitude
    ex_amp = usid.hdf_utils.get_attr(h5Group, 'excitation_amplitude_[V]')

    # Assume that we have already filterd and reshaped the data
    # Now load in a filtered and reshaped data
    h5_filt_grp = usid.hdf_utils.find_results_groups(h5Main, "FFT_Filtering")[-1]
    h5_filt = h5_filt_grp["Filtered_Data"]
    h5_resh_grp = usid.hdf_utils.find_results_groups(h5_filt, "Reshape")[-1]
    h5_resh = usid.USIDataset(h5_resh_grp["Reshaped_Data"])

    breakpoint()

    # Just return the voltage and response for now
    return [float(v) for v in single_AO[::4]], [float(v) for v in h5_resh[pixelNumber, ::4]]

# Takes float lists fullV and fullImeas and splits them into forward
# and reverse directions. Assumes the fullV is a sine wave.
# Returns Vfor, Ifor, Vrev, Irev as 1D numpy arrays.
def getForwardAndReverseData(pixelData):
    # Parse our input into what we want.
    fullV, fullImeas = pixelData

    # Since we have a sine wave (which goes up, then down, then up),
    # we want to take the last bit that goes up and move it to the
    # front, such that the data first goes from -6 to 6 V, then
    # from 6 to -6 V.
    lowestInd = len(fullV) - 1
    while(fullV[lowestInd-1] < fullV[lowestInd]):
        lowestInd -= 1

    fullV = fullV[lowestInd:] + fullV[:lowestInd]
    fullImeas = fullImeas[lowestInd:] + fullImeas[:lowestInd]

    # Then we do another walk to get the boundary between the
    # forward and reverse sections.
    forwardEnd = 0
    while(fullV[forwardEnd] < fullV[forwardEnd + 1]):
        forwardEnd += 1

    # Finally, we split the data into forward and reverse pieces
    # and return them
    Vfor = np.array(fullV[:forwardEnd]) #+ 1])
    Ifor = np.array(fullImeas[:forwardEnd])# + 1])
    Vrev = np.array(fullV[forwardEnd:])
    Irev = np.array(fullImeas[forwardEnd:])

    return Vfor, Ifor, Vrev, Irev

# Sets up a lot of constants, vectors, and matrices for the computations
# used in the adaptive metropolis algorithm.
# Returns ppp, A, V, dV, Imeas, gam, P0, mm, Rmax, Rmin for adaptive metropolis
def setUpConstantsAndInitialConditions(V, Imeas, f=200, V0=6):
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
    Imeas = Imeas[np.newaxis].T

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
    Lap = (-np.diag(np.power(x.T[0][:-1], 0), -1) - np.diag(np.power(x.T[0][:-1], 0), 1) + 2*np.diag(np.power(x.T[0], 0), 0))/dx/dx
    Lap[0, 0] = 1/dx/dx
    Lap[-1, -1] = 1/dx/dx

    P0 = np.zeros((M+1, M+1)).astype(complex)
    P0[:M, :M] = (1/sigma/sigma)*(np.eye(M) + np.matmul(Lap, Lap))
    P0[M, M] = 1/sigc/sigc

    Sigma = np.linalg.inv(np.matmul(A1.T, A1)/gam/gam + P0).astype(complex)
    m = np.matmul(Sigma, np.matmul(A1.T, Imeas)/gam/gam)

    # Tuning parameters
    Mint = 1000
    Mb = 100
    r = 1.1
    beta = 1
    P = np.zeros((M+2, Ns))

    # Define prior
    SS = np.matmul(spla.sqrtm(Sigma), np.random.randn(M+1, Ns)) + np.tile(m, (1, Ns))
    print("SS's shape is {}".format(SS.shape))
    RR = np.concatenate((np.log(1/np.maximum(SS[:M, :], np.full((SS[:M, :]).shape, np.finfo(float).eps))), SS[M, :][np.newaxis]), axis=0)
    mr = 1/Ns*np.sum(RR, axis=1)[np.newaxis].T
    SP = 1/Ns*np.matmul(RR - np.tile(mr, (1, Ns)), (RR - np.tile(mr, (1, Ns))).T)
    amp = 100
    C0 = amp**2 * SP[:M, :M]
    SR = spla.sqrtm(C0)

    # Initial guess for Sigma from Eq 1.8 in the notes
    S = np.concatenate((np.concatenate((SR, np.zeros((2, M))), axis=0), np.concatenate((np.zeros((M, 2)), amp*(1e-2)*np.eye(2)), axis=0)), axis=1).astype(complex)
    S2 = np.matmul(S, S.T)
    S1 = np.zeros((M+2, 1)).astype(complex)
    mm = np.append(mr, r_extra)[np.newaxis].T
    ppp = mm # using p as a variable makes pdb bug out
    P0 = np.linalg.inv(C0)

    return ppp, A, V, dV, Imeas, gam, P0, mm, Rmax, Rmin, Ns, beta, r, S, S1, S2, M, Mb, Mint, A1, m, x #N, t, x (these last three are for simple graphs)

# Does math stuff and returns a number relevant to some probability distribution.
# Used only in doAdaptiveMetropolis()
def logpoR1(pp, A, V, dV, y, gam, P0, mm, Rmax, Rmin, Cmax, Cmin):
    if pp[-1] > Rmax or pp[-1] < Rmin:
        return np.inf
    if pp[-2] > Cmax or pp[-2] < Cmin:
        return np.inf
    
    out = np.linalg.norm(V*np.exp(np.matmul(-A[:, :-1], pp[:-2])) + \
                         pp[-2][0] * (dV + pp[-1][0]*V) - y)**2/2/gam/gam + \
          np.matmul(np.matmul((pp[:-2]-mm[:-2]).T, P0), pp[:-2]-mm[:-2])/2

    return out

# This runs through our adaptive metropolis algorithm.
# Takes in the output of setUpConstantsAndInitialConditions().
# Returns V, dV, Imeas, Ns, M, S, S1, S1lag, P, A for graphing.
def doAdaptiveMetropolis(metroData):
    # Parse the input into what we want
    ppp, A, V, dV, Imeas, gam, P0, mm, Rmax, Rmin, Ns, beta, r, S, S1, S2, M, Mb, Mint = metroData

    # Quick setup for adaptive metropolis.
    i = 0
    j = 0
    nacc = 0 # number of accepted proposals
    P = np.zeros((M+2, Ns)).astype(complex) # an array to hold the sample values
    logpold = logpoR1(ppp, A, V, dV, Imeas, gam, P0, mm, Rmax, Rmin, Rmax, Rmin)

    print("Starting Active Metropolis...")
    metStartTime = time.time()
    while i < Ns:
        pppp = ppp + beta*np.matmul(S, np.random.randn(M+2, 1))
        logpnew = logpoR1(pppp, A, V, dV, Imeas, gam, P0, mm, Rmax, Rmin, Rmax, Rmin)

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
    return V, dV, Imeas, Ns, M, S, S1, S1lag, P, A

# Plots either forward or reverse graphs. Used mainly for testing.
# Takes in the output of doAdaptiveMetropolis(). Returns nothing.
def plotSimpleGraphs(graphData, metroBits):
    # Parse the input to what we want.
    V, dV, Imeas, Ns, M, S, S1, S1lag, P, A = graphData
    N, t, x = metroBits

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
    plt.plot(V, Imeas, "ro", mfc="none")
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

# Plots the 3 graphs side by side
def plotPycroscopyGraphs(fullV, fullImeas, meanr, varr, x, Irec, capacitance):
    #capacitance = m[-1] / 2

    #dt = 1.0 / (ex_freq * single_AO.size)
    dt = 0.0001 # temporary substitution
    #r_extra = 0.110 # also temporary substitution
    r_extra = .110
    dvdt = np.diff(fullV) / dt # Use fullV instead of single_AO
    dvdt = np.append(dvdt, dvdt[-1])
    point_i_cap = capacitance * dvdt
    point_i_extra = r_extra * 2 * capacitance * fullV
    point_i_corr = fullImeas - point_i_cap - point_i_extra

    #point_i_corr *= 0 # Cuz this bad boi isn't working >:(

    # Mean and variance of resistance
    #meanr = np.matmul(np.exp(P[:M, int(Ns/2):Ns]), np.ones((int(Ns/2), 1))) / (Ns/2)
    #mom2r = np.matmul(np.exp(P[:M, int(Ns/2):Ns]), np.exp(P[:M, int(Ns/2):Ns]).T) / (Ns/2)
    #varr = mom2r - np.matmul(meanr, meanr.T).astype(complex)

    #Irec = np.matmul(A1, m)

    #breakpoint()

    fig = px.analysis.utils.giv_utils.plot_bayesian_results(fullV, fullImeas, point_i_corr, x.T[0], meanr.T[0], varr, Irec.T[0])
    fig.show()

    # Wait here to allow the user to view the graphs
    input("Press <Enter> to continue...")

# Here is our main function if we want to do a simple graph
def simpleMain():
    #pixelData = getDataFromTxt("imeas_220.txt")
    h5Path = "pzt_nanocap_6_split_bayesian_compensation_R_correction.h5"
    pixelData = getDataFromUSID(h5Path, 220)

    Vfor, Ifor, Vrev, Irev = getForwardAndReverseData(pixelData)

    # Do the forward direction
    print("Going in the forward direction.")
    forwardMetroData = setUpConstantsAndInitialConditions(Vfor, Ifor)
    forwardGraphData = doAdaptiveMetropolis(forwardMetroData[:-3])
    plotSimpleGraphs(forwardGraphData, forwardMetroData[-3:])

    # Do the reverse direction
    print("Going in the reverse direction.")
    reverseMetroData = setUpConstantsAndInitialConditions(Vrev, Irev)
    reverseGraphData = doAdaptiveMetropolis(reverseMetroData[:-3])
    plotSimpleGraphs(reverseGraphData, forwardMetroData[-3:])

    # That is all
    print("Done.")

# Here is our main function if we want to do a USID graph
def USIDMain():
    pixelData = getDataFromTxt("imeas_220.txt")
    #h5Path = "pzt_nanocap_6_split_bayesian_compensation_R_correction.h5"
    #pixelData = getDataFromUSID(h5Path, 32444)

    Vfor, Ifor, Vrev, Irev = getForwardAndReverseData(pixelData)

    # Do the forward direction
    print("Going in the forward direction.")
    forwardMetroData = setUpConstantsAndInitialConditions(Vfor, Ifor)
    forwardGraphData = doAdaptiveMetropolis(forwardMetroData[:-3])
    #plotSimpleGraphs(forwardGraphData, forwardMetroData[-3:])

    # Do the reverse direction
    print("Going in the reverse direction.")
    reverseMetroData = setUpConstantsAndInitialConditions(Vrev, Irev)
    reverseGraphData = doAdaptiveMetropolis(reverseMetroData[:-3])
    #plotSimpleGraphs(reverseGraphData, forwardMetroData[-3:])

    #breakpoint()

    fullV = np.concatenate((Vfor, Vrev))
    fullImeas = np.concatenate((Ifor, Irev))
    NsFor = forwardGraphData[3]
    MFor = forwardGraphData[4]
    PFor = forwardGraphData[8]
    NsRev = reverseGraphData[3]
    MRev = reverseGraphData[4]
    PRev = reverseGraphData[8]

    # Mean and variance of resistance
    meanrFor = np.matmul(np.exp(PFor[:MFor, int(NsFor/2):NsFor]), np.ones((int(NsFor/2), 1))) / (NsFor/2)
    mom2rFor = np.matmul(np.exp(PFor[:MFor, int(NsFor/2):NsFor]), np.exp(PFor[:MFor, int(NsFor/2):NsFor]).T) / (NsFor/2)
    varrFor = mom2rFor - np.matmul(meanrFor, meanrFor.T).astype(complex)
    meanrRev = np.matmul(np.exp(PRev[:MRev, int(NsRev/2):NsRev]), np.ones((int(NsRev/2), 1))) / (NsRev/2)
    mom2rRev = np.matmul(np.exp(PRev[:MRev, int(NsRev/2):NsRev]), np.exp(PRev[:MRev, int(NsRev/2):NsRev]).T) / (NsRev/2)
    varrRev = mom2rRev - np.matmul(meanrRev, meanrRev.T).astype(complex)

    meanr = np.concatenate((meanrFor, meanrRev))
    varr = np.concatenate((np.diag(varrFor), np.diag(varrRev)))

    x = np.concatenate((forwardMetroData[-1], -1*reverseMetroData[-1]))
    IrecFor = np.matmul(forwardMetroData[-3], forwardMetroData[-2])
    IrecRev = np.matmul(reverseMetroData[-3], reverseMetroData[-2])
    Irec = np.concatenate((IrecFor, IrecRev))
    capacitance = (forwardMetroData[-2][-1]+reverseMetroData[-2][-1])[0]/2
    
    #breakpoint()

    plotPycroscopyGraphs(fullV, fullImeas, meanr, varr, x, Irec, capacitance)
    #plotPycroscopyGraphs(np.array(pixelData[0]), np.array(pixelData[1]), meanr, varr, x, Irec, capacitance)

    #breakpoint()

    # That is all
    print("Done.")

# Here is our main function. It brings all the other functions together.
if __name__ == '__main__':
    # Either graph simple graphs or USID graphs
    # Note: the return values for setUpConstraintsAndInitialConditions()
    # have to be changed to switch between simple and USID graphs
    USIDMain()