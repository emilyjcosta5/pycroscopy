# Oak Ridge National Lab
# Center for Nanophase Materials Sciences
# 6/6/2019
# written by Alvin Tan
# based on Matlab code by Kody J.H. Law

import os
import time
import math
import numpy as np
import scipy.linalg as spla
from matplotlib import pyplot as plt

# The logpoR1 function from the Matlab code
def logpoR1(pp, A, V, dV, y, gam, P0, mm, Rmax, Rmin, Cmax, Cmin):#a, bias, dv, y, gam, p0, ff, Rmax=1E4, Rmin=0, Cmax=1E4, Cmin=0):
    '''
    Parameters
    ----------
    pp
    a
    v
    dv
    y
    gam
    p0
    ff

    Returns
    -------
    '''
    #print("pp shape is {}, pp[-1] is {}, pp[-2] is {}".format(pp.shape, pp[-1], pp[-2]))
    if pp[-1] > Rmax or pp[-1] < Rmin:
        return np.inf
    if pp[-2] > Cmax or pp[-2] < Cmin:
        return np.inf
    
    out = np.linalg.norm(V*np.exp(np.matmul(-A[:, :-1], pp[:-2])) + \
                         pp[-2][0] * (dV + pp[-1][0]*V) - y)**2/2/gam/gam + \
          np.matmul(np.matmul((pp[:-2]-mm[:-2]).T, P0), pp[:-2]-mm[:-2])/2

    '''
    out = np.linalg.norm(np.multiply(bias, np.exp(np.matmul(-a[:, :-1], pp[:-2]))) + \
                         pp[-2] * np.subtract(np.add(dV, pp[-1]*bias), y))**2/2/gam/gam + \
                         np.divide(np.matmul((np.subtract(pp[:-2], ff[:-2])).T, np.matmul(p0, np.subtract(pp[:-2], ff[:-2]))), 2)
    '''
    return out

# Grab the start time so we can see how long this takes
startTime = time.time()

nt = 1000;
nx = 32;

r_extra = 0.110
ff = 1e0

gam = 0.01
sigc = 10
sigma = 1

# Load real data
with open("imeas_256.txt") as dataFile:
    # Reads in the units of the lines
    firstLine = dataFile.readline().split()
    # Reads in the data in a Nx2 matrix
    data = np.array([line.split() for line in dataFile])

# We could use the full data for the pixel if we want
#fullV = [float(v) for v in data[:, 0]]
#fullImeas = [float(i) for i in data[:, 1]]

# Or parse it down a little to speed up calculations
#fullV = [float(v) for v in data[::4, 0]]
#fullImeas = [float(i) for i in data[::4, 1]]
# The following gives the same parsed data that the Matlab does
fullV = [float(v) for v in data[1::4, 0]]
fullImeas = [float(i) for i in data[1::4, 1]]

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

################################################################
#        Here, we can choose to do one of three things:        #
################################################################
# 1. Just use the forward direction
V = np.array(fullV[:forwardEnd + 1])
Imeas = np.array(fullImeas[:forwardEnd + 1])
# 2. Just use the reverse direction
#V = np.array(fullV[forwardEnd:])
#Imeas = np.array(fullImeas[forwardEnd:])
# 3. Use the entire cycle
#V = np.array(fullV)
#Imeas = np.array(fullImeas)
################################################################

'''
# Alternatively, in order to check the matrix math we are doing,
# we can load in preprocessed data from Matlab
with open("cleanI.txt") as dataFile:
    Imeas = np.array([float(line) for line in dataFile])

with open("cleanV.txt") as dataFile:
    V = np.array([float(line) for line in dataFile])

# with a quick sanity check of course
#np.savetxt("cleanI.out", Imeas)
#np.savetxt("cleanV.out", V)
'''

f = 200
tmax = 1/f/2
t = np.linspace(0, tmax, V.size)
dt = t[1] - t[0]
dV = np.diff(V)/dt
dV = np.append(dV, dV[dV.size-1])
N = V.size
dx = 0.1
V0 = max(V)
x = np.arange(-V0, V0+dx, dx)[np.newaxis].T
M = x.size

# Change V and dV into column vectors for computations
# Note: V has to be a row vector for np.diff(V) and
# max(V) to work properly
dV = dV[np.newaxis].T
V = V[np.newaxis].T
Imeas = Imeas[np.newaxis].T

'''
# Quick sanity check to make sure we aren't way off track
plt.plot(V)
plt.show()
plt.plot(V, Imeas)
plt.show()
'''

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
#np.savetxt("Ainit.out", A)
#breakpoint()
#input("Pause to inspect A")

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
#breakpoint()
Lap[0, 0] = 1/dx/dx
Lap[-1, -1] = 1/dx/dx

P0 = np.zeros((M+1, M+1)).astype(complex)
P0[:M, :M] = (1/sigma/sigma)*(np.eye(M) + np.matmul(Lap, Lap))
P0[M, M] = 1/sigc/sigc

Sigma = np.linalg.inv(np.matmul(A1.T, A1)/gam/gam + P0).astype(complex)
m = np.matmul(Sigma, np.matmul(A1.T, Imeas)/gam/gam)
#print("m's shape is {}".format(m.shape))

# Another quick sanity check. The values seem to be similar enough
# to the Matlab code
#np.savetxt("Sigma_python.txt", Sigma)
#np.savetxt("m_python.txt", m)

Ns = int(1e4)

# Tuning parameters
Mint = 1000
Mb = 100
r = 1.1
beta = 1
nacc = 0
P = np.zeros((M+2, Ns))

# Define prior
SS = np.matmul(spla.sqrtm(Sigma), np.random.randn(M+1, Ns)) + np.tile(m, (1, Ns))
print("SS's shape is {}".format(SS.shape))
RR = np.concatenate((np.log(1/np.maximum(SS[:M, :], np.full((SS[:M, :]).shape, np.finfo(float).eps))), SS[M, :][np.newaxis]), axis=0)
mr = 1/Ns*np.sum(RR, axis=1)[np.newaxis].T
SP = 1/Ns*np.matmul(RR - np.tile(mr, (1, Ns)), (RR - np.tile(mr, (1, Ns))).T)

P = np.zeros((M+2, Ns)).astype(complex)
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

#breakpoint()

# Quick setup for Adaptive Metropolis
# Note: since i and j are used to index into arrays, I have them starting at
# 0 instead of 1 as in the Matlab code. However, since they are also used in
# general computations, there are some +1's that appear and -1's that disappear.
# It's rather annoying.
i = 0
j = 0
Rmax = 100
Rmin = -1e-6
logpold = logpoR1(ppp, A, V, dV, Imeas, gam, P0, mm, Rmax, Rmin, Rmax, Rmin)

#breakpoint()

# Adaptive Metropolis
print("Starting Active Metropolis...")
metStartTime = time.time()
while i < Ns:
    #print("p shape is {}".format(p.shape))
    pppp = ppp + beta*np.matmul(S, np.random.randn(M+2, 1)) # using pp also makes gdb bug out
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

print("Finished Adaptive Metropolis!\nAdaptive Metropolis took {}.\nTotal time taken so far is {}.".format(time.time() - metStartTime, time.time() - startTime))

#print("S1 is {}\nS1lag is {}\nS is {}\nwith sizes S1: {}, S1lag: {}, and S: {}".format(S1, S1lag, S, S1.shape, S1lag.shape, S.shape))
#exit()

""" The validity of the remainder of this sript has been confirmed
# To confirm the remainder of this script, we will import data that has
# been generated by Matlab. We will need S1lag, S1, S, P, Ns, V, A, M, dV, varr, var, N, Imeas, x
def loadVar(varName):
    with open(varName + ".txt") as dataFile:
        varData = np.array([line.split() for line in dataFile]) #.astype(np.float)

    return varData.astype(np.complex)

#varsToLoad = ["S1lag", "S1", "S", "P", "Ns", "V", "A", "M", "dV", "varr", "var"]
S1lag = loadVar("S1lag")
#print(S1lag)
S1 = loadVar("S1")
#print(S1)
S = loadVar("S")
P = loadVar("P")
Ns = int(1e4) #Ns = loadVar("Ns")
V = loadVar("V")
A = loadVar("A")
M = 121 #M = loadVar("M")
dV = loadVar("dV")
varr = loadVar("varr")
var = loadVar("var")
N = 64
Imeas = loadVar("Imeas")
t = loadVar("t")
x = loadVar("x")
"""

ACL1 = (S1lag - np.square(S1))/np.diag(np.matmul(S, S.T))[np.newaxis].T

# Mean and variance log resistances
meani = np.mean(P[:, int(Ns/2):Ns], axis=1)[np.newaxis].T
mom2 = np.matmul(P[:, int(Ns/2):Ns], P[:, int(Ns/2):Ns].T)/(Ns/2)
var = mom2 - np.matmul(meani, meani.T).astype(complex)
#print("meani shape is {}, mom2 shape is {}, var shape is {}".format(meani.shape, mom2.shape, var.shape))
#print("meani is {}".format(meani))

# Mean and variance of resistance
meanr = np.matmul(np.exp(P[:M, int(Ns/2):Ns]), np.ones((int(Ns/2), 1))) / (Ns/2)
mom2r = np.matmul(np.exp(P[:M, int(Ns/2):Ns]), np.exp(P[:M, int(Ns/2):Ns]).T) / (Ns/2)
varr = mom2r - np.matmul(meanr, meanr.T).astype(complex)
#print("meanr shape is {}, mom2r shape is {}, varr shape is {}".format(meanr.shape, mom2r.shape, varr.shape))
#print("meanr is {}".format(meanr))

#print("S1lag shape: {}\nS shape: {}\nS1 shape: {}\nP shape: {}\nV shape: {}\nA shape: {}\nvar shape: {}\nvarr shape: {}\n".format(S1lag.shape, S.shape, S1.shape, P.shape, V.shape, A.shape, var.shape, varr.shape))
#print("Ns = {} and M = {}".format(Ns, M))

#np.savetxt("S1lag.out", S1lag)
#np.savetxt("S1.out", S1)
#np.savetxt("S.out", S)
#np.savetxt("P.out", P)
#np.savetxt("A.out", A)

#input("Pause to check data")
#print("Continuing...")

# Change a couple vectors into column vectors
# NOTE: Unnecessary at this point, I think...
#dV = dV[np.newaxis].T
#V = V[np.newaxis].T
#print("P shape: {}\ndV shape: {}\nV shape: {}, Ns: {}, Ns/2: {}, M: {}".format(P.shape, dV.shape, V.shape, Ns, Ns/2, M))

# Mean of current
meanI = np.tile(V, (1, int(Ns/2))) * np.matmul(np.exp(np.matmul(-A[:, :M], P[:M, int(Ns/2):Ns])), np.ones((int(Ns/2), 1))) / (Ns/2) + \
        np.matmul(np.tile(P[M, int(Ns/2):Ns], (N, 1))*(np.tile(dV, (1, int(Ns/2))) + P[M+1, int(Ns/2):Ns]*np.tile(V, (1, int(Ns/2)))), \
                  np.ones((int(Ns/2), 1))) / (Ns/2)

# Calculate the standard deviation of R and U to avoid redundant calculations
sigmaR = np.sqrt(np.diag(varr[:M, :M]))[np.newaxis].T
sigmaU = np.sqrt(np.diag(var[:M, :M]))[np.newaxis].T
#print("sigmaR: {}, sigmaU: {}".format(sigmaR, sigmaU))
#print("sigmaR shape is {}, sigmaU shape is {}".format(sigmaR.shape, sigmaU.shape))
#input("Pause to inspect sigmaR and sigmaU")
'''
figSane = plt.figure(420)
plt.plot(sigmaR)
plt.plot(sigmaU)
#plt.plot(dV)
plt.plot(V)
plt.plot(meani)
plt.plot(meanr)
plt.plot(meanI)
plt.show()
'''
#exit()

# Now graph out our results
fig101 = plt.figure(101)
plt.plot(V, Imeas, "ro", mfc="none")
plt.plot(V, meanI, "gx")
nn = min(700, max(t.shape))

#print("suspect shape: {}".format(np.exp(np.matmul(-A[:nn, :M], meani[:M])).shape))
#print("suspect 2 shape: {}".format(np.matmul(dV[:nn] + meani[-1]*V[:nn], meani[M]).shape))

plt.plot(V[:nn], V[:nn]*np.exp(np.matmul(-A[:nn, :M], meani[:M])) + \
         np.matmul(dV[:nn] + meani[-1]*V[:nn], meani[M])[np.newaxis].T, "go", mfc="none")
plt.plot(V[:nn], V[:nn]*np.exp(np.matmul(-A[:nn, :M], meani[:M]+sigmaU)) + \
         (dV[:nn] + (meani[-1] + np.sqrt(var[M+1][0])) * V[:nn]) * (meani[M] + np.sqrt(var[M][0])), "rx")
plt.plot(V[:nn], V[:nn]*np.exp(np.matmul(-A[:nn, :M], meani[:M]-sigmaU)) + \
         (dV[:nn] + (meani[-1] - np.sqrt(var[M+1][0])) * V[:nn]) * (meani[M] - np.sqrt(var[M][0])), "ko-", mfc="none")
fig101.show()

#print("nn: {}\ndV shape: {}\ndv[:nn] shape: {}\nmeani shape: {}\nvar shape: {}\nV[:nn] shape: {}".format(nn, dV.shape, dV[:nn].shape, meani.shape, var.shape, V[:nn].shape))
#print("sigmaU shape: {}, sigmaR shape: {}".format(sigmaU.shape, sigmaR.shape))
#input("Kill me now")

fig1 = plt.figure(1)
plt.plot(x, meanr[:M], label="ER")
plt.plot(x, meanr[:M] + sigmaR, '--', label="ER+\u03C3_R")
plt.plot(x, meanr[:M] - sigmaR, '--', label="ER-\u03C3_R")
plt.plot(x, np.exp(meani[:M]), label="exp(AEu)")
plt.plot(x, np.exp(meani[:M] + sigmaU), '--', label="exp\{A(Eu+\u03C3_u)\}")
plt.plot(x, np.exp(meani[:M] - sigmaU), '--', label="exp\{A(Eu-\u03C3_u)\}")
plt.legend()
fig1.show()

#breakpoint()
#print("Total program took {} seconds to run.".format(time.time() - startTime))
# Wait here to allow the user to view the graphs
input("Press <Enter> to exit program...")
















