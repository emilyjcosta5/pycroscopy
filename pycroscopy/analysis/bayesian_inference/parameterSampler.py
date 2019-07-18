# This script runs through a few values for M and Ns and times
# how long it takes to process a single pixel

import h5py
import io
import time
from bayesian_inference import AdaptiveBayesianInference
from matplotlib import pyplot as plt
import pyUSID as pyUSID
import numpy as np 
import pyUSID as usid

h5_path = r"C:\Users\Administrator\Dropbox\GIv Bayesian June 2019\pzt_nanocap_6_split_bayesian_compensation_R_correction (Alvin Tan's conflicted copy 2019-06-25).h5"

mVals = [25, 75, 125, 175]
NsVals = [int(1e7), int(5e7), int(1e8), int(5e8), int(1e9)]
timingResults = np.zeros((len(NsVals), len(mVals)))

with h5py.File(h5_path, mode='r+') as h5_f:

    h5_grp = h5_f['Measurement_000/Channel_000']
    f = usid.hdf_utils.get_attr(h5_grp, 'excitation_frequency_[Hz]')
    V0 = usid.hdf_utils.get_attr(h5_grp, 'excitation_amplitude_[V]')

    h5_resh = h5_f['Measurement_000/Channel_000/Raw_Data-FFT_Filtering_000/Filtered_Data-Reshape_000/Reshaped_Data']

    for i in range(len(NsVals)):
        for j in range(len(mVals)):
            print("Starting run with Ns = {} and M = {}".format(NsVals[i], mVals[j]))

            # initialize the Process with the desired parameters
            abi = AdaptiveBayesianInference(h5_resh, f=f, V0=V0, Ns=NsVals[i], M=mVals[j])

            # record the time it takes to compute one random pixel
            startTime = time.time()
            pix_ind, figBoi = abi.test()
            totalTime = time.time() - startTime

            # store the time for graphing later
            timingResults[i][j] = totalTime

            # record the time in a file pixel by pixel so we don't have to wait for all
            # the pixels to finish (in case something crashes haha)
            outputFile = open("parameterTests/timings.txt", "a")
            outputFile.write("Ns = {} and M = {} took {} seconds on pixel {}\n".format(NsVals[i], mVals[j], totalTime, pix_ind))
            outputFile.close()

            # size and save the resulting graph for visual comparisons
            figBoi.set_size_inches(40, 10)
            figBoi.savefig("parameterTests/Pixel{}Ns{}M{}.png".format(pix_ind, NsVals[i], mVals[j]))

# graph our timing results to see trends
# note pyplot plots column by column
sweepM4eachNs = plt.figure()
for i in range(len(NsVals)):
    plt.plot(mVals, timingResults[i], label="Ns = {}".format(NsVals[i]))
plt.ylabel("Time taken (sec)", fontsize=15)
plt.xlabel("M", fontsize=15)
plt.legend()
sweepM4eachNs.set_size_inches(10, 10)
sweepM4eachNs.savefig("parameterTests/sweepM4eachNs.png")

sweepNs4eachM = plt.figure()
for j in range(len(mVals)):
    plt.plot(NsVals, timingResults.T[j], label="M = {}".format(mVals[j]))
plt.ylabel("Time taken (sec)", fontsize=15)
plt.xlabel("Ns", fontsize=15)
plt.legend()
sweepNs4eachM.set_size_inches(10, 10)
sweepNs4eachM.savefig("parameterTests/sweepNs4eachM.png")








