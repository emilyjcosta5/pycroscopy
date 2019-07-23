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

mVals = [75, 100, 125, 150, 175]
NsVals = [int(1e8)]
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
            pix_ind, figBois = abi.test(pix_ind=64550, traces=True)
            totalTime = time.time() - startTime

            # record the time in a file pixel by pixel so we don't have to wait for all
            # the pixels to finish (in case something crashes haha)
            outputFile = open("parameterTests2/timings.txt", "a")
            outputFile.write("Ns = {} and M = {} took {} seconds on pixel {} with traces\n".format(NsVals[i], mVals[j], totalTime, pix_ind))
            outputFile.close()

            figBoi, traceBoi = figBois

            # size and save the resulting graph for visual comparisons
            figBoi.set_size_inches(40, 10)
            figBoi.savefig("parameterTests2/Pixel{}Ns{}M{}.png".format(pix_ind, NsVals[i], mVals[j]))
            traceBoi.set_size_inches(50, 20)
            traceBoi.savefig("parameterTests2/Pixel{}Ns{}M{}Traces.png".format(pix_ind, NsVals[i], mVals[j]))






