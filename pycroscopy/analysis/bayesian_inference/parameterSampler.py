# This script runs through a few values for M and Ns and times
# how long it takes to process a single pixel

import h5py
import io
import time
from bayesian_inference import AdaptiveBayesianInference
from matplotlib import pyplot as pyplot
import pyUSID as pyUSID
import numpy as np 

h5_path = r"C:\Users\Administrator\Dropbox\GIv Bayesian June 2019\pzt_nanocap_6_split_bayesian_compensation_R_correction (Alvin Tan's conflicted copy 2019-06-25).h5"

mVals = [25, 75, 125, 175]
NsVals = [1e7, 5e7, 1e8, 5e8, 1e9]

with h5py.File(h5_path, mode='r+') as h5_f:

    h5_grp = h5_f['Measurement_000/Channel_000']
    f = usid.hdf_utils.get_attr(h5_grp, 'excitation_frequency_[Hz]')
    V0 = usid.hdf_utils.get_attr(h5_grp, 'excitation_amplitude_[V]')

    h5_resh = h5_f['Measurement_000/Channel_000/Raw_Data-FFT_Filtering_000/Filtered_Data-Reshape_000/Reshaped_Data']

    for i in range(len(mVals)):
        for j in range(len(NsVals)):
            abi = AdaptiveBayesianInference(h5_resh, f=f, V0=V0, Ns=NsVals[j], M=mVals[i])

            startTime = time.time()
            pix_ind, figBoi = abi.test()
            totalTime = time.time() - startTime

            outputFile = open("parameterTests/timings.txt", "a")
            outF.write("Ns = {} and M = {} took {} seconds on pixel {}\n".format(NsVals[j], mVals[i], totalTime, pix_ind))
            outputFile.close()

            figBoi.set_size_inches(40, 10)

            figFor.savefig("parameterTests/Pixel{}Ns{}M{}.png".format(pix_ind, NsVals[j], mVals[i]))










