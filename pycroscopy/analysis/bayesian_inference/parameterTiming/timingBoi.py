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

h5_path = r"/lustre/or-hydra/cades-birthright/29t/pzt_nanocap_6_split_bayesian_compensation_R_correction (Alvin Tan's conflicted copy 2019-06-25).h5"

M = 100
Ns = int(7e7)

with h5py.File(h5_path, mode='r+') as h5_f:

    h5_grp = h5_f['Measurement_000/Channel_000']
    f = usid.hdf_utils.get_attr(h5_grp, 'excitation_frequency_[Hz]')
    V0 = usid.hdf_utils.get_attr(h5_grp, 'excitation_amplitude_[V]')

    h5_resh = h5_f['Measurement_000/Channel_000/Raw_Data-FFT_Filtering_000/Filtered_Data-Reshape_000/Reshaped_Data']

    # initialize the Process with the desired parameters
    abi = AdaptiveBayesianInference(h5_resh, f=f, V0=V0, Ns=Ns, M=M)

    # record the time it takes to compute one random pixel
    startTime = time.time()
    pixResultBoi = abi.test()
    totalTime = time.time() - startTime

# record the time in a file pixel by pixel so we don't have to wait for all
# the pixels to finish (in case something crashes haha)
outputFile = open("timings.txt", "a")
outputFile.write("{}\n".format(totalTime))
outputFile.close()







