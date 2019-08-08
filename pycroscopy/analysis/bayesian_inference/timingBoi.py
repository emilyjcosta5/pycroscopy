# This script runs through a few values for M and Ns and times
# how long it takes to process a single pixel

import os
import subprocess
import sys
import h5py
import io
import time
from matplotlib import pyplot as plt
import numpy as np
import importlib
from shutil import copyfile

# Helper for importing packages
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", "--user", package])

try:
    import pyUSID as usid
except ImportError:
    print("pyUSID not found. Will install with pip.")
    import pip
    install("pyUSID")
    import pyUSID as usid

h5_path = r"/home/cades/pzt_nanocap_6_split_bayesian_compensation_R_correction (Alvin Tan's conflicted copy 2019-06-25).h5"

timingNames = ["parameterTiming", "../NumbaTiming", "../unoptimizedTiming"]
timingM = [100, 125, 125]
#timingNs = [int(7e7), int(5e8), int(5e8)]
timingNs = [int(7e3), int(5e4), int(5e4)]

with h5py.File(h5_path, mode='r+') as h5_f:

    h5_grp = h5_f['Measurement_000/Channel_000']
    f = usid.hdf_utils.get_attr(h5_grp, 'excitation_frequency_[Hz]')
    V0 = usid.hdf_utils.get_attr(h5_grp, 'excitation_amplitude_[V]')

    h5_resh = h5_f['Measurement_000/Channel_000/Raw_Data-FFT_Filtering_000/Filtered_Data-Reshape_000/Reshaped_Data']

    for i in range(len(timingNames)):
        print("Starting running stuff in {}".format(timingNames[i]))
        os.chdir(timingNames[i])
        from bayesian_inference import AdaptiveBayesianInference
        
        # initialize the Process with the desired parameters
        abi = AdaptiveBayesianInference(h5_resh, f=f, V0=V0, Ns=timingNs[i], M=timingM[i])
        
        for j in range(10):
            # record the time it takes to compute one random pixel
            startTime = time.time()
            pixResultBoi = abi.test()
            totalTime = time.time() - startTime

            # record the time in a file pixel by pixel so we don't have to wait for all
            # the pixels to finish (in case something crashes haha)
            outputFile = open("VMtimings.txt", "a")
            outputFile.write("{}\n".format(totalTime))
            outputFile.close()

        del AdaptiveBayesianInference
        print("Finished running stuff in {}".format(timingNames[i]))









