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

from bayesian_inference import AdaptiveBayesianInference

# Make a copy of the dataset so multiple processes can access it at the same time.
h5_og_path = r"/home/29t/pzt_nanocap_6_split_bayesian_compensation_R_correction (Alvin Tan's conflicted copy 2019-06-25).h5"
h5_path = r"/lustre/or-hydra/cades-birthright/29t/NumbaDataBoi{}.h5".format(time.time())
copyfile(h5_og_path, h5_path)

M = 125
Ns = int(5e8)

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

# delete the copy of the dataset we created
os.remove(h5_path)

# record the time in a file pixel by pixel so we don't have to wait for all
# the pixels to finish (in case something crashes haha)
outputFile = open("timings.txt", "a")
outputFile.write("{}\n".format(totalTime))
outputFile.close()







