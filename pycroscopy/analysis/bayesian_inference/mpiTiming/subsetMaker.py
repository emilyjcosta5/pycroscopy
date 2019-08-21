from __future__ import division, print_function, absolute_import, unicode_literals
import os
import subprocess
import sys
import io
import h5py
#from mpi4py import MPI
import time
import random
from shutil import copyfile
import numpy as np

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
from pyUSID.io.hdf_utils import write_main_dataset, write_ind_val_dsets
from pyUSID.io.write_utils import Dimension

# Make a copy of the dataset so multiple processes can access it at the same time
#h5_og_path = r"/home/29t/pzt_nanocap_6_split_bayesian_compensation_R_correction (Alvin Tan's conflicted copy 2019-06-25).h5"
#h5_path = r"/lustre/or-hydra/cades-birthright/29t/computeDataBoi{}.h5".format(time.time())
#copyfile(h5_og_path, h5_path)

h5_path = r"C:/Users/Administrator/Dropbox/GIv Bayesian June 2019/pzt_nanocap_6_split_bayesian_compensation_R_correction (Alvin Tan's conflicted copy 2019-06-25).h5"

numPixels = 500
#M = 100
#Ns = int(7e7)

#skip mpi for now
#with h5py.File(h5_path, mode='r+', driver='mpio', comm=MPI.COMM_WORLD) as h5_f:
with h5py.File(h5_path, mode='r+') as h5_f:
    h5_grp = h5_f['Measurement_000/Channel_000']
    f = usid.hdf_utils.get_attr(h5_grp, 'excitation_frequency_[Hz]')
    V0 = usid.hdf_utils.get_attr(h5_grp, 'excitation_amplitude_[V]')

    #original dataset
    h5_resh = usid.USIDataset(h5_grp['Raw_Data-FFT_Filtering_000/Filtered_Data-Reshape_000/Reshaped_Data'])
    pixelInds = np.random.randint(0, h5_resh[()].shape[0], numPixels)
    print("PixelInds is {} with shape {}".format(pixelInds, pixelInds.shape))

    #copy subset to new h5 file
    f = h5py.File('subsetFile{}.h5'.format(time.time()), 'a')
    subsetGroup = f.create_group("subsetBoi")
    h5_spec_inds, h5_spec_vals = write_ind_val_dsets(subsetGroup, Dimension("Bias", "V", int(h5_resh.h5_spec_inds.size)), is_spectral=True)
    h5_spec_vals[()] = h5_resh.h5_spec_vals[()]
    h5_pos_inds, h5_pos_vals = write_ind_val_dsets(subsetGroup, Dimension("Position", "m", numPixels), is_spectral=False)
    #h5_pos_vals[()] = h5_resh.h5_pos_vals[()][pixelInds, :]
    h5_subset = write_main_dataset(subsetGroup, (numPixels, h5_resh.shape[1]), "Measured Current", "Current",
                                   "nA", None, None, dtype=np.float64,
                                   h5_pos_inds = h5_pos_inds,
                                   h5_pos_vals = h5_pos_vals,
                                   h5_spec_inds = h5_spec_inds,
                                   h5_spec_vals = h5_spec_vals)
    print("check if main returns: {}".format(usid.hdf_utils.check_if_main(h5_subset)))

f.close()
