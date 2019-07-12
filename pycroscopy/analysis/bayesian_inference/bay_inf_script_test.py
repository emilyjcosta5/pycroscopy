import h5py
#from mpi4py import MPI
import io
from bayesian_inference import AdaptiveBayesianInference
from matplotlib import pyplot as plt
import pyUSID as usid
import numpy as np

h5_path = r"C:\Users\Administrator\Dropbox\GIv Bayesian June 2019\pzt_nanocap_6_split_bayesian_compensation_R_correction.h5"

with h5py.File(h5_path, mode='r+') as h5_f:

    h5_grp = h5_f['Measurement_000/Channel_000']
    f = usid.hdf_utils.get_attr(h5_grp, 'excitation_frequency_[Hz]')
    V0 = usid.hdf_utils.get_attr(h5_grp, 'excitation_amplitude_[V]')

    h5_resh = h5_f['Measurement_000/Channel_000/Raw_Data-FFT_Filtering_000/Filtered_Data-Reshape_000/Reshaped_Data']

    abi = AdaptiveBayesianInference(h5_resh, f=f, V0=V0, Ns=int(1e7))

    figFor, figRev = abi.test(pix_ind=27935)

    figFor.show()
    figRev.show()

    breakpoint()

