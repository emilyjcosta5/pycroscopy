import h5py
#from mpi4py import MPI
import io
from bayesian_inference import AdaptiveBayesianInference
from matplotlib import pyplot as plt
import pyUSID as usid
import numpy as np

h5_path = r"C:\Users\Administrator\Dropbox\GIv Bayesian June 2019\pzt_nanocap_6_split_bayesian_compensation_R_correction (Alvin Tan's conflicted copy 2019-06-25).h5"

with h5py.File(h5_path, mode='r+') as h5_f:

    h5_grp = h5_f['Measurement_000/Channel_000']
    f = usid.hdf_utils.get_attr(h5_grp, 'excitation_frequency_[Hz]')
    V0 = usid.hdf_utils.get_attr(h5_grp, 'excitation_amplitude_[V]')

    h5_resh = h5_f['Measurement_000/Channel_000/Raw_Data-FFT_Filtering_000/Filtered_Data-Reshape_000/Reshaped_Data']

    Ns = int(1e8)
    M = 125

    abi = AdaptiveBayesianInference(h5_resh, f=f, V0=V0, Ns=Ns, M=M)

    pos_in_batch = [138, 3994, 27935]

    # lol bad coding practices for the win
    abi._create_results_datasets()
    abi._unit_computation(pos_in_batch=pos_in_batch)
    abi._write_results_chunk(pos_in_batch=pos_in_batch)

    #usid.hdf_utils.print_tree(h5_f['Measurement_000/Channel_000/Raw_Data-FFT_Filtering_000/Filtered_Data-Reshape_000'])

    figs = [None, None, None]
    for i in range(len(pos_in_batch)):
        figs[i] = abi.plotPixel(pos_in_batch[i])
        figs[i].set_size_inches(40, 10)
        figs[i].savefig("Pixel{}_Ns{}_M{}.png".format(pos_in_batch[i], Ns, M))
        figs[i].show()

    input("Pause here to inspect graphs... Press <Enter> to exit program.")

    #figFor.show()
    #figRev.show()

    #breakpoint()

