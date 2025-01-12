import h5py
#from mpi4py import MPI
from bayesian_inference import AdaptiveBayesianInference

h5_path = 'pzt_nanocap_6_split_bayesian_compensation_R_correction.h5'

with h5py.File(h5_path, mode='r+') as h5_f:

    h5_grp = h5_f['Measurement_000/Channel_000']

    h5_resh = h5_grp['Raw_Data-FFT_Filtering_000/Filtered_Data-Reshape_000/Reshaped_Data']

    abi = AdaptiveBayesianInference(h5_resh, verbose=True)

    h5_bayes_grp = abi.compute()
