import h5py
import matplotlib.pyplot as plt
#from mpi4py import MPI
from bayesian_inference import AdaptiveBayesianInference

h5_path = 'pzt_nanocap_6_split_bayesian_compensation_R_correction.h5'

with h5py.File(h5_path, mode='r+') as h5_f:

    h5_grp = h5_f['Measurement_000/Channel_000']

    h5_resh = h5_grp['Raw_Data-FFT_Filtering_000/Filtered_Data-Reshape_000/Reshaped_Data']

    abi = AdaptiveBayesianInference(h5_resh, verbose=True)

    fig1, fig2 = abi.test()
    plt.savefig('result0.png', 'result1.png')
