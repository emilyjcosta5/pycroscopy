import h5py
#from mpi4py import MPI
from bayesian_inference import AdaptiveBayesianInference
import time


startTime = time.time()

h5_path = 'pzt_nanocap_6_split_bayesian_compensation_R_correction(EmilyAddedSubset).h5'

#skip mpi for now
#with h5py.File(h5_path, mode='r+', driver='mpio', comm=MPI.COMM_WORLD) as h5_f:
with h5py.File(h5_path, mode='r+') as h5_f:
    h5_grp = h5_f['Measurement_000/Channel_000']

    h5_resh = h5_f['Measurement_000/Channel_000/Raw_Data-FFT_Filtering_000/Filtered_Data-Reshape_000/Reshaped_Data']
    myDict = {}
    myDict['X'] = [0,1]
    abi = AdaptiveBayesianInference(h5_resh)
    subset = abi.h5_main.slice_to_dataset(slice_dict=myDict)
    abi_subset = AdaptiveBayesianInference(subset)
    h5_bayes_group = abi_subset.compute()

print(time.time() - startTime)
