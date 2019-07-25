import h5py
#from mpi4py import MPI
from bayesian_inference import AdaptiveBayesianInference
import time


startTime = time.time()

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

h5_path = 'pzt_nanocap_6_split_bayesian_compensation_R_correction(EmilyAddedSubset).h5'

#skip mpi for now
#with h5py.File(h5_path, mode='r+', driver='mpio', comm=MPI.COMM_WORLD) as h5_f:
module load py-pip
module load hdf5/1.10.3
module load python
try:
    # This package is not part of anaconda and may need to be installed.
    import pycroscopy
except ImportError:
    warn('wget not found.  Will install with pip.')
    import pip
    install('pycroscopy')
    import pycroscopy
try:
    import pyUSID as usid
except ImportError:
    warn('pyUSID not found.  Will install with pip.')
    import pip
    install('pyUSID')
    import pyUSID as usid
try:
    import numba
except ImportError:
    warn('pyUSID not found.  Will install with pip.')
    import numba
    install('pyUSID')
    import numba
try:
    # This package is not part of anaconda and may need to be installed.
    import h5py
except ImportError:
    warn('wget not found.  Will install with pip.')
    import pip
    install('h5py')
    import h5py
with h5py.File(h5_path, mode='r+') as h5_f:
    h5_grp = h5_f['Measurement_000/Channel_000']
    #original dataset
    h5_resh = h5_grp['Raw_Data-FFT_Filtering_000/Filtered_Data-Reshape_000/Reshaped_Data']
    # Each 'X' axes has 256 images, so [0,1] have 512 images
    myDict = {}
    myDict['X'] = [0,1]
    # Creates object from original data
    abi = AdaptiveBayesianInference(h5_resh)
    #slice data
    subset = abi.h5_main.slice_to_dataset(slice_dict=myDict)
    #create object from sliced data
    abi_subset = AdaptiveBayesianInference(subset)
    #compute
    h5_bayes_group = abi_subset.compute()
    #copy result group to new h5 file
    with h5py.File('bayesian_results.h5', 'a') as f:
        f.create_group("results0")
        h5py.copy(h5_bayes_group, f, name="h5_bayes_group")
print(time.time() - startTime)
