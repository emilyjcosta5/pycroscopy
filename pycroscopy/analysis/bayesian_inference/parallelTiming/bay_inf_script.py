import os
import subprocess
import sys
import io
import h5py
#from mpi4py import MPI
import time
import random
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

# Make a copy of the dataset so multiple processes can access it at the same time
h5_og_path = r"/home/29t/pzt_nanocap_6_split_bayesian_compensation_R_correction (Alvin Tan's conflicted copy 2019-06-25).h5"
h5_path = r"/lustre/or-hydra/cades-birthright/29t/computeDataBoi{}.h5".format(time.time())
copyfile(h5_og_path, h5_path)

#h5_path = r"C:/Users/Administrator/Dropbox/GIv Bayesian June 2019/pzt_nanocap_6_split_bayesian_compensation_R_correction (Alvin Tan's conflicted copy 2019-06-25).h5"

M = 100
Ns = int(7e7)

#skip mpi for now
#with h5py.File(h5_path, mode='r+', driver='mpio', comm=MPI.COMM_WORLD) as h5_f:
with h5py.File(h5_path, mode='r+') as h5_f:
    h5_grp = h5_f['Measurement_000/Channel_000']
    f = usid.hdf_utils.get_attr(h5_grp, 'excitation_frequency_[Hz]')
    V0 = usid.hdf_utils.get_attr(h5_grp, 'excitation_amplitude_[V]')

    #original dataset
    h5_resh = h5_grp['Raw_Data-FFT_Filtering_000/Filtered_Data-Reshape_000/Reshaped_Data']

    # Each 'X' axes has 256 images, so [0,1] have 512 images
    myDict = {}
    #myDict['X'] = [0,1]
    '''
    # Pseudorandomly select two rows to process
    randNum1 = random.randrange(266)
    randNum2 = random.randrange(266)
    myDict['X'] = [randNum1, randNum2]

    print("Running compute on rows {} and {}".format(randNum1, randNum2))
    '''
    # Creates object from original data
    abi = AdaptiveBayesianInference(h5_resh)
    '''
    #slice data
    subset = abi.h5_main.slice_to_dataset(slice_dict=myDict)
    #create object from sliced data
    abi_subset = AdaptiveBayesianInference(subset, f=f, V0=V0, Ns=Ns, M=M)
    '''
    #compute
    startTime = time.time()
    #h5_bayes_group = abi_subset.test()
    h5_bayes_group = abi.compute()
    totalTime = time.time() - startTime

    #copy result group to new h5 file
    with h5py.File('bayesian_results.h5', 'a') as f:
        f.create_group("results0")
        h5_f.copy(h5_bayes_group, f, name="h5_bayes_group")

#print(time.time() - startTime)
outputFile = open("parallelTimings.txt", "a")
outputFile.write("{}\n".format(totalTime))
outputFile.close()
