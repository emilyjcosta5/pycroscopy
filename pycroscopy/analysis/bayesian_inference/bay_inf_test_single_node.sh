export PATH=/gpfs/alpine/gen011/world-shared/native-build/anaconda3/bin:$PATH
export PYTHONPATH=/gpfs/alpine/gen011/scratch/ecost020/numba/numba:$PYTHONPATH
export PYTHONPATH=/gpfs/alpine/gen011/scratch/ecost020/cupy/cupy:$PYTHONPATH
export PYTHONPATH=/gpfs/alpine/gen011/scratch/ecost020/pycroscopy/pycroscopy:$PYTHONPATH

jsrun -n1 -a1 -g6 -c42 python bay_inf_test_single_node.py
