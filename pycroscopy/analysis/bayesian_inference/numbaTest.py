from numba import jit
import numpy as np

def someFunction(arr, arrMax, arrMin):
    if arr[-1] > arrMax or arr[-1] < arrMin:
        return np.inf

    return 42.0

arr = np.array([[110], [110]])
arrMax = 120
arrMin = 100
print(someFunction(arr, arrMax, arrMin))
