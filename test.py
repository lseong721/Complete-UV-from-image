from numba import njit, prange
import numpy as np

@njit(parallel=True)
def process_items(items):
    # output = items.copy()
    output = np.zeros(5)
    print(output.dtype)
    for i in prange(len(items)):
        output[i] = items[i] * items[i]
    return output

items = np.array([1, 2, 3, 4, 5])
results = process_items(items)

print(results)