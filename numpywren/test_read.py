import numpy as np
import time

arrcpy = np.memmap("/tmp/test", dtype='float64', mode='r', shape=(4096,4096))
tick = time.time()
copy = arrcpy.copy()
tock = time.time()
print(tock - tick)
