import numpy as np
import time
import mmap
import pickle
import os
import sys

arr = np.random.rand(4096, 4096)
tick = time.time()
arrcpy = np.memmap("/tmp/test", dtype='float64', mode='w+', shape=(4096,4096))
arrcpy[:] = arr[:]
tock = time.time()
print(tock - tick)

tick = time.time()
fd = os.open('/tmp/test_mmap', os.O_RDWR | os.O_CREAT)
os.ftruncate(fd, arr.nbytes)
mm = mmap.mmap(fd, 0)
# arr.tofile(mm)
mm.write(arr.tobytes())
tock = time.time()
print(tock - tick)
os.close(fd)


# tick = time.time()
# copy = arrcpy.copy()
# tock = time.time()
# print(tock - tick)
