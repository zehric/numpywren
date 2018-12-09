from numpywren.matrix import *
import numpy as np
import pywren
from numpywren.matrix_init import shard_matrix

def main():
    #X = np.random.randn(64, 64)
    #A = X.dot(X.T) + np.eye(X.shape[0])
    #y = np.random.randn(16)
    A = np.arange(64*64).reshape((64,64))
    pwex = pywren.default_executor()
    shard_size = 16
    shard_sizes = (shard_size, shard_size)
    A_sharded= BigMatrix("cholesky_test_A", shape=A.shape, shard_sizes=shard_sizes, write_header=True)
    A_sharded.free()
    shard_matrix(A_sharded, A)
    print()
    print()
    print(A_sharded.get_block(0,0)[:5,:5])
    print()
    print()
    print(A[:5, :5])

if __name__ == '__main__':
    main()


