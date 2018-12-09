from __future__ import print_function

from fastclient import FastClient
import ctypes
import numpy as np



def main():
    DATA_SIZE=104857600
    NUM_KEYS=320
    NUM_KEYS_PER_CORE=10
    BENCHMARK_DIR = 'broadcast_benchmark--large_key3'
    KEY_DIR = '{0}/key_data'.format(BENCHMARK_DIR)
    key_dat_path = "../cache/{0}/put_data_2exp{1}_{2}.npy".format(KEY_DIR, int(np.log2(DATA_SIZE)), NUM_KEYS)
    KEYS = np.load(key_dat_path).item()['keys']



    ret = read_keys_with_cache(None, [KEYS[0]])
    print("Returned key = {0}".format(ret))



def read_keys_with_cache(bucket, key_list):
    fclient = FastClient(so_bucket="zehric-pywren-149")
    fclient.cache_so()
    print("Keys for S3: {0}".format(key_list))
    return fclient.pin_objects2(key_list, num_threads=1)
    #return list(zip(doubles_s, doubles_f))




if __name__ == '__main__':
    main()





