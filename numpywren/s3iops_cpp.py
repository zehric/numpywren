import numpy as np
import matplotlib.pyplot as plt
import ctypes
import argparse
import botocore.session
import boto3
import os
import hashlib
import time
import math
import pywren
from fastio import FastIO

parser = argparse.ArgumentParser(description="S3 IO benchmarks")
parser.add_argument('data_size', type=float)
parser.add_argument('num_keys', type=int)

# DATA_SIZE = 134217728
args = parser.parse_args()
DATA_SIZE = args.data_size
NUM_KEYS = args.num_keys
OUTPUT_NAME = "e{}_{}".format(int(math.log10(DATA_SIZE)), NUM_KEYS)
bucket = "uric-cache-benchmarks"

def write_keys(prefix, size=DATA_SIZE, num_keys=10):
    client = boto3.client('s3')
    fio = FastIO(so_bucket="zehric-pywren-149", prefix=prefix)
    fio.cache_so()
    fio.start_api()
    mat_in = np.random.randint(0, 256, int(size)).astype('uint8')
    t = time.time()
    ptrs_in = []
    ptrs_out = []
    buckets = []
    keys = []
    buffer_sizes = []
    start = time.time()
    for i in range(num_keys):
        mat_ptr_in = mat_in
        # mat_ptr_out =  mat_out[i*obj_size:(i+1)*obj_size]
        ptr_in = ctypes.c_void_p(mat_ptr_in.ctypes.data)
        ptrs_in.append(ptr_in)
        # ptr_out = ctypes.c_void_p(mat_ptr_out.ctypes.data)
        # ptrs_out.append(ptr_out)
        key = "s3iops/{0}_{1}_{2}".format(prefix, i, os.urandom(256))
        sha1 = hashlib.sha1()
        sha1.update(key.encode())
        key = sha1.hexdigest()
        buckets.append(ctypes.c_char_p(bucket.encode()))
        keys.append(ctypes.c_char_p(key.encode()))
        buffer_sizes.append(ctypes.c_long(mat_ptr_in.nbytes))
    fio.put_objects(ptrs_in, buffer_sizes, buckets, keys, threads=num_keys)
    e = time.time()
    return t,e,num_keys

def profile_iops(results):
    min_time = min(results, key=lambda x: x[0])[0]
    max_time = max(results, key=lambda x: x[1])[1]
    tot_time = (max_time - min_time)

    bins = np.linspace(min_time, max_time, max_time - min_time)
    #return bins, min_time, max_time
    iops = np.zeros(len(bins))

    for start_time, end_time, num_ops in results:
        start_bin, end_bin = np.searchsorted(bins, [int(start_time), int(end_time)])
        iops[start_bin:((end_bin)+1)]  += num_ops/(end_time - start_time)
    return iops, bins


# w = write_keys("poop", num_keys=1000)
# print(w[1] - w[0])
# exit()

session = botocore.session.get_session()

extra_env = {"AWS_DEFAULT_REGION": "us-west-2", "AWS_ACCESS_KEY_ID": session.get_credentials().access_key, "AWS_SECRET_ACCESS_KEY": session.get_credentials().secret_key}

config = pywren.wrenconfig.default()
config['runtime']['s3_bucket'] = 'numpywrenpublic'
key = "pywren.runtime/pywren_runtime-3.6-numpywren.tar.gz"
config['runtime']['s3_key'] = key

pwex = pywren.standalone_executor(config=config)

futures = pwex.map(lambda x: write_keys(x, num_keys=NUM_KEYS), range(25*36), extra_env=extra_env)

pywren.wait(futures)

results = []
for i,f in enumerate(futures):
    try:
        if (f.done()):
            results.append(f.result())
    except Exception as e:
        print(e)
        pass

print(len(results))
if len(results) == 0:
    exit()
print(results[0][1] - results[0][0])
iops, bins = profile_iops(results)

np.savez("s3iops_cpp_write_{}".format(OUTPUT_NAME), iops=iops, bins=bins)

plt.plot(bins - min(bins), iops)
plt.ylabel("(write) IOPS/s")
plt.xlabel("time")
plt.savefig('s3iops_cpp_write_{}.png'.format(OUTPUT_NAME))
