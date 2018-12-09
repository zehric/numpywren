import numpy as np
import matplotlib.pyplot as plt
import argparse
import boto3
import io
import os
import hashlib
import botocore.session
import aiobotocore
import asyncio
import time
import math
import pywren
import concurrent.futures as fs

parser = argparse.ArgumentParser(description="S3 IO benchmarks")
parser.add_argument('data_size', type=float)
parser.add_argument('num_keys', type=int)
parser.add_argument('method', type=str)

# DATA_SIZE = 134217728
args = parser.parse_args()
DATA_SIZE = args.data_size
NUM_KEYS = args.num_keys
METHOD = args.method
OUTPUT_NAME = "{}_e{}_{}".format(METHOD, int(math.log10(DATA_SIZE)), NUM_KEYS)

def write_keys(prefix, size=DATA_SIZE, num_keys=10):
    write_bytes = b"\x00"+os.urandom(int(size))+ b"\x00"
    client = boto3.client('s3')
    t = time.time()
    for i in range(num_keys):
        key = "{0}_{1}_{2}".format(prefix, i, os.urandom(256))
        sha1 = hashlib.sha1()
        sha1.update(key.encode())
        key = sha1.hexdigest()
        client.put_object(Key=key, Body=write_bytes, Bucket="uric-cache-benchmarks")
    e = time.time()
    return t,e,num_keys

def put_object(client, key, data, bucket):
    backoff = 1
    while True:
        try:
            return client.put_object(Key=key, Body=data, Bucket="uric-cache-benchmarks")
        except:
            time.sleep(backoff)
            backoff *= 2

def get_object(client, key, bucket):
    backoff = 1
    while True:
        try:
            return client.get_object(Key=key, Bucket="uric-cache-benchmarks")['Body'].read()
        except:
            time.sleep(backoff)
            backoff *= 2

def write_keys_threaded(prefix, size=DATA_SIZE, num_keys=10, threads=10):
    write_bytes = b"\x00"+os.urandom(int(size))+ b"\x00"
    t = time.time()
    executor = fs.ThreadPoolExecutor(threads)
    client = boto3.client('s3')
    futures = []
    for i in range(num_keys):
        key = "{0}_{1}_{2}".format(prefix, i, os.urandom(256))
        sha1 = hashlib.sha1()
        sha1.update(key.encode())
        key = sha1.hexdigest()
        futures.append(executor.submit(put_object, client, key, write_bytes, "uric-cache-benchmarks"))
    fs.wait(futures)
    [f.result() for f in futures]
    e = time.time()
    return t,e,num_keys

def read_keys_threaded(prefix, num_keys=10, threads=10):
    t = time.time()
    executor = fs.ThreadPoolExecutor(threads)
    client = boto3.client('s3')
    futures = []
    for i in range(num_keys):
        key = "{0}_{1}_{2}".format(prefix, i, os.urandom(256))
        key = sha1.hexdigest()
        futures.append(executor.submit(get_object, client, key, "uric-cache-benchmarks"))
    fs.wait(futures)
    arr = [f.result() for f in futures]
    e = time.time()
    return t,e,num_keys

async def put_object_backoff(client, key, bucket, data, backoff_start=1):
    backoff = backoff_start
    while True:
        try:
            response_task = await client.put_object(Key=key, Bucket=bucket, Body=data)
            break
        except:
            await asyncio.sleep(backoff)
            backoff *= 2
    return response_task
            
async def _write_keys_async(loop, keys_data, bucket="uric-cache-benchmarks"):
    session = aiobotocore.get_session(loop=loop)
    async with session.create_client('s3', use_ssl=False, verify=False, region_name='us-west-2') as client:
        tasks = []
        outb = io.BytesIO()
        for key, data in keys_data:
            tasks.append(asyncio.ensure_future(put_object_backoff(client, key, bucket, data)))
        await asyncio.gather(*tasks)
    return 

def write_keys_async(prefix, size=DATA_SIZE, num_keys=100):
    client = boto3.client('s3')
    write_bytes = b"\x00"+os.urandom(int(size))+ b"\x00" 
    keys_data = []
    t = time.time()
    for i in range(num_keys):
        key = "{0}_{1}_{2}".format(prefix, i, os.urandom(256))
        sha1 = hashlib.sha1()
        sha1.update(key.encode())
        key = sha1.hexdigest()
        keys_data.append((key, write_bytes))
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_write_keys_async(loop, keys_data))
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


# w = write_keys_threaded("poop", num_keys=10, threads=10)
# print(w[1] - w[0])
# r = write_keys_async("poop", num_keys=10)
# print(r[1] - r[0])
# exit()

session = botocore.session.get_session()

extra_env = {"AWS_DEFAULT_REGION": "us-west-2", "AWS_ACCESS_KEY_ID": session.get_credentials().access_key, "AWS_SECRET_ACCESS_KEY": session.get_credentials().secret_key}

config = pywren.wrenconfig.default()
config['runtime']['s3_bucket'] = 'numpywrenpublic'
key = "pywren.runtime/pywren_runtime-3.6-numpywren.tar.gz"
config['runtime']['s3_key'] = key

pwex = pywren.standalone_executor(config=config)

if METHOD == "threaded":
    futures = pwex.map(lambda x: write_keys_threaded(x, num_keys=NUM_KEYS, threads=NUM_KEYS), range(36*25), extra_env=extra_env)
else:
    futures = pwex.map(lambda x: write_keys_async(x, num_keys=NUM_KEYS), range(36*25), extra_env=extra_env)


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
iops, bins = profile_iops(results)

np.savez("s3iops_pure_python_write_{}".format(OUTPUT_NAME), iops=iops, bins=bins)

plt.plot(bins - min(bins), iops)
plt.ylabel("(write) IOPS/s")
plt.xlabel("time")
plt.savefig('s3iops_pure_python_write_{}.png'.format(OUTPUT_NAME))
