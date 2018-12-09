from __future__ import print_function

import aiobotocore
import argparse
import asyncio
import boto3
import botocore
import botocore.session
import botocore.exceptions
import multiprocessing
import concurrent.futures as fs
import hashlib
import io
import json
import logging
import os
import pywren
import pywren.runtime
import sys
import time
import zipfile

from fastclient import FastClient
import numpy as np
import ctypes

from pywren import ec2standalone


flatten = lambda l: [item for sublist in l for item in sublist]



#def gen_fixed_experiment(num_instances, num_cores_per_instance, num_keys_per_core, keys):
 #   key_mapping = [[[] for j in range(num_cores_per_instance)] for i in range(num_instances)]
 #   used_keys = set()
 #   unused_keys = set(keys)
 #   for i in range(num_instances):
 #       for k in range(num_keys_per_core):
 #           key_k = keys[(i*num_keys_per_core+k) % len(keys)]
 #           used_keys.add(key_k)
 #           if key_k in unused_keys: unused_keys.remove(key_k)
 #           for j in range(num_cores_per_instance):
 #               key_mapping[i][j].append(key_k)
 #   return key_mapping, used_keys, unused_keys
def gen_fixed_experiment(num_instances, num_cores_per_instance, num_keys_per_core, keys):
    rng = np.random.RandomState(105)
    key_mapping = [[[] for j in range(num_cores_per_instance)] for i in range(num_instances)]
    used_keys = set()
    unused_keys = set(keys)
    for i in range(num_instances):
        for k in range(num_keys_per_core):
            key_k = keys[(i*num_keys_per_core+k) % len(keys)]
            used_keys.add(key_k)
            if key_k in unused_keys: unused_keys.remove(key_k)
            for j in range(num_cores_per_instance):
                key_mapping[i][j].append(key_k)
    key_mapping = np.array(key_mapping)
    for i in range(num_instances):
        for j in range(num_cores_per_instance):
            swap = j % num_keys_per_core
            a = key_mapping[i][j][0]
            b = key_mapping[i][j][swap]
            key_mapping[i][j][0] = b
            key_mapping[i][j][swap] = a
            rng.shuffle(key_mapping[i][j][1:])
    return key_mapping, used_keys, unused_keys
def gen_rand_experiment(num_instances, num_cores_per_instance, num_keys_per_core, keys, key_dist=None):
    if key_dist is not None:
        assert len(key_dist) == len(keys)
    rng = np.random.RandomState(101)
    key_mapping = rng.choice(keys, 
                             size=num_instances*num_keys_per_core*num_cores_per_instance,
                             replace=True,
                             p=key_dist)
    used_keys = np.unique(key_mapping)
    unused_keys = np.setdiff1d(key_mapping, used_keys)
    key_mapping = key_mapping.reshape((num_instances, num_cores_per_instance, num_keys_per_core))
    used_keys = set(used_keys)
    unused_keys = set(unused_keys)
    return key_mapping, used_keys, unused_keys
def gen_clustered_rand_experiment(num_instances, num_cores_per_instance, num_keys_per_core, keys):
    rng = np.random.RandomState(102)
    selected_keys = rng.randint(len(keys), size=max(1,len(keys)//10))
    key_dist = np.abs(rng.randn(len(keys))) + 1e-3
    key_dist[selected_keys] += 2
    key_dist /= sum(key_dist)
    return gen_rand_experiment(num_instances, num_cores_per_instance, num_keys_per_core, keys, key_dist)





def gen_key(folder, prefix, key_id, data_size):
    key = "{0}_{1}_{2}".format(prefix, key_id, data_size)
    sha1 = hashlib.sha1()
    sha1.update(key.encode())
    key = sha1.hexdigest()
    return "{0}/{1}".format(folder, key)
def gen_instance_unique_name(base, instance_id):
    return "{0}_{1}".format(base, instance_id)

def create_sqs_queue(sqs_queue_name):
    sqs = boto3.resource('sqs', region_name='us-west-2')
    try:
        sqs.get_queue_by_name(QueueName=sqs_queue_name)
    except:
        sqs.create_queue(QueueName=sqs_queue_name,
                         Attributes={'VisibilityTimeout' : "20"})
def launch_many_instances(num_instances, **kwargs):
    config = pywren.wrenconfig.default()
    sqs = config['standalone']['sqs_queue_name']
    default_extra_args = {'pywren_git_branch':'master',
                          'pywren_git_commit':None,
                          'parallelism':1,
                          'spot_price':0.0,
                          'instance_type':None,
                          'idle_terminate_granularity':None,
                          'max_idle_time':None,
                          'number':1,}
    default_extra_args.update(kwargs)
    inst_list = []
    for i in range(num_instances):
        config['standalone']['sqs_queue_name'] = gen_instance_unique_name(sqs, i)
        create_sqs_queue(config['standalone']['sqs_queue_name'])
        # inst_list += launch_instances(config, **default_extra_args)
        p = multiprocessing.Process(target=launch_instances, args=(config, default_extra_args))
        inst_list.append(p)
        p.start()
        time.sleep(1)
    for p in inst_list:
        p.join()
    
def launch_instances(config, kwargs):
    '''From pywren.ec2standalone and modified to work for multiple sqs queues.'''
    sc = config['standalone']
    aws_region = config['account']['aws_region']
    pywren_git_branch = kwargs['pywren_git_branch']
    pywren_git_commit = kwargs['pywren_git_commit']
    parallelism = kwargs['parallelism']
    spot_price = kwargs['spot_price']
    ec2_instance_type = kwargs['ec2_instance_type']
    idle_terminate_granularity = kwargs['idle_terminate_granularity']
    max_idle_time = kwargs['max_idle_time']
    number = kwargs['number']
    cache_size = kwargs['cache_size']
    if max_idle_time is not None:
        sc['max_idle_time'] = max_idle_time
    if idle_terminate_granularity is not None:
        sc['idle_terminate_granularity'] = idle_terminate_granularity
    if ec2_instance_type is not None:
        sc['ec2_instance_type'] = ec2_instance_type
    use_fast_io = sc.get("fast_io", False)
    availability_zone = sc.get("availability_zone", None)
    inst_list = ec2standalone.launch_instances(number,
                                               sc['target_ami'], aws_region,
                                               sc['ec2_ssh_key'],
                                               sc['ec2_instance_type'],
                                               sc['instance_name'],
                                               sc['instance_profile_name'],
                                               sc['sqs_queue_name'],
                                               config['s3']['bucket'],
                                               cache_size=cache_size,
                                               max_idle_time=sc['max_idle_time'],
                                               idle_terminate_granularity=\
                                               sc['idle_terminate_granularity'],
                                               pywren_git_branch=pywren_git_branch,
                                               pywren_git_commit=pywren_git_commit,
                                               availability_zone=availability_zone,
                                               fast_io=use_fast_io,
                                               parallelism=parallelism,
                                               spot_price=spot_price)
    print("launched {0}:".format(sc.get('instance_id','')))
    ec2standalone.prettyprint_instances(inst_list)
    return inst_list
def terminate_instances(inst_list):
    print("terminate")
    ec2standalone.prettyprint_instances(inst_list)
    ec2standalone.terminate_instances(inst_list)

###################################################################################################
########################## v Writing keys v #######################################################
###################################################################################################
def write_keys(folder, prefix, bucket, size=1e4, num_keys=10):
    write_bytes = b"\x00"+os.urandom(int(size))+ b"\x00" 
    client = boto3.client('s3')
    t = time.time()
    key_names = []
    for i in range(num_keys):
        key_names.append(gen_key(folder, prefix, i, int(size)))
        client.put_object(Key=key_names[-1],
                          Body=write_bytes,
                          Bucket=bucket)
    e = time.time() 
    return t,e,num_keys,key_names

def put_object(client, key, data, bucket):
    backoff = 1
    try:
        return client.put_object(Key=key, Body=data, Bucket="s3iopstest")
    except:
        time.sleep(backoff)
        backoff *= 2
        

def write_keys_threaded(folder, prefix, bucket, size=1e4, num_keys=10, threads=10):
    write_bytes = b"\x00"+os.urandom(int(size))+ b"\x00" 
    t = time.time()
    executor = fs.ThreadPoolExecutor(threads)
    client = boto3.client('s3')
    futures = []
    key_names = []
    for i in range(num_keys):
        key_names.append(gen_key(folder, prefix, i, int(size)))
        futures.append(executor.submit(put_object, client,
                                                   key_names[-1],
                                                   write_bytes,
                                                   bucket))
    fs.wait(futures)
    [f.result() for f in futures]
    e = time.time() 
    return t,e,num_keys,key_names

async def put_object_backoff(client, key, bucket, data, backoff_start=0.3):
    backoff = backoff_start
    while True:
        try:
            response_task = await client.put_object(Key=key, Bucket=bucket, Body=data)
            break
        except:
            await asyncio.sleep(backoff)
            backoff *= 2
    return response_task
            
async def _write_keys_async(loop, keys_data, bucket):
    session = aiobotocore.get_session(loop=loop)
    async with session.create_client('s3', use_ssl=False, verify=False, region_name='us-west-2',) as client:
        tasks = []
        outb = io.BytesIO()
        for key, data in keys_data:
            tasks.append(asyncio.ensure_future(put_object_backoff(client, key, bucket, data)))
        await asyncio.gather(*tasks)
    return 

def write_keys_async(folder, prefix='test', size=1e4, num_keys=10):
    client = boto3.client('s3')
    write_bytes = b"\x00"+os.urandom(int(size))+ b"\x00" 
    keys_data = []
    t = time.time()
    for i in range(num_keys):
        key_names.append(gen_key(folder, prefix, i, int(size)))
        keys_data.append((key_names[-1], write_bytes))
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_write_keys_async(loop, keys_data))
    e = time.time() 
    return t,e,num_keys,key_names
###################################################################################################
########################## ^ Writing keys ^ #######################################################
###################################################################################################

###################################################################################################
########################## v No Cache v ###########################################################
###################################################################################################
def get_object_backoff(client, key, bucket):
    backoff = 1
    num_tries = 1
    try:
        return (client.get_object(Key=key, Bucket=bucket), num_tries)
    except:
        time.sleep(backoff)
        backoff *= 2
        num_tries += 1

def read_keys_no_cache(bucket, instance_id, key_list):
    t1 = time.time()
    client = boto3.client('s3')
    t2 = time.time()
    key_names = []
    objects = []
    time_gets = []
    num_tries_list = []
    for key in key_list:
        s = time.time()
        obj, num_tries = get_object_backoff(client, key, bucket)
        data = obj['Body'].read()
        f = time.time()
        objects.append(data)
        num_tries_list.append(num_tries)
        time_gets.append((s, f))
    e = time.time()
    ret_dict = {}
    ret_dict['time_all'] = e-t1
    ret_dict['time_client'] = t2-t1
    ret_dict['time_gets'] = time_gets
    ret_dict['time_all_te'] = (t1,e)
    ret_dict['time_client_te'] = (t1,t2)
    ret_dict['n_tries'] = num_tries_list
    ret_dict['key'] = key_list
    ret_dict['instance_id'] = instance_id
    
    return ret_dict

# def read_keys_no_cache(bucket, instance_id, key_list):
#     t = time.time()
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)

#     tasks = [read_key_no_cache_async(loop, bucket, key, instance_id) for key in key_list]
#     futures = [asyncio.ensure_future(task) for task in tasks]
#     results = loop.run_until_complete(asyncio.wait(futures))
#     results_futures = [future.result() for future in futures]
#     e = time.time()
#     # in milliseconds 
#     for r in results_futures:
#         r['total_time_instance'] = e-t
#         r['total_time_instance_te'] = (t,e)
#     return results_futures

    
# async def read_key_no_cache_async(loop, bucket, key, instance_id):
#     t1 = time.time()
#     session = aiobotocore.get_session(loop=loop)
#     t2 = time.time()
#     async with session.create_client('s3', use_ssl=False, verify=False, region_name='us-west-2') as client:
#         t3 = time.time()
#         n_tries = 0
#         max_n_tries = 5
#         ret = None
#         while ret is None and n_tries <= max_n_tries:
#             try:
#                 ret = await client.get_object(Bucket=bucket, Key=key)
#             except Exception as e:
#                 raise
#                 n_tries += 1
#         t4 = time.time()
#     e = time.time()
    
#     # in milliseconds 
#     ret_dict = {}
#     ret_dict['time_all'] = e-t1
#     ret_dict['time_session'] = t2-t1
#     ret_dict['time_client'] = t3-t2
#     ret_dict['time_get'] = t4-t3
#     ret_dict['time_all_te'] = (t1,e)
#     ret_dict['time_session_te'] = (t1,t2)
#     ret_dict['time_client_te'] = (t2,t3)
#     ret_dict['time_get_te'] = (t3,t4)
#     ret_dict['n_tries'] = (n_tries+1)
#     ret_dict['success'] = (ret is not None)
#     ret_dict['key'] = key
#     ret_dict['instance_id'] = instance_id
    
#     return ret_dict

###################################################################################################
########################## ^ No Cache ^ ###########################################################
###################################################################################################

###################################################################################################
########################## v For Cache v ##########################################################
###################################################################################################
def read_keys_with_cache(bucket, instance_id, key_list):
    fclient = FastClient(so_bucket="zehric-pywren-149")
    fclient.cache_so()
    doubles_s = np.zeros(len(key_list), np.float64)
    doubles_f = np.zeros(len(key_list), np.float64)
    d_ptr_s = doubles_s.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    d_ptr_f = doubles_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    keys = []
    for key in key_list:
        keys.append(ctypes.c_char_p(key.encode()))

    fclient.pin_objects(keys, d_ptr_s, d_ptr_f, num_threads=1)
    return list(zip(doubles_s, doubles_f))

###################################################################################################
########################## ^ For Cache ^ ##########################################################
###################################################################################################
    
    
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


