# %%%%%%%%%%%%%%%% IMPORTS %%%%%%%%%%%%%%%%

# library
import copy
import logging
import numpy as np
import os
import pywren
import pywren.wrenconfig as wc
import sys
import time

# local
from benchmark_functions import *

# just for cache
if False:
    import ctypes
    from fastclient import FastClient
    
# %%%%%%%%%%%%%%%% GLOBALS %%%%%%%%%%%%%%%%
def initialize(args):
    global DATA_SIZE, NUM_KEYS, \
           OUTPUT_NAME ,BENCHMARK_DIR, FIGURE_DIR, KEY_DIR, RESULT_DIR, \
           EXTRA_ENV, CONFIG, PYWREN_BUCKET, STANDALONE_INSTANCES, KEYS
    
    
    DATA_SIZE = args.data_size
    NUM_KEYS = args.num_keys
    
    OUTPUT_NAME = lambda : "2exp{0}_{1}".format(int(np.log2(DATA_SIZE)), NUM_KEYS)
    BENCHMARK_DIR = 'broadcast_benchmark--{0}'.format(args.trial_number)
    FIGURE_DIR = 'figures'
    KEY_DIR = 'key_data'
    RESULT_DIR = 'results'
    FIGURE_DIR = '{0}/figures'.format(BENCHMARK_DIR)
    KEY_DIR = '{0}/key_data'.format(BENCHMARK_DIR)
    RESULT_DIR = '{0}/results'.format(BENCHMARK_DIR)
    for d in [FIGURE_DIR, KEY_DIR, RESULT_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    region = wc.default()["account"]["aws_region"]
    #session = botocore.session.get_session()
    EXTRA_ENV = {"AWS_DEFAULT_REGION": region,
                 #"AWS_ACCESS_KEY_ID": session.get_credentials().access_key,
                 #"AWS_SECRET_ACCESS_KEY": session.get_credentials().secret_key
                }
    
    def CONFIG():
        config = wc.default()
        pywren_bucket = config['s3']['bucket']
        config['runtime']['s3_bucket'] = 'numpywrenpublic'
        key = "pywren.runtime/pywren_runtime-3.6-numpywren.tar.gz"
        config['runtime']['s3_key'] = key
        return config
    config = CONFIG()
    PYWREN_BUCKET = config['s3']['bucket']
    pwex_key = pywren.default_executor(config=config)
    # 1 core = 2 vCPUs
    STANDALONE_INSTANCES = {
                            1  : {'name':'m4.large','target_ami':'ami-0bb5806b2e825a199'},
                            2  : {'name':'m4.xlarge','target_ami':'ami-0bb5806b2e825a199'},
                            4  : {'name':'m4.2xlarge','target_ami':'ami-0bb5806b2e825a199'},
                            8  : {'name':'m4.4xlarge','target_ami':'ami-0bb5806b2e825a199'},
                            20 : {'name':'m4.10xlarge','target_ami':'ami-0bb5806b2e825a199'},
                            32 : {'name':'m4.16xlarge','target_ami':'ami-0bb5806b2e825a199'},
                           } 
    key_dat_path = "{0}/put_data_2exp{1}_{2}.npy".format(KEY_DIR, int(np.log2(DATA_SIZE)), NUM_KEYS)
    if os.path.exists(key_dat_path):
        pass
    else:
        futures = pwex_key.map(lambda x: write_keys(BENCHMARK_DIR,
                                                    x,
                                                    PYWREN_BUCKET, 
                                                    size=DATA_SIZE, 
                                                    num_keys=1), 
                           range(NUM_KEYS), extra_env=EXTRA_ENV)
        pywren.wait(futures)
        results = [f.result() for f in futures if f.done()]
        assert len(results) == NUM_KEYS, "{0} < {1}".format(len(results), NUM_KEYS)
        np.save(key_dat_path, {'data_size':DATA_SIZE,
                               'keys':[r[-1][0] for r in results],
                               'time':[r[1]-r[2] for r in results]})
    KEYS = np.load(key_dat_path).item()['keys']
        
def get_inst_list():
    return globals()['__ec2standalone_inst_list']
def launch_instance_group(num_instances, num_cores):
    assert (num_cores in STANDALONE_INSTANCES)
    kwargs = {'ec2_instance_type':STANDALONE_INSTANCES[num_cores]['name'],
              'target_ami':STANDALONE_INSTANCES[num_cores]['target_ami'],
              'parallelism':num_cores,
              }
    inst_list = launch_many_instances(num_instances, **kwargs)
    if '__ec2standalone_inst_list' not in globals():
        globals()['__ec2standalone_inst_list'] = []
    globals()['__ec2standalone_inst_list'] += inst_list
    return inst_list
def get_executors(num_instances, num_cores, job_max_runtime=3600):
    assert (num_cores in STANDALONE_INSTANCES)
    executor_list = []
    for i in range(num_instances):
        config = CONFIG()
        config['standalone']['sqs_queue_name'] = gen_instance_unique_name(config['standalone']['sqs_queue_name'], i)
        config['standalone']['ec2_instance_type'] = STANDALONE_INSTANCES[num_cores]['name']
        config['standalone']['target_ami'] = STANDALONE_INSTANCES[num_cores]['target_ami']
        config['standalone']['parallelism'] = num_cores
        executor_list.append(pywren.standalone_executor(config=config))
    return executor_list
def terminate_instance_group():
    terminate_instances(globals()['__ec2standalone_inst_list'])
    globals()['__ec2standalone_inst_list'] = []      
def benchmark_broadcast(get_function,
                        keys,
                        num_instances,
                        num_cores_per_instance,
                        num_keys_per_core,
                        gen_experiment,
                        launch_group=True,
                        terminate_after=True,
                        **unused):
    assert num_cores_per_instance in STANDALONE_INSTANCES
    result_name = "{0}__{1}__{2}__{3}__{4}__{5}__{6}__{7}".format(get_function.__name__, len(keys), 
                    DATA_SIZE, num_instances, STANDALONE_INSTANCES[num_cores_per_instance],
                    num_cores_per_instance, num_keys_per_core, gen_experiment.__name__)
   # if os.path.exists("{0}/{1}.npy".format(RESULT_DIR,result_name)):
   #     raise Exception("Results already present.")
        
    print(pcolor.OKBLUE+
          "Running broadcast benchmark with parameters:"
          "\n  - function:         {0}"
          "\n  - # keys:           {1}"
          "\n  - # bytes/key:      {2}"
          "\n  - # instances:      {3}"
          "\n  - instance type:    {4}"
          "\n  - # cores/instance: {5}"
          "\n  - # keys/core:      {6}"
          "\n  - experiment:       {7}"
          "".format(get_function.__name__, len(keys), DATA_SIZE,
                    num_instances, STANDALONE_INSTANCES[num_cores_per_instance],
                    num_cores_per_instance, num_keys_per_core, gen_experiment.__name__)
         + pcolor.ENDC)
    print("-"*20)
    
    
    
   # if launch_group:
   #     print(pcolor.FAIL+"Launching instances..."+pcolor.ENDC)
   #     launch_instance_group(num_instances,num_cores_per_instance)
   #     print(pcolor.OKGREEN+"Finished launching."+pcolor.ENDC)
   # else:
   #     print(pcolor.OKGREEN+"Instances already launched."+pcolor.ENDC)
   # benchmark_executors = get_executors(num_instances,num_cores_per_instance)
    
    key_mapping, used_keys, unused_keys = gen_experiment(num_instances,
                                                         num_cores_per_instance,
                                                         num_keys_per_core,
                                                         keys)
                
    t = time.time()
   # futures = []
   # for i, pk in enumerate(zip(benchmark_executors, key_mapping)):
   #     pwex,key_list = pk
    print(len(np.unique(key_mapping[0][0])))
    print(np.array(key_mapping).shape)
    print(0,key_mapping[0][0])
    get_function(PYWREN_BUCKET, 0, key_mapping[0][0])
    print(pcolor.FAIL+"Waiting on results..."+pcolor.ENDC)
    #pywren.wait(futures)
    print(pcolor.OKGREEN+"Received results."+pcolor.ENDC)
    #results = [f.result() for f in futures if f.done()]
    e = time.time()

#     if terminate_after:
#         terminate_instance_group()

#     ret_dict = {}
#     ret_dict['total_time'] = (t,e)
#     ret_dict['function'] = get_function.__name__
#     ret_dict['num_keys'] = len(keys)
#     ret_dict['data_size'] = DATA_SIZE
#     ret_dict['num_instances'] = num_instances
#     ret_dict['instance_type'] = STANDALONE_INSTANCES[num_cores_per_instance]['name']
#     ret_dict['num_cores_per_instance'] = num_cores_per_instance
#     ret_dict['num_keys_per_core'] = num_keys_per_core
#     ret_dict['broadcast_distribution'] = gen_experiment.__name__
#     ret_dict['results'] = results
#     ret_dict['num_used_keys'] = len(used_keys)
#     ret_dict['num_unused_keys'] = len(unused_keys)
#     ret_dict['used_keys'] = used_keys
#     ret_dict['unused_keys'] = unused_keys
    
#     np.save("{0}/{1}.npy".format(RESULT_DIR, result_name), ret_dict)
    
    print()
    print(pcolor.OKGREEN+"Total time: {0} seconds".format(e-t)+pcolor.ENDC)
    
#     return ret_dict

class pcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def main():
    parser = argparse.ArgumentParser(description="Broadcast benchmarks")
    parser.add_argument('get_function', type=str)
    parser.add_argument('data_size', type=int)
    parser.add_argument('num_keys', type=int)
    parser.add_argument('num_instances', type=int)
    parser.add_argument('num_cores_per_instance', type=int)
    parser.add_argument('num_keys_per_core', type=int)
    parser.add_argument('gen_experiment', type=str)
    parser.add_argument('launch_group', type=str)
    parser.add_argument('terminate_after', type=str)
    parser.add_argument('trial_number', type=str)                                                  
    
    get_function_mapping = {'read_keys_no_cache': read_keys_no_cache,
                            'read_keys_with_cache':read_keys_with_cache,
                            'launch':None}
    gen_experiment_mapping = {'gen_fixed_experiment':gen_fixed_experiment,
                              'gen_rand_experiment':gen_rand_experiment,
                              'gen_clustered_rand_experiment':gen_clustered_rand_experiment,}
    
    args = parser.parse_args()
    
    args.get_function = get_function_mapping[args.get_function]
    args.gen_experiment = gen_experiment_mapping[args.gen_experiment]
    args.launch_group = (args.launch_group.lower() == 'true')
    args.terminate_after = (args.terminate_after.lower() == 'true')
    
    initialize(args)
    
    print()
    if args.get_function is None:
        print("sleeping")
        sleep(120)
    else:
        benchmark_broadcast(keys=KEYS, **vars(args))
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
        
