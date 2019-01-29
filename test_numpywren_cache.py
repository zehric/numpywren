import argparse

import boto3

import time
import random
from timeit import default_timer as timer
import string
import concurrent.futures as fs
import os

from numpywren import compiler, job_runner, kernels
from numpywren.matrix import BigMatrix, DEFAULT_BUCKET
from numpywren.alg_wrappers import cholesky, tsqr, gemm, qr
from numpywren import matrix_utils
from numpywren import binops
import numpy as np
from numpywren.matrix_init import shard_matrix
from numpywren.algs import *
from numpywren.compiler import lpcompile, walk_program, find_parents, find_children
import numpywren as npw
import pywren
import pywren.wrenconfig as wc
from numpywren import lambdapack as lp

import multiprocessing
from pywren import ec2standalone
import concurrent.futures as fs





def run_program_in_pywren(program, num_instances, num_cores):
    def pywren_run(_):
        job_runner.lambdapack_run(program, timeout=3600, idle_timeout=120)
    default_npw_config = npw.config.default()
    pywren_config = wc.default()
    npw_config = npw.config.default()
    pywren_config['runtime']['s3_bucket'] = npw_config['runtime']['bucket']
    pywren_config['runtime']['s3_key'] = npw_config['runtime']['s3_key']
    
    
    pwex = get_executor(num_cores)
    futures = pwex.map(pywren_run, range(num_instances*num_cores), extra_env=EXTRA_ENV)

    # executor = fs.ProcessPoolExecutor(num_cores)
    # program.start()
    # futures = executor.submit(job_runner.lambdapack_run, program, timeout=3600, idle_timeout=30)
    return futures

def parse_int(x):
    if x is None: return 0
    return int(x)

def benchmark_function(num_instances, 
                       num_cores, 
                       trial,
                       run_func,
                       args):
    print(pcolor.FAIL, end="")
    print("Running: {0}".format(run_func.__name__))
    print(pcolor.ENDC)

    pwex = get_executor(num_cores)

    t = time.time()
    program, meta = run_func(*args)
    L_sharded = meta["outputs"][0]
    pipeline_width = num_cores

    pywren_config = pwex.config
    e = time.time()
    print("Program compile took {0} seconds".format(e - t))
    print("program.hash", program.hash)
    REDIS_CLIENT = program.control_plane.client
    done_counts = []
    ready_counts = []
    post_op_counts = []
    not_ready_counts = []
    running_counts = []
    sqs_invis_counts = []
    sqs_vis_counts = []
    up_workers_counts = []
    busy_workers_counts = []
    read_objects = []
    write_objects = []
    all_read_timeouts = []
    all_write_timeouts = []
    all_redis_timeouts = []
    times = [time.time()]
    flops = [0]
    reads = [0]
    writes = [0]
    lru=False
    eager=False
    standalone=True
    log_granularity = 5
    print("LRU", lru)
    print("eager", eager)
    exp = {}
    exp["redis_done_counts"] = done_counts
    exp["redis_ready_counts"] = ready_counts
    exp["redis_post_op_counts"] = post_op_counts
    exp["redis_not_ready_counts"] = not_ready_counts
    exp["redis_running_counts"] = running_counts
    exp["sqs_invis_counts"] = sqs_invis_counts
    exp["sqs_vis_counts"] = sqs_vis_counts
    exp["busy_workers"] = busy_workers_counts
    exp["up_workers"] = up_workers_counts
    exp["times"] = times
    exp["problem_size"] = MAT.shape
    exp["shard_size"] = MAT.shard_sizes
    exp["flops"] = flops
    exp["reads"] = reads
    exp["writes"] = writes
    exp["read_objects"] = read_objects
    exp["write_objects"] = write_objects
    exp["read_timeouts"] = all_read_timeouts
    exp["write_timeouts"] = all_write_timeouts 
    exp["redis_timeouts"] = all_redis_timeouts 
    exp["trial"] = trial
    exp["standalone"] = standalone
    exp["time_steps"] = 1
    exp["failed"] = False
    exp["log_granularity"] = log_granularity
    INFO_FREQ = 10

    program.start()
    t = time.time()
    all_futures = pwex.map(lambda x: job_runner.lambdapack_run(program, pipeline_width=1, timeout=3600), range(NUM_CORES*NUM_INSTANCES), extra_env=EXTRA_ENV)
    start_time = time.time()
    last_run_time = start_time
    print(program.program_status())
    print("QUEUE URLS", len(program.queue_urls))
    total_lambda_epochs = NUM_CORES
    try:
        while(program.program_status() == lp.PS.RUNNING):
            time.sleep(log_granularity)
            curr_time = int(time.time() - start_time)
            p = program.get_progress()
            if (p is None):
                print("no progress...")
                continue
            else:
               p = int(p)
            times.append(int(time.time()))
            max_pc = p
            waiting = 0
            running = 0
            for i, queue_url in enumerate(program.queue_urls):
                client = boto3.client('sqs')
                attrs = client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible'])['Attributes']
                waiting += int(attrs["ApproximateNumberOfMessages"])
                running += int(attrs["ApproximateNumberOfMessagesNotVisible"])
            sqs_invis_counts.append(running)
            sqs_vis_counts.append(waiting)
            busy_workers = REDIS_CLIENT.get("{0}_busy".format(program.hash))
            sparse_writes  = parse_int(REDIS_CLIENT.get("{0}_write_sparse".format(program.hash)))/1e9

            if (busy_workers == None):
                busy_workers = 0
            else:
                busy_workers = int(busy_workers)
            up_workers = program.get_up()

            if (up_workers == None):
                up_workers = 0
            else:
                up_workers = int(up_workers)
            up_workers_counts.append(up_workers)
            busy_workers_counts.append(busy_workers)

            #print("{2}: Up Workers: {0}, Busy Workers: {1}".format(up_workers, busy_workers, curr_time))
            #if ((curr_time % INFO_FREQ) == 0):
            #    print("Waiting: {0}, Currently Processing: {1}".format(waiting, running))
            #    print("{2}: Up Workers: {0}, Busy Workers: {1}".format(up_workers, busy_workers, curr_time))

            current_gflops = program.get_flops()
            if (current_gflops is None):
                current_gflops = 0
            else:
                current_gflops = int(current_gflops)/1e9

            flops.append(current_gflops)
            current_gbytes_read = program.get_read()
            if (current_gbytes_read is None):
                current_gbytes_read = 0
            else:
                current_gbytes_read = int(current_gbytes_read)/1e9

            reads.append(current_gbytes_read)
            current_gbytes_write = program.get_write()
            if (current_gbytes_write is None):
                current_gbytes_write = 0
            else:
                current_gbytes_write = int(current_gbytes_write)/1e9
            writes.append(current_gbytes_write)

            gflops_rate = flops[-1]/(times[-1] - times[0])
            greads_rate = reads[-1]/(times[-1] - times[0])
            gwrites_rate = writes[-1]/(times[-1] - times[0])
            b = args[0].shard_sizes[0]
            current_objects_read = (current_gbytes_read*1e9)/(b*b*8)
            current_objects_write = (current_gbytes_write*1e9)/(b*b*8)
            read_objects.append(current_objects_read)
            write_objects.append(current_objects_write)
            read_rate = read_objects[-1]/(times[-1] - times[0])
            write_rate = write_objects[-1]/(times[-1] - times[0])

            avg_workers = np.mean(up_workers_counts)
            smooth_len = 10
            if (len(flops) > smooth_len + 5):
                gflops_rate_5_min_window = (flops[-1] - flops[-smooth_len])/(times[-1] - times[-smooth_len])
                gread_rate_5_min_window = (reads[-1] - reads[-smooth_len])/(times[-1] - times[-smooth_len])
                gwrite_rate_5_min_window = (writes[-1] - writes[-smooth_len])/(times[-1] - times[-smooth_len])
                read_rate_5_min_window = (read_objects[-1] - read_objects[-smooth_len])/(times[-1] - times[-smooth_len])
                write_rate_5_min_window = (write_objects[-1] - write_objects[-smooth_len])/(times[-1] - times[-smooth_len])
                workers_5_min_window = np.mean(up_workers_counts[-smooth_len:])
            else:
                gflops_rate_5_min_window =  "N/A"
                gread_rate_5_min_window = "N/A"
                gwrite_rate_5_min_window = "N/A"
                workers_5_min_window = "N/A"
                read_rate_5_min_window = "N/A"
                write_rate_5_min_window = "N/A"


            read_timeouts = int(parse_int(REDIS_CLIENT.get("s3.timeouts.read")))
            write_timeouts = int(parse_int(REDIS_CLIENT.get("s3.timeouts.write")))
            redis_timeouts = int(parse_int(REDIS_CLIENT.get("redis.timeouts")))
            all_read_timeouts.append(read_timeouts)
            all_write_timeouts.append(write_timeouts)
            all_redis_timeouts.append(redis_timeouts)
            read_timeouts_fraction = read_timeouts/current_objects_read
            write_timeouts_fraction = write_timeouts/current_objects_write
            print("=======================================")
            print("Max PC is {0}".format(max_pc))
            print("Waiting: {0}, Currently Processing: {1}".format(waiting, running))
            print("{2}: Up Workers: {0}, Busy Workers: {1}".format(up_workers, busy_workers, curr_time))
            print("{0}: Total GFLOPS {1}, Total GBytes Read {2}, Total GBytes Write {3}, Total Gbytes Write Sparse : {4}".format(curr_time, current_gflops, current_gbytes_read, current_gbytes_write, sparse_writes))
            print("{0}: Average GFLOPS rate {1}, Average GBytes Read rate {2}, Average GBytes Write  rate {3}, Average Worker Count {4}".format(curr_time, gflops_rate, greads_rate, gwrites_rate, avg_workers))
            print("{0}: Average read txns/s {1}, Average write txns/s {2}".format(curr_time, read_rate, write_rate))
            print("{0}: smoothed GFLOPS rate {1}, smoothed GBytes Read rate {2}, smoothed GBytes Write  rate {3}, smoothed Worker Count {4}".format(curr_time, gflops_rate_5_min_window, gread_rate_5_min_window, gwrite_rate_5_min_window, workers_5_min_window))
            print("{0}: smoothed read txns/s {1}, smoothed write txns/s {2}".format(curr_time, read_rate_5_min_window, write_rate_5_min_window))
            print("{0}: Read timeouts: {1}, Write timeouts: {2}, Redis timeouts: {3}  ".format(curr_time, read_timeouts, write_timeouts, redis_timeouts))
            print("{0}: Read timeouts fraction: {1}, Write timeouts fraction: {2}".format(curr_time, read_timeouts_fraction, write_timeouts_fraction))
            print("=======================================")

            time_since_launch = time.time() - last_run_time

            exp["time_steps"] += 1

    except KeyboardInterrupt:
        exp["failed"] = True
        program.stop()
        pass
    except Exception as e:
        traceback.print_exc()
        exp["failed"] = True
        program.stop()
        raise
        pass
    print(program.program_status())
    #exp["all_futures"] = all_futures
    #exp_bytes = dill.dumps(exp)
    #client = boto3.client('s3')
    #client.put_object(Key="lambdapack/{0}/runtime.pickle".format(program.hash), Body=exp_bytes, Bucket=program.bucket)
    print("=======================")
    print("=======================")
    print("Execution Summary:")
    print("Executed Program ID: {0}".format(program.hash))
    print("Program Success: {0}".format((not exp["failed"])))
    print("Problem Size: {0}".format(exp["problem_size"]))
    print("Shard Size: {0}".format(exp["shard_size"]))
    print("Total Execution time: {0}".format(times[-1] - times[0]))
    print("Average Flop Rate (GFlop/s): {0}".format(exp["flops"][-1]/(1e-6+times[-1] - times[0])))


    
    # t_all = time.time()
    # program, meta =  run_func(*args)
    # t_program = time.time()
    # futures = run_program_in_pywren(program, num_instances, num_cores)
    # program.start()
    # t = time.time()
    # program.start()
    # program.wait()
    # e = time.time()
    # time.sleep(5)
    # #program.free()
    # L_sharded = meta["outputs"][0]
    

    ret_dict = {}
    ret_dict['num_cores_per_instance'] = NUM_CORES
    ret_dict['num_instances'] = NUM_INSTANCES
    ret_dict['block_size'] = BLOCK_SIZE
    ret_dict['block_mat_size'] = NUM_BLOCK_ROWS 
    ret_dict['use_cache'] = USE_CACHE
    ret_dict['function'] = run_func.__name__
    ret_dict.update(exp)
    np.save("{0}/{1}.npy".format(RESULT_DIR, OUTPUT_NAME()), ret_dict)


    print(pcolor.FAIL, end="")
    print("Finished and saved {0}".format("{0}/{1}.npy".format(RESULT_DIR, OUTPUT_NAME())))
    print(pcolor.ENDC)






def gen_instance_unique_name(base, instance_id):
    return "numpywren_test"

def launch_instance_group(num_instances, num_cores, cache_size=40):
    assert (num_cores in STANDALONE_INSTANCES)
    kwargs = {'ec2_instance_type':STANDALONE_INSTANCES[num_cores]['name'],
              'target_ami':STANDALONE_INSTANCES[num_cores]['target_ami'],
              'parallelism':num_cores,
              'cache_size':cache_size
              }
    inst_list = launch_many_instances(num_instances, **kwargs)
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
        #config['standalone']['sqs_queue_name'] = gen_instance_unique_name(sqs, i)
        #create_sqs_queue(config['standalone']['sqs_queue_name'])
        # inst_list += launch_instances(config, **default_extra_args)
        p = multiprocessing.Process(target=launch_instances, args=(config, default_extra_args, i))
        inst_list.append(p)
        p.start()
        time.sleep(1)
    for p in inst_list:
        p.join()
def launch_instances(config, kwargs, i=1):
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
                                               "{0}-{1}".format(sc['instance_name'], i),
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



def get_executor(num_cores, job_max_runtime=3600):
    assert (num_cores in STANDALONE_INSTANCES)
    config = CONFIG()
    #config['standalone']['sqs_queue_name'] = gen_instance_unique_name(config['standalone']['sqs_queue_name'], i)
    config['standalone']['ec2_instance_type'] = STANDALONE_INSTANCES[num_cores]['name']
    config['standalone']['target_ami'] = STANDALONE_INSTANCES[num_cores]['target_ami']
    config['standalone']['parallelism'] = num_cores
    return pywren.standalone_executor(config=config)

def get_bigm(bigm_descs, args):
    bigm = BigMatrix(key=bigm_descs['key'],
            bucket=bigm_descs['bucket'],
            prefix='numpywren.objects/',
            parent_fn=matrix_utils.constant_zeros,
            write_header=False,
            autosqueeze=True,
            lambdav=0.0,
            region=bigm_descs['region'],
            use_cache=args.use_cache)
    return bigm

def gen_sharded_mat(pwex, config, problem_size, shard_size, num_rows_per_block, save_path, args):
    shard_mats = save_path
    if os.path.exists(shard_mats):
        bigm_descs = np.load(shard_mats).item()
    else:
        bigm_descs = {}

    np.random.seed(0)
    nr, nc, sr = problem_size, shard_size, shard_size
    

    X = np.random.randn(problem_size, 1)
    shard_sizes = [shard_size, 1]
    if '{0}_{1}_{2}'.format(nr,nc,sr) not in bigm_descs:
        X_sharded = BigMatrix("rp_local_test_{0}_{1}".format(problem_size, shard_size),
                              shape=X.shape, shard_sizes=shard_sizes,
                              write_header=True,
                              autosqueeze=False,
                              bucket=DEFAULT_BUCKET,
                              parent_fn=matrix_utils.constant_zeros,
                              use_cache=False)
        shard_matrix(X_sharded, X)

        t = time.time()
        print("------------------------------(X_sharded.shape) = ", X_sharded.shape)
        ret = binops.gemm(pwex, X_sharded, X_sharded.T, overwrite=False, gemm_chunk_size=16, local=True)
        e = time.time()
        print("------------------------------GEMM took {0} seconds".format(e - t))
        print("------------------------------ret.shape={0}".format(ret.shape))
        print("------------------------------ret.shard_sizes={0}".format(ret.shard_sizes))
        bigm_descs['{0}_{1}_{2}'.format(nr,nc,sr)] = {'key':ret.key, 'bucket':config['s3']['bucket'], 'region':config['account']['aws_region']}
        np.save(shard_mats, bigm_descs)


    mat = get_bigm(bigm_descs['{0}_{1}_{2}'.format(nr,nc,sr)], args)
    mat.lambdav = problem_size*20e12
    return mat

def initialize(args):
    global BLOCK_SIZE, NUM_BLOCK_ROWS, NUM_INSTANCES, NUM_CORES, TRIAL_NUMBER, \
           SAVE_FOLDER, OUTPUT_NAME ,BENCHMARK_DIR, FIGURE_DIR, MAT_DIR, RESULT_DIR, \
           EXTRA_ENV, CONFIG, PYWREN_BUCKET, STANDALONE_INSTANCES, MAT
    
    BLOCK_SIZE = args.block_size
    NUM_BLOCK_ROWS = args.num_block_rows
    NUM_INSTANCES = args.num_instances
    NUM_CORES = args.num_cores_per_instance
    TRIAL_NUMBER = args.trial_number
    SAVE_FOLDER = args.save_folder
    

    
    OUTPUT_NAME = lambda : "{0}_wc_{1}__{2}_{3}_{4}_{5}--{6}".format(args.run_func, args.use_cache, BLOCK_SIZE, NUM_BLOCK_ROWS, NUM_INSTANCES, NUM_CORES, TRIAL_NUMBER)
    print(OUTPUT_NAME())
    BENCHMARK_DIR = SAVE_FOLDER
    FIGURE_DIR = '{0}/figures'.format(BENCHMARK_DIR)
    MAT_DIR = '{0}/mat_data'.format(BENCHMARK_DIR)
    RESULT_DIR = '{0}/results'.format(BENCHMARK_DIR)
    for d in [FIGURE_DIR, MAT_DIR, RESULT_DIR]:
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
                            # 20 : {'name':'m4.10xlarge','target_ami':'ami-0bb5806b2e825a199'},
                            32 : {'name':'m4.16xlarge','target_ami':'ami-0bb5806b2e825a199'},
                            36 : {'name':'c5.18xlarge','target_ami':'ami-a0cfeed8'}
                           } 
    mat_dat_path = "{0}/put_data_{1}_{2}.npy".format(MAT_DIR, BLOCK_SIZE, NUM_BLOCK_ROWS)
    region = wc.default()["account"]["aws_region"]
    extra_env = {"AWS_DEFAULT_REGION":region}
    config = wc.default()
    config_npw = npw.config.default()
    config['runtime']['s3_bucket'] = config_npw['runtime']['bucket']
    config['runtime']['s3_key'] = config_npw['runtime']['s3_key']
    pwex_l = pywren.default_executor(config=config)
    print("Getting matrix")
    MAT = gen_sharded_mat(pwex_l, config, BLOCK_SIZE*NUM_BLOCK_ROWS, BLOCK_SIZE, num_rows_per_block=BLOCK_SIZE, save_path=mat_dat_path, args=args)
    print()

    if args.launch_group:
        print(pcolor.FAIL+"Launching instances..."+pcolor.ENDC)
        launch_instance_group(NUM_INSTANCES,NUM_CORES,cache_size=40)
        print(pcolor.OKGREEN+"Finished launching."+pcolor.ENDC)
    else:
        print(pcolor.OKGREEN+"Instances already launched."+pcolor.ENDC)
    
class pcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Numpywren benchmarks")
    parser.add_argument('block_size', type=int)
    parser.add_argument('num_block_rows', type=int)
    parser.add_argument('num_instances', type=int)
    parser.add_argument('num_cores_per_instance', type=int)
    parser.add_argument('launch_group', type=str)
    parser.add_argument('terminate_after', type=str)
    parser.add_argument('trial_number', type=int)
    parser.add_argument('run_func', type=str)
    parser.add_argument('save_folder', type=str)
    parser.add_argument('use_cache', type=str)

    run_func_mapping = {'cholesky' : cholesky,
                        'tsqr'     : tsqr,
                        'gemm'     : gemm,
                        'qr'       : qr,
                        'launch'   : None}

    args = parser.parse_args()
    args.launch_group = (args.launch_group.lower() == 'true')
    args.terminate_after = (args.terminate_after.lower() == 'true')
    args.use_cache = (args.use_cache.lower() == 'true')

    global USE_CACHE
    USE_CACHE = args.use_cache

    print(pcolor.OKBLUE,end="")
    print(args,end="")
    print(pcolor.ENDC)

    print("Initializing")
    initialize(args)


    run_func = run_func_mapping[args.run_func]


    
    if run_func is not None:
        s = time.time()
        a = [MAT]
        if args.run_func == 'gemm':
            a = [MAT, MAT]
        benchmark_function(num_instances=NUM_INSTANCES, num_cores=NUM_CORES, trial=TRIAL_NUMBER, run_func=run_func, args=a)
        e = time.time()
        print(s, e, (e-s))


