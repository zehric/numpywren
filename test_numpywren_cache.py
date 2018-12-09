import argparse



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

import multiprocessing
from pywren import ec2standalone






def run_program_in_pywren(program, num_instances, num_cores):
    def pywren_run(_):
        job_runner.lambdapack_run(program, timeout=60, idle_timeout=6)
    default_npw_config = npw.config.default()
    pywren_config = wc.default()
    npw_config = npw.config.default()
    pywren_config['runtime']['s3_bucket'] = npw_config['runtime']['bucket']
    pywren_config['runtime']['s3_key'] = npw_config['runtime']['s3_key']
    pwex = get_executor(num_cores)
    futures = pwex.map(pywren_run, range(num_instances*num_cores))
    return futures



def benchmark_function(num_instances, 
                       num_cores, 
                       run_func,
                       args):
    program, meta =  run_func(*args)
    print("Running: {0}".format(run_func.__name__))
    t = time.time()
    futures = run_program_in_pywren(program, num_instances, num_cores)
    program.start()
    program.wait()
    program.free()
    e = time.time()
    ret_dict = {}
    ret_dict['total_time_te'] = (t, e)
    ret_dict['total_time'] = e-t
    np.save("{0}/{1}.npy".format(RESULT_DIR, OUTPUT_NAME()), result_dict)




# def test_tsqr_lambda():
#     np.random.seed(1)
#     size = 256
#     shard_size = 32
#     X = np.random.randn(size, shard_size)
#     Q,R = np.linalg.qr(X)
#     q0, r0 = np.linalg.qr(X[:2,:2])
#     q1, r1 = np.linalg.qr(X[2:,:2])
#     r2 = np.linalg.qr(np.vstack((r0,r1)))[1]
#     shard_sizes = (shard_size, X.shape[1])
#     X_sharded = BigMatrix("tsqr_test_X", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
#     shard_matrix(X_sharded, X)
#     program, meta = tsqr(X_sharded)
#     executor = fs.ProcessPoolExecutor(1)
#     print("starting program")
#     program.start()
#     futures = run_program_in_pywren(program)
#     program.wait()
#     program.free()
#     R_sharded = meta["outputs"][0]
#     num_tree_levels = int(np.log(np.ceil(size/shard_size))/np.log(2))
#     print("num_tree_levels", num_tree_levels)
#     R_npw = R_sharded.get_block(max(num_tree_levels, 0), 0)
#     sign_matrix_local = np.eye(R.shape[0])
#     sign_matrix_remote = np.eye(R.shape[0])
#     sign_matrix_local[np.where(np.diag(R) <= 0)]  *= -1
#     sign_matrix_remote[np.where(np.diag(R_npw) <= 0)]  *= -1
#     # make the signs match
#     R_npw *= np.diag(sign_matrix_remote)[:, np.newaxis]
#     R  *= np.diag(sign_matrix_local)[:, np.newaxis]
#     assert(np.allclose(R_npw, R))



# def test_gemm_lambda():
#     size = 32
#     A = np.random.randn(size, size)
#     B = np.random.randn(size, size)
#     C = np.dot(A, B)
#     shard_sizes = (8,8)
#     A_sharded = BigMatrix("Gemm_test_A", shape=A.shape, shard_sizes=shard_sizes, write_header=True)
#     B_sharded = BigMatrix("Gemm_test_B", shape=A.shape, shard_sizes=shard_sizes, write_header=True)
#     shard_matrix(A_sharded, A)
#     shard_matrix(B_sharded, B)
#     program, meta = gemm(A_sharded, B_sharded)
#     executor = fs.ProcessPoolExecutor(1)
#     program.start()
#     run_program_in_pywren(program)
#     program.wait()
#     program.free()
#     C_sharded = meta["outputs"][0]
#     C_npw = C_sharded.numpy()
#     assert(np.allclose(C_npw, C))
#     return


# def test_qr_lambda():
#     N = 16
#     shard_size = 4
#     shard_sizes = (shard_size, shard_size)
#     X = np.random.randn(N, N)
#     X_sharded = BigMatrix("QR_input_X", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
#     N_blocks = X_sharded.num_blocks(0)
#     shard_matrix(X_sharded, X)
#     program, meta = qr(X_sharded)
#     program.start()
#     print("starting program...")
#     futures = run_program_in_pywren(program)
#     program.wait()
#     program.free()
#     Rs = meta["outputs"][0]
#     R_remote = Rs.get_block(N_blocks - 1, N_blocks - 1, 0)
#     R_local = np.linalg.qr(X)[1][-shard_size:, -shard_size:]
#     sign_matrix_local = np.eye(R_local.shape[0])
#     sign_matrix_remote = np.eye(R_local.shape[0])
#     sign_matrix_local[np.where(np.diag(R_local) <= 0)]  *= -1
#     sign_matrix_remote[np.where(np.diag(R_remote) <= 0)]  *= -1
#     # make the signs match
#     R_remote *= np.diag(sign_matrix_remote)[:, np.newaxis]
#     R_local  *= np.diag(sign_matrix_local)[:, np.newaxis]
#     assert(np.allclose(R_local, R_remote))




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
    X2 = np.random.randn(problem_size, 1)
    shard_sizes2 = [shard_size, 1]
    if num_rows_per_block != 0: 
        shard_sizes = [num_rows_per_block, 1]
        sr = num_rows_per_block
    if '{0}_{1}_{2}'.format(nr,nc,sr) not in bigm_descs:
        X_sharded = BigMatrix("rp_local_test_{0}_{1}--1".format(problem_size, shard_size),
                              shape=X.shape, shard_sizes=shard_sizes,
                              write_header=True,
                              autosqueeze=False,
                              bucket=DEFAULT_BUCKET,
                              parent_fn=matrix_utils.constant_zeros,
                              use_cache=False)
        X_sharded2 = BigMatrix("rp_local_test_{0}_{1}--2".format(problem_size, shard_size),
                              shape=X2.shape, shard_sizes=shard_sizes2,
                              write_header=True,
                              autosqueeze=False,
                              bucket=DEFAULT_BUCKET,
                              parent_fn=matrix_utils.constant_zeros,
                              use_cache=False)
        shard_matrix(X_sharded, X)
        shard_matrix(X_sharded2, X2)

        t = time.time()
        print("------------------------------(X_sharded.shape) = ", X_sharded.shape)
        print("------------------------------(X_sharded2.shape) = ", X_sharded2.shape)
        ret = binops.gemm(pwex, X_sharded, X_sharded2.T, overwrite=False, gemm_chunk_size=16, local=True)
        e = time.time()
        print("------------------------------GEMM took {0} seconds".format(e - t))
        print("------------------------------ret.shape={0}".format(ret.shape))
        print("------------------------------ret.shard_sizes={0}".format(ret.shard_sizes))
        bigm_descs['{0}_{1}_{2}'.format(nr,nc,sr)] = {'key':ret.key, 'bucket':config['s3']['bucket'], 'region':config['account']['aws_region']}
        np.save(shard_mats, bigm_descs)


    return get_bigm(bigm_descs['{0}_{1}_{2}'.format(nr,nc,sr)], args)

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
    

    
    OUTPUT_NAME = lambda : "{0}_{1}_{2}--{3}".format(args.run_func, BLOCK_SIZE, NUM_BLOCK_ROWS, TRIAL_NUMBER)
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
    parser.add_argument('trial_number', type=str)
    parser.add_argument('run_func', type=str)
    parser.add_argument('save_folder', type=str)
    parser.add_argument('use_cache', type=str)

    run_func_mapping = {'cholesky' : cholesky,
                        'tsqr'     : tsqr,
                        'gemm'     : gemm,
                        'qr'       : qr,
                        'launch'   : None}

    args = parser.parse_args()
    args.use_cache = (args.use_cache.lower() == 'true')

    print(args)

    print("Initializing")
    initialize(args)


    args.run_func = run_func_mapping[args.run_func]


    
    if args.run_func is not None:
        s = time.time()
        benchmark_function(num_instances=NUM_INSTANCES, num_cores=NUM_CORES, run_func=args.run_func, args=[MAT])
        e = time.time()
        print(s, e, (e-s))


