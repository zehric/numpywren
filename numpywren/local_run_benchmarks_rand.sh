# GET_FUNCTION in {read_keys_no_cache, read_keys_with_cache}
# GEN_EXPERIMENT in {gen_fixed_experiment, gen_rand_experiment, gen_clustered_rand_experiment}
# 2^20 = 1MB = 1048576

# GET_FUNCTION= str
DATA_SIZE=1048576 # int
NUM_KEYS=320 # int
# NUM_INSTANCES= int
# NUM_CORES_PER_INSTANCE= int
NUM_KEYS_PER_CORE=10 # int
# GEN_EXPERIMENT= # str
# LAUNCH_GROUP= # bool
# TERMINATE_AFTER= # bool


declare -a run_func=("read_keys_no_cache" "read_keys_with_cache")
declare -a num_cores=("1" "2" "4" "8" "32")
declare -a num_instances=("128" "64" "32" "16" "4")



python3 rand_log.py read_keys_with_cache $DATA_SIZE $NUM_KEYS 1 1 300 gen_rand_experiment False False 0
