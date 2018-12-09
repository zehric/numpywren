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



declare -a num_trials=("rand0" "rand1" "rand2" "rand3" "rand4" "rand5" "rand6" "rand7" "rand8" "rand9")
for trial in "${num_trials[@]}"
do
    for index in ${!num_cores[*]}; do                                               
        n_ci=${num_cores[$index]}                                                   
        n_ii=${num_instances[$index]}                                               
        launch_group=False                                                           
        terminate_after=False                                                       
        total_to_launch=2 # len(experiments) * len(run_func)  
                                                                             
        python3 broadcast_benchmark_script.py launch $DATA_SIZE $NUM_KEYS $n_ii $n_ci $NUM_KEYS_PER_CORE gen_rand_experiment True False 0
        echo Sleeping
        sleep 120
        echo Waking up
        for r_fi in "${run_func[@]}"                                            
        do                                                                      
            #echo $r_fi $DATA_SIZE $NUM_KEYS $n_ii $n_ci $NUM_KEYS_PER_CORE $exi $launch_group $terminate_after
            python3 broadcast_benchmark_script.py $r_fi $DATA_SIZE $NUM_KEYS $n_ii $n_ci $NUM_KEYS_PER_CORE gen_rand_experiment False False $trial                                                     
                                                                 

        done
        pywren standalone terminate_instances
                                                                      
    done
done

