
# 4096*4096*10*10*8 + 128 = 13421772928
# 4096*4096*8 + 128 = 134217856


BLOCK_SIZE=4096
NUM_BLOCK_ROWS=10
declare -a NUM_INSTANCES=("32" "16" "8" "4" "1")
declare -a NUM_CORES_PER_INSTANCE=("1" "2" "4" "8" "32")
declare -a NUM_TRIALS=("0") # "1" "2" "3" "4" "5" "6" "7" "8" "9")
declare -a RUN_FUNCS=("cholesky") # "read_keys_with_cache")
declare -a USE_CACHE=("yes" "no")
SAVE_FOLDER=with_cache_cholesky




for trial_number in "${NUM_TRIALS[@]}"
do
    for index in ${!NUM_CORES_PER_INSTANCE[*]}; do                                               
        num_cores_per_instance=${NUM_CORES_PER_INSTANCE[$index]}                                                   
        num_instances=${NUM_INSTANCES[$index]}                                                                                                    
        total_to_launch=1   
        echo $BLOCK_SIZE $NUM_BLOCK_ROWS $num_instances $num_cores_per_instance False False $trial_number $SAVE_FOLDER                                                                                       
        python3 test_numpywren_cache.py $BLOCK_SIZE $NUM_BLOCK_ROWS $num_instances $num_cores_per_instance True False $trial_number launch $SAVE_FOLDER False
        echo Sleeping for 150 seconds
        sleep 150
        echo Waking up
        for run_func in "${RUN_FUNCS[@]}"                                            
        do
            for use_cache in "${USE_CACHE[@]}"
            do                                                                      
                echo $BLOCK_SIZE $NUM_BLOCK_ROWS $num_instances $num_cores_per_instance False False $trial_number $run_func $SAVE_FOLDER $use_cache
                python3 test_numpywren_cache.py $BLOCK_SIZE $NUM_BLOCK_ROWS $num_instances $num_cores_per_instance False False $trial_number $run_func $SAVE_FOLDER $use_cache
                                                            
                total_to_launch="$(($total_to_launch-1))"                           
                if [ "$total_to_launch" == "0" ]; then                              
                    #echo $total_to_launch                                          
                    pywren standalone terminate_instances                           
                fi            
            done 
        done                                                                    
    done
done


