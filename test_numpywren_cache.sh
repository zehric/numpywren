
# 4096*4096*10*10*8 + 128 = 13421772928
# 4096*4096*8 + 128 = 134217856


BLOCK_SIZE=4096
NUM_BLOCK_ROWS=8
declare -a NUM_INSTANCES=("32" "16" "8" "4") # "1")
declare -a NUM_CORES_PER_INSTANCE=("1" "2" "4" "8") # "32")
declare -a NUM_TRIALS=("2" "3" "4" "5") # "1" "2" "3" "4" "5" "6" "7" "8" "9")
declare -a RUN_FUNCS=("cholesky" "gemm") 
declare -a USE_CACHE=("False" "True")
SAVE_FOLDER=with_cache_cholesky_take3

# python3 test_numpywren_cache.py 4096 2 1 32 True False 0 cholesky with_cache_cholesky True


for trial_number in "${NUM_TRIALS[@]}"
do
    for index in ${!NUM_CORES_PER_INSTANCE[*]}; do                                               
        num_cores_per_instance=${NUM_CORES_PER_INSTANCE[$index]}                                                   
        num_instances=${NUM_INSTANCES[$index]}  
        for run_func in "${RUN_FUNCS[@]}"                                            
        do                                                                                                  
            total_to_launch=2 
            echo $BLOCK_SIZE $NUM_BLOCK_ROWS $num_instances $num_cores_per_instance False False $trial_number $SAVE_FOLDER                                                                                       
            python3 test_numpywren_cache.py $BLOCK_SIZE $NUM_BLOCK_ROWS $num_instances $num_cores_per_instance True False $trial_number launch $SAVE_FOLDER False
            echo Sleeping for 60 seconds
            sleep 60
            echo Waking up

            # .numpywren/cpp/cache 134217856 16 zehric-pywren-149 40
            #CACHE=$!
            for use_cache in "${USE_CACHE[@]}"
            do                                                                      
                echo $BLOCK_SIZE $NUM_BLOCK_ROWS $num_instances $num_cores_per_instance False False $trial_number $run_func $SAVE_FOLDER $use_cache
                python3 test_numpywren_cache.py $BLOCK_SIZE $NUM_BLOCK_ROWS $num_instances $num_cores_per_instance False False $trial_number $run_func $SAVE_FOLDER $use_cache
                #aws s3 rm "s3://zehric-pywren-149/numpywren.objects/Cholesky.Intermediate(gemm(BigMatrix(rp_local_test_40960_4096), BigMatrix(rp_local_test_40960_4096).T(1, 40960)))/" --recursive                    
                #aws s3 rm "s3://zehric-pywren-149/numpywren.objects/Cholesky(gemm(BigMatrix(rp_local_test_40960_4096), BigMatrix(rp_local_test_40960_4096).T(1, 40960)))/" --recursive
                total_to_launch="$(($total_to_launch-1))"                           
                if [ "$total_to_launch" == "0" ]; then                              
                    #echo $total_to_launch                                          
                    pywren standalone terminate_instances                           
                fi            
            done
            #kill $CACHE
            #rm /tmp/entry-*
        done                                                                    
    done
done


