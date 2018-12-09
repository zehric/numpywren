#/bin/bash

pywren standalone launch_instances 25 --spot_price 0.0 --parallelism 36
sleep 60
python3 s3iops.py 1e6 1000 threaded
pywren standalone terminate_instances
pywren standalone purge_queue
aws s3 rm s3://uric-cache-benchmarks --recursive

pywren standalone launch_instances 25 --spot_price 0.0 --parallelism 36
sleep 60

for method in async threaded
do
	for data_size in 1e6 1e7
	do
		for num_keys in 10 25 50 100
		do
			python3 s3iops.py $data_size $num_keys $method
		done
	done
done
pywren standalone terminate_instances
aws s3 rm s3://uric-cache-benchmarks --recursive
