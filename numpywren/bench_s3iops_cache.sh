#/bin/bash

pywren standalone launch_instances 25 --spot_price 0.0 --parallelism 36 --cache_size 900
sleep 100
python3 s3iops_cache.py 1e6 1000
pywren standalone terminate_instances || pywren standalone terminate_instances
pywren standalone purge_queue
pywren standalone launch_instances 25 --spot_price 0.0 --parallelism 36 --cache_size 90
sleep 100
python3 s3iops_cache.py 1e6 100
pywren standalone terminate_instances || pywren standalone terminate_instances
pywren standalone purge_queue
aws s3 rm s3://uric-cache-benchmarks --recursive --exclude "test*" > /dev/null
