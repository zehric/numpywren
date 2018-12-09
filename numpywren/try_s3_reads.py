import boto3
import time
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

    a, b = get_object_backoff(client, key_list[0], bucket)
    a['Body'].read()
    
    for key in key_list:                                             
        s = time.time()                                              
        obj, num_tries = get_object_backoff(client, key, bucket)     
        objects.append(obj['Body'].read())
        f = time.time()                                              
        # objects.append(data)                                         
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

# ret_dict = (read_keys_no_cache("zehric-pywren-149", 0, ['broadcast_benchmark/002499955e221506c691b2686b91a147d9676f47',
#     'broadcast_benchmark/005a517624a1f7ed8104b6f5f6c66c8f3adfac71',
#     'broadcast_benchmark/01dfcddf081768211a2990d709913f5c02a102e0',
#     'broadcast_benchmark/03986c48def5af46dabd256c62afdf0503d336c7',
#     'broadcast_benchmark/03f5a655445e95766df43fe054ee45f3a035484a']))
# ret_dict = (read_keys_no_cache("zehric-pywren-149", 0, ['broadcast_benchmark/002499955e221506c691b2686b91a147d9676f47'
#     ]))

# ret_dict = (read_keys_no_cache("uric-cache-benchmarks", 0, ['000019a88427b58592085c28233ceb3072bee950',
#     '5000082b17a31b06e2c88c9df2e923d358188413',
#     '300014a591bc7c9dbe793cd6c2971c485bcfc2b7',
#     '7000014c25c15fe0306acd574d1aafeab1f0991e',
#     '90000b22f4e87ed0024308fb404899b8b1c4d575']))

ret_dict = (read_keys_no_cache("zehric-pywren-149", 0, ['broadcast_benchmark--0/002499955e221506c691b2686b91a147d9676f47',
    'broadcast_benchmark/002499955e221506c691b2686b91a147d9676f47']))
ret_dict2 = (read_keys_no_cache("zehric-pywren-149", 0, ['broadcast_benchmark--0/002499955e221506c691b2686b91a147d9676f47',
    'broadcast_benchmark/002499955e221506c691b2686b91a147d9676f47']))

for s, f in ret_dict['time_gets']:
    print(f - s)
for s, f in ret_dict2['time_gets']:
    print(f - s)
