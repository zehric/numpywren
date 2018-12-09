from multiprocessing import Process
from multiprocessing.connection import Client
import numpy as np
import os
import time

def client_run(id_, num_repeat, log_info):
    tick = time.time()
    key_pre = "test{}"
    client = Client(address='local_cache')
    for i in range(10):
        key = key_pre.format(i)
        randarr = np.full((4096, 4096), i, dtype='float64')
        arr = np.memmap("/tmp/" + key, dtype='float64', mode='w+', shape=(4096,4096))
        arr[:] = randarr[:]
        start_time = time.time()
        client.send((1, key)) # put
        res = client.recv()
        print("put {}: {}".format(i, time.time() - start_time))

    for i in range(10):
        key = key_pre.format(i)
        start_time = time.time()
        client.send((0, key, True)) # get and pin
        res = client.recv()
        arr = np.memmap(res, dtype='float64', mode='r', shape=(4096,4096))
        print(arr)
        print("get {}: {}".format(i, time.time() - start_time))
        client.send((2, key)) # release
        res = client.recv()

    for i in range(10):
        key = key_pre.format(i)
        start_time = time.time()
        client.send((0, key)) # get and pin
        res = client.recv()
        arr = np.frombuffer(res, dtype='float64').reshape((4096, 4096))
        print(arr)
        print("get {}: {}".format(i, time.time() - start_time))
    # for _ in range(num_repeat):
    #     client.send((0, "test{}".format(id_)))
    #     get = client.recv()
    #     if cache_funcs is not None:
    #         get = cache_funcs['get'](get)
    #     print(get.shape)
    tock = time.time()
    log_info[id_] = (tock-tick)


if __name__ == '__main__':
    num_clients = 1
    num_repeat = 10
    # data_size = int(1e6)
    # use_shared_data = True

    # if use_shared_data:
    #     data = np.zeros(data_size, dtype=np.float64)
    #     np.save('local_cache_%s.dat'%'mat',data)
    #     data = 'local_cache_%s.dat'%'mat'
    #     cache_funcs = {'set':lambda k,v: np.save('local_cache_%s.dat'%k,data),
    #                    'get': lambda k: np.load('%s.npy'%k,mmap_mode='r+')}
    # else:
    #     data = np.zeros(data_size, dtype=np.float64)
    #     cache_funcs = None
    # init_data = {'mat':data}
    # exit_pipe = Client(address='local_cache')
    # for k,v in init_data.items():
    #     cache.put(k, v)

    # mmap_benchmark = np.memmap(filename=os.path.join(mkdtemp(), 'benchmark_reads.dat'),
    #                            dtype='float32',
    #                            mode='w+',
    #                            shape=(num_clients,))

    clients = []

    for i in range(num_clients):
        clients.append(Process(target=client_run, args=(i,num_repeat,{})))

    tick = time.time()
    for client in clients:
        client.start()

    for client in clients:
        client.join()

    # special pipe for terminating the cache

    # exit_pipe.send(('exit',))
    tock = time.time()


    print("outer time = %0.4f seconds" % (tock-tick))
    #print(mmap_benchmark)
    # print("mean       = %0.4f seconds" % np.mean(mmap_benchmark))
    #print("variance   = %0.4f seconds" % np.var(mmap_benchmark))

