import ctypes
from . import load_shared_lib


class FastClient(object):

    def __init__(self, so_bucket="zehric-pywren-149"):
        self.so_bucket = so_bucket
        self.so_key = "cache_client"
        self.__so_cached = None
        self.__api_started = False

    @property
    def so(self):
        if (self.__so_cached) == None:
            return load_shared_lib.load_shared_lib(key=self.so_key, bucket=self.so_bucket)
        else:
            return self.__so_cached

    def run_benchmark_pins(self, num, unique, d_ptr_s, d_ptr_f):
        so = self.so
        so.run_benchmark_pins.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)];
        return so.run_benchmark_pins(num, unique, d_ptr_s, d_ptr_f)

    def run_benchmark_gets(self, num):
        so = self.so
        so.run_benchmark_gets.argtypes = [ctypes.c_int];
        return so.run_benchmark_gets(num)

    def run_benchmark_puts(self, num, size):
        so = self.so
        so.run_benchmark_gets.argtypes = [ctypes.c_int, ctypes.c_long];
        return so.run_benchmark_gets(num, size)

    def pin_objects(self, keys, d_ptr_s, d_ptr_f, num_threads=10):
        so = self.so
        num_objects = len(keys)

        char_star_star = ctypes.c_char_p * num_objects
        c_keys_array = char_star_star(*keys)

        so.pin_objects.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        return so.pin_objects(num_objects, c_keys_array, d_ptr_s, d_ptr_f, num_threads)




    def pin_objects2(self, keys, num_threads=10):
        so = self.so
        num_objects = len(keys)

        keys = [ctypes.c_char_p(key.encode()) for key in keys]
        char_star_star = ctypes.c_char_p * num_objects
        c_keys_array = char_star_star(*keys)

        rets = [" "*50 for _ in range(len(keys))]
        rets = [ctypes.c_char_p(ret.encode()) for ret in rets]
        char_star_star2 = ctypes.c_char_p * num_objects
        c_rets_array = char_star_star2(*rets)

        so.pin_objects.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
        ret = so.pin_objects2(num_objects, c_keys_array, c_rets_array, num_threads)
        return [r.value for r in rets]

    def release_objects(self, keys, num_threads=10):
        so = self.so
        num_objects = len(keys)

        keys = [ctypes.c_char_p(key.encode()) for key in keys]
        char_star_star = ctypes.c_char_p * num_objects
        c_keys_array = char_star_star(*keys)


        so.pin_objects.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
        ret = so.release_objects(num_objects, c_keys_array, num_threads)
        return ret








    def cache_so(self):
        ''' Cache the fastio.so in memory
          NOTE: THIS WILL MAKE THIS OBJECT UNSERIALIZABLE PLEASE CALL
          .uncache_so(self) if you want to send this object over the wire
         '''
        self.__so_cached = self.so

    def uncache_so(self):
        ''' Uncache the fastio.so from memory so object is serializable
         '''
        self.__so_cached = None

# if __name__=='__main__':
#     import numpy as np
#     import random
#     fclient = FastClient(so_bucket="zehric-pywren-149")
#     fclient.cache_so()
#     key_list = open('large_keys').read().split("\n")
#     random.shuffle(key_list)
#     key_list = key_list[:10]
#     # key_list=['broadcast_benchmark/002499955e221506c691b2686b91a147d9676f47', 'broadcast_benchmark/1638d13d1ad53c5bf037faf65f44bbfe1451bd0d',
#     #         'broadcast_benchmark/3f7befecbc140a8b98fc74915294c14e394c04d3', 'broadcast_benchmark/5ac14ba7d0678f07ba592cbf87b43af247ab23e2',
#     #         'broadcast_benchmark/8e86b6f32bd537c6e8b08c214935714ebfe156b5', 'broadcast_benchmark/b3b26fe56a98837a51501f7276330086d39908a6']
#     doubles_s = np.zeros(len(key_list), np.float64)
#     doubles_f = np.zeros(len(key_list), np.float64)
#     d_ptr_s = doubles_s.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     d_ptr_f = doubles_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     keys = []
#     for key in key_list:
#         keys.append(ctypes.c_char_p(key.encode()))

#     fclient.pin_objects(keys, d_ptr_s, d_ptr_f, num_threads=1)
#     results = (list(zip(doubles_s, doubles_f)))
#     for s, f in results:
#         print(f - s)
