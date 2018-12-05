/*
Copyright 2018 Vaishaal Shankar

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <aws/core/Aws.h>
#include <aws/core/Region.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <iostream>
#include <fstream>
#include <streambuf>
#include <string>
#include "bufferstream.hpp"
#include <time.h>
#include <unistd.h>
#include "threadpool.h"
#include <aws/core/utils/threading/Executor.h>
#include <climits>

#define ALLOCATION_TAG "NUMPYWREN_FASTIO"
extern "C" {
   int put_object(void* buffer, long buffer_size, const char* bucket, const char* key, double *start, double *finish);
   int get_object(void* buffer, long buffer_size, const char* bucket, const char* key, double *start, double *finish);
   int put_objects(void**obj_buffers, long num_objects, long* buffer_sizes, const char** buckets, const char** keys, int num_threads, double *start_times, double *finish_times);
   int put_objects_async(void**obj_buffers, long num_objects, long* buffer_sizes, const char** buckets, const char** keys, double *start_times, double *finish_times);
   int get_objects_async(void**obj_buffers, long num_objects, long* buffer_sizes, const char** buckets, const char** keys, double *start_times, double *finish_times);
   int get_objects(void**obj_buffers, long num_objects, long* buffer_sizes, const char** buckets, const char** keys, int num_threads, double *start_times, double *finish_times);
   void start_api();
   void stop_api();
}

typedef Aws::S3::S3Client S3Client;


static ThreadPool *pool;
static std::condition_variable cv;
static std::mutex mutex;
static int num_tasks;

class FinishContext: public Aws::Client::AsyncCallerContext
{
    private:
        double *finish;
    public:
        void SetFinish(double f) const {
            *finish = f;
        }
        FinishContext(double *f) {
            finish = f;
        }
};

int _put_object_internal(Aws::S3::S3Client &client, char* &buffer, long buffer_size, const char* bucket, const char* key, double *start, double *finish) {
    Aws::S3::Model::PutObjectRequest request;
    auto bstream = new boost::interprocess::bufferstream((char*) buffer, buffer_size);
    std::shared_ptr<Aws::IOStream> objBuffer =  std::shared_ptr<Aws::IOStream>(bstream);
    request.WithBucket(bucket).WithKey(key).SetBody(objBuffer);

    struct timespec start_t, finish_t;

    clock_gettime(CLOCK_REALTIME, &start_t);
    *start = start_t.tv_sec + ((double) start_t.tv_nsec / 1e9);
    auto put_object_response = client.PutObject(request);
    clock_gettime(CLOCK_REALTIME, &finish_t);
    *finish = finish_t.tv_sec + ((double) finish_t.tv_nsec / 1e9);
    if (!put_object_response.IsSuccess())
    {
        std::cout << "PutObject error: " <<
            put_object_response.GetError().GetExceptionName() << " " <<
            put_object_response.GetError().GetMessage() << std::endl;
        return -1;
    } else {
        return 0;
    }
}

void decrement() {
    std::unique_lock<std::mutex> lock(mutex);
    num_tasks--;
    if (num_tasks == 0) {
        cv.notify_one();
    }
}

void put_handler(const Aws::S3::S3Client *client, const Aws::S3::Model::PutObjectRequest &request, const Aws::S3::Model::PutObjectOutcome &outcome, const std::shared_ptr<const Aws::Client::AsyncCallerContext> &context) {
    struct timespec finish_t;
    auto fcontext = std::dynamic_pointer_cast<const FinishContext>(context);
    if (outcome.IsSuccess()) {
        clock_gettime(CLOCK_REALTIME, &finish_t);
        fcontext->SetFinish(finish_t.tv_sec + ((double) finish_t.tv_nsec / 1e9));
    } else {
        std::ofstream out("/tmp/" + std::to_string(rand()) + ".txt");
        out << "PutObject error: " <<
            outcome.GetError().GetExceptionName() << " " <<
            outcome.GetError().GetMessage() << std::endl;

        /* int backoff = 1000; */
        /* while (1) { */
        /*     auto put_object_response = client->PutObject(request); */
        /*     if (!put_object_response.IsSuccess()) { */
        /*         usleep(backoff); */
        /*         backoff *= rand() % (backoff * 2 + 1) + backoff; */
        /*     } else { */
        /*         clock_gettime(CLOCK_REALTIME, &finish_t); */
        /*         fcontext->SetFinish(finish_t.tv_sec + ((double) finish_t.tv_nsec / 1e9)); */
        /*         decrement(); */
        /*         return; */
        /*     } */
        /* } */
        fcontext->SetFinish(-1.0);
    }
    decrement();
}

void get_handler(const Aws::S3::S3Client *client, const Aws::S3::Model::GetObjectRequest &request, const Aws::S3::Model::GetObjectOutcome &outcome, const std::shared_ptr<const Aws::Client::AsyncCallerContext> &context) {
    struct timespec finish_t;
    auto fcontext = std::dynamic_pointer_cast<const FinishContext>(context);
    if (outcome.IsSuccess()) {
        clock_gettime(CLOCK_REALTIME, &finish_t);
        fcontext->SetFinish(finish_t.tv_sec + ((double) finish_t.tv_nsec / 1e9));
    } else {
        std::ofstream out("/tmp/" + std::to_string(rand()) + ".txt");
        out << "GetObject error: " <<
            outcome.GetError().GetExceptionName() << " " <<
            outcome.GetError().GetMessage() << std::endl;

        fcontext->SetFinish(-1.0);
    }
    decrement();
}
int _put_object_internal_async(Aws::S3::S3Client &client, char* &buffer, long buffer_size, const char* bucket, const char* key, double *start, double *finish) {
    Aws::S3::Model::PutObjectRequest request;
    auto bstream = new boost::interprocess::bufferstream((char*) buffer, buffer_size);
    std::shared_ptr<Aws::IOStream> objBuffer =  std::shared_ptr<Aws::IOStream>(bstream);
    request.WithBucket(bucket).WithKey(key).SetBody(objBuffer);

    struct timespec start_t;
    const FinishContext *context = new FinishContext(finish);
    const std::shared_ptr<const Aws::Client::AsyncCallerContext>& sh_context = std::shared_ptr<const Aws::Client::AsyncCallerContext>(context);
    clock_gettime(CLOCK_REALTIME, &start_t);
    *start = start_t.tv_sec + ((double) start_t.tv_nsec / 1e9);
    client.PutObjectAsync(request, put_handler, sh_context);
    return 0;
}

int _get_object_internal_async(Aws::S3::S3Client &client, char* &buffer, long buffer_size, const char* bucket, const char* key, double *start, double *finish) {
    Aws::S3::Model::GetObjectRequest request;
    request.WithBucket(bucket).WithKey(key);
    request.SetResponseStreamFactory(
        [buffer, buffer_size]()
        {
            return Aws::New<boost::interprocess::bufferstream>(ALLOCATION_TAG, (char*) buffer, buffer_size);
        });

    struct timespec start_t;
    const FinishContext *context = new FinishContext(finish);
    const std::shared_ptr<const Aws::Client::AsyncCallerContext>& sh_context = std::shared_ptr<const Aws::Client::AsyncCallerContext>(context);
    clock_gettime(CLOCK_REALTIME, &start_t);
    *start = start_t.tv_sec + ((double) start_t.tv_nsec / 1e9);
    client.GetObjectAsync(request, get_handler, sh_context);
    return 0;
}

int _get_object_internal(Aws::S3::S3Client &client, char* &buffer, long buffer_size, const char* bucket, const char* key, double *start, double *finish) {

    Aws::S3::Model::GetObjectRequest request;
    request.WithBucket(bucket).WithKey(key);
    request.SetResponseStreamFactory(
        [buffer, buffer_size]()
        {
            return Aws::New<boost::interprocess::bufferstream>(ALLOCATION_TAG, (char*) buffer, buffer_size);
        });
    struct timespec start_t, finish_t;

    clock_gettime(CLOCK_REALTIME, &start_t);
    *start = start_t.tv_sec + ((double) start_t.tv_nsec / 1e9);
    auto get_object_response = client.GetObject(request);
    clock_gettime(CLOCK_REALTIME, &finish_t);
    *finish = finish_t.tv_sec + ((double) finish_t.tv_nsec / 1e9);
    if (!get_object_response.IsSuccess())
    {
        std::cout << "GetObject error: " << (int) get_object_response.GetError().GetResponseCode() << " " << get_object_response.GetError().GetMessage() << std::endl;
        std::cout << "BUCKET" << bucket  << std::endl;
        std::cout << "key" << key << std::endl;
        return -1;
    } else {
        return 0;
    }
}

int put_objects_async(void**obj_buffers, long num_objects, long* buffer_sizes, const char** buckets, const char** keys, double *start_times, double *finish_times) {
    srand(time(NULL));
    num_tasks = num_objects;
    struct timespec start_t, finish_t;
    auto region = Aws::Region::US_WEST_2;
    Aws::Client::ClientConfiguration cfg;
    cfg.region = region;
    cfg.executor = Aws::MakeShared<Aws::Utils::Threading::PooledThreadExecutor>("executor", 10);
    /* cfg.maxConnections = UINT_MAX; */
    /* cfg.requestTimeoutMs = LONG_MAX; */
    /* cfg.connectTimeoutMs = LONG_MAX; */
    Aws::S3::S3Client s3_client(cfg);

    for (int i = 0; i < num_objects; i++) {
        char* buffer_to_use = (char*) obj_buffers[i];
        _put_object_internal_async(s3_client, buffer_to_use, buffer_sizes[i], buckets[i], keys[i], start_times + i, finish_times + i);
    }
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock);
}

int get_objects_async(void**obj_buffers, long num_objects, long* buffer_sizes, const char** buckets, const char** keys, double *start_times, double *finish_times) {
    srand(time(NULL));
    num_tasks = num_objects;
    struct timespec start_t, finish_t;
    auto region = Aws::Region::US_WEST_2;
    Aws::Client::ClientConfiguration cfg;
    cfg.region = region;
    cfg.executor = Aws::MakeShared<Aws::Utils::Threading::PooledThreadExecutor>("executor", 10);
    /* cfg.maxConnections = UINT_MAX; */
    /* cfg.requestTimeoutMs = LONG_MAX; */
    /* cfg.connectTimeoutMs = LONG_MAX; */
    Aws::S3::S3Client s3_client(cfg);

    for (int i = 0; i < num_objects; i++) {
        char* buffer_to_use = (char*) obj_buffers[i];
        _get_object_internal_async(s3_client, buffer_to_use, buffer_sizes[i], buckets[i], keys[i], start_times + i, finish_times + i);
    }
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock);
}

int put_objects(void**obj_buffers, long num_objects, long* buffer_sizes, const char** buckets, const char** keys, int num_threads, double *start_times, double *finish_times) {
    ThreadPool threadpool(num_threads);
    std::vector<std::future<int>> put_futures;
    for (int i = 0; i < num_objects; i++) {
        char* buffer_to_use = (char*) obj_buffers[i];
        auto future = threadpool.enqueue(put_object, buffer_to_use, buffer_sizes[i], buckets[i], keys[i], start_times + i, finish_times + i);
        put_futures.push_back(std::move(future));
    }

    for (int i = 0; i < num_objects; i++) {
        auto res = put_futures[i].get();
    }

}

int get_objects(void**obj_buffers, long num_objects, long* buffer_sizes, const char** buckets, const char** keys, int num_threads, double *start_times, double *finish_times) {
    ThreadPool threadpool(num_threads);
    std::vector<std::future<int>> get_futures;
    for (int i = 0; i < num_objects; i++) {
        char* buffer_to_use = (char*) obj_buffers[i];
        auto future = threadpool.enqueue(get_object, buffer_to_use, buffer_sizes[i], buckets[i], keys[i], start_times + i, finish_times + i);
        get_futures.push_back(std::move(future));
    }

    for (int i = 0; i < num_objects; i++) {
        auto res = get_futures[i].get();
    }
}

int put_object(void* buffer, long buffer_size, const char* bucket, const char* key, double *start, double *finish) {
    auto region = Aws::Region::US_WEST_2;
    Aws::Client::ClientConfiguration cfg;
    cfg.region = region;
    Aws::S3::S3Client s3_client(cfg);
    char* char_buffer = (char*) buffer;
    int ret = _put_object_internal(s3_client, char_buffer, buffer_size, bucket, key, start, finish);
    return ret;
}

int get_object(void* buffer, long buffer_size, const char* bucket, const char* key, double *start, double *finish) {
    auto region = Aws::Region::US_WEST_2;
    Aws::Client::ClientConfiguration cfg;
    cfg.region = region;
    Aws::S3::S3Client s3_client(cfg);
    char* char_buffer = (char*) buffer;
    int ret = _get_object_internal(s3_client, char_buffer, buffer_size, bucket, key, start, finish);
    return ret;
}

void start_api() {
    Aws::SDKOptions options;
    Aws::InitAPI(options);
}

void stop_api() {
    Aws::SDKOptions options;
    Aws::ShutdownAPI(options);
}

void init_thread_pool(int num_threads) {
    pool = new ThreadPool(num_threads);
}


/**
 * Get an object from an Amazon S3 bucket.
 */
int main(int argc, char** argv)
{
    if (argc < 5)
    {
        std::cout << std::endl <<
            "This benchmark will upload data to s3 and then download it "
            << std::endl << "" << std::endl << std::endl <<
            "Ex: fastio <objsizebytes> <num_objects> <bucketname> <prefix>\n" << std::endl;
        exit(1);
    }

    Aws::SDKOptions options;
    Aws::InitAPI(options);
    {
        /* auto objsizebytes = std::stol(argv[1]); */
        /* auto num_objects = std::stol(argv[2]); */
        /* auto bucket = std::string(argv[3]); */
        /* auto prefix = std::string(argv[4]); */

        /* std::cout << "Object Size " << argv[1] << std::endl; */
        /* std::cout << "num_objects " << argv[2] << std::endl; */
        /* std::cout << "buffersize " << objsizebytes*num_objects << std::endl; */

        /* char* buffer = (char*) malloc(objsizebytes*num_objects); */
        /* memset(buffer,7,objsizebytes*num_objects); */
        /* char* buffer2 = (char*) malloc(objsizebytes*num_objects); */
        /* memset(buffer2,8,objsizebytes*num_objects); */
        /* int cmp0 = memcmp(buffer,buffer2,objsizebytes*num_objects); */

        /* std::cout << "IO test start..."  <<  std::endl; */

        /* auto region = Aws::Region::US_WEST_2; */
        /* Aws::Client::ClientConfiguration cfg; */
        /* cfg.region = region; */

        /* Aws::S3::S3Client s3_client(cfg); */
        /* std::vector<std::future<int>> put_futures; */
        /* std::vector<std::future<int>> get_futures; */
        /* struct timespec start, finish; */
        /* double elapsed_write; */
        /* double elapsed_read; */

        /* clock_gettime(CLOCK_MONOTONIC, &start); */

        /* ThreadPool write_pool(num_objects); */
        /* ThreadPool read_pool(num_objects); */
        /* std::vector<std::string> keynames; */
        /* const char* buffers_write[num_objects]; */
        /* const char* keys[num_objects]; */
        /* const char* buffers_read[num_objects]; */
        /* const char* buckets[num_objects]; */
        /* long buffer_sizes[num_objects]; */



	/* for (int i = 0; i < num_objects; i++) { */
	    /* auto keyname = prefix + "/" + std::to_string(i); */
        /*     keynames.push_back(keyname); */
        /* } */

	/* for (int i = 0; i < num_objects; i++) { */
        /*     keys[i] = keynames[i].c_str(); */
        /*     buffers_write[i] =  (char*) buffer + objsizebytes*i; */
        /*     buffers_read[i] =  (char*) buffer2 + objsizebytes*i; */
        /*     buffer_sizes[i]  = objsizebytes; */
        /*     buckets[i] = bucket.c_str(); */
        /* } */

        /* put_objects((void**) buffers_write,num_objects, buffer_sizes, buckets, keys, num_objects); */
        /* clock_gettime(CLOCK_MONOTONIC, &finish); */
        /* elapsed_write = (finish.tv_sec - start.tv_sec); */
        /* elapsed_write += (finish.tv_nsec - start.tv_nsec) / 1000000000.0; */
        /* double write_gb_sec = (objsizebytes*num_objects/elapsed_write)/1e9; */
        /* clock_gettime(CLOCK_MONOTONIC, &start); */
        /* get_objects((void**) buffers_read,num_objects, buffer_sizes, buckets, keys, num_objects); */
        /* clock_gettime(CLOCK_MONOTONIC, &finish); */
        /* int cmp1 = memcmp(buffer,buffer2,objsizebytes*num_objects); */
        /* std::cout << "Pre-Download Buffer1 vs Buffer2 memcmp: " << cmp0 << std::endl; */
        /* std::cout << "Post-Download Buffer1 vs Buffer2 memcmp: " << cmp1 << std::endl; */

        /* elapsed_read = (finish.tv_sec - start.tv_sec); */
        /* elapsed_read += (finish.tv_nsec - start.tv_nsec) / 1000000000.0; */
        /* double read_gb_sec = (objsizebytes*num_objects/elapsed_read)/1e9; */
        /* std::cout << "S3IO Bench SUMMARY: " << std::endl; */
        /* std::cout << "====================================" << std::endl; */
        /* std::cout << "Total Bytes Written: " << objsizebytes*num_objects << std::endl; */
        /* std::cout << "Num Keys Written: " << num_objects << std::endl; */
        /* std::cout << "Size per Key Writen: " << objsizebytes << std::endl; */
        /* std::cout << "Time taken to write: " << elapsed_write << std::endl; */
        /* std::cout << "Write GB/s: " << write_gb_sec << std::endl; */

        /* std::cout << "Total Bytes Read : " << objsizebytes*num_objects << std::endl; */
        /* std::cout << "Num Keys Read : " << num_objects << std::endl; */
        /* std::cout << "Size per Key Read: " << objsizebytes << std::endl; */
        /* std::cout << "Time taken to read: " << elapsed_read << std::endl; */
        /* std::cout << "Read GB/s: " << read_gb_sec << std::endl; */





		/*
        Aws::S3::Model::GetObjectRequest object_request;
        object_request.SetBucket(bucket_name);
        object_request.SetKey(key_name);

        uint32_t bufferSize = 10249416;
        char* buffer = (char*) calloc(bufferSize, sizeof(char));

        object_request.SetResponseStreamFactory(
            [buffer, bufferSize]()
            {
            	return Aws::New<boost::interprocess::bufferstream>("fastio", (char*)buffer, bufferSize);
            });

        auto get_object_outcome = s3_client.GetObject(object_request);
        Aws::OFStream local_file;
        local_file.open(key_name.c_str(), std::ios::out | std::ios::binary);
        auto f = new boost::interprocess::bufferstream((char*)buffer, bufferSize);
        local_file << f->rdbuf();
        std::cout << "Done!" << std::endl;
        */
    }

    Aws::ShutdownAPI(options);



}

