#include <fstream>
#include <iostream>
#include <string>
#include <zmq.hpp>
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <mutex>

extern "C" {
    int run_benchmark_gets(int);
    int run_benchmark_pins(int, int, double *, double *);
    int run_benchmark_puts(int, long);
    int pin_objects(long num_objects, const char** keys_, double *start_times_, double *finish_times_, int num_threads);
    int pin_objects2(long num_objects, const char** keys_, const char** rets_,  int num_threads);
    int release_objects(long num_objects, const char** keys_, int num_threads);
}

static std::mutex mutex;
static int curr_task = 0;
static int num_tasks;
static double *start_times;
static double *finish_times;
static int unique;
static const char **keys;
static const char **rets;


int get_task() {
    std::unique_lock<std::mutex> lock(mutex);
    return curr_task++;
}

void *pins_worker(void *arg) {
    std::string prefix = std::string("test");
    //  Prepare our context and socket
    zmq::context_t *context = (zmq::context_t *) arg;
    zmq::socket_t socket (*context, ZMQ_REQ);
    socket.connect ("ipc:///tmp/local_cache");
    zmq::socket_t rel_socket(*context, ZMQ_REQ);
    rel_socket.connect("ipc:///tmp/local_cache_release");

    int task;
    while ((task = get_task()) < num_tasks) {
        double *start = start_times + task;
        double *finish = finish_times + task;
        struct stat stat;
        int fd;
        long size;
        void *data;

        std::string key = prefix + std::to_string(rand() % unique);
        int len = strlen(key.c_str()) + 1;
        zmq::message_t request (len + 1);
        char *msg = (char *) request.data();
        msg[0] = 0;
        struct timespec start_t, finish_t;
        clock_gettime(CLOCK_REALTIME, &start_t);
        *start = start_t.tv_sec + ((double) start_t.tv_nsec / 1e9);
        memcpy (msg + 1, key.c_str(), len);
        socket.send (request);

        //  Get the reply.
        zmq::message_t reply;
        socket.recv (&reply);

        std::string path = std::string((char *) reply.data());
        fd = open(path.c_str(), O_RDWR);
        fstat(fd, &stat);
        size = stat.st_size;
        data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

        clock_gettime(CLOCK_REALTIME, &finish_t);
        *finish = finish_t.tv_sec + ((double) finish_t.tv_nsec / 1e9);

        munmap(data, size);
        close(fd);
        zmq::message_t request2(len + 1);
        msg = (char *) request2.data();
        msg[0] = 1;
        memcpy (msg + 1, key.c_str(), len);
        rel_socket.send (request2);
        //  Get the reply.
        rel_socket.recv (&reply);

    }
}

int run_benchmark_pins(int num, int unique_, double *start_times_, double *finish_times_)
{
    srand(time(0));
    curr_task = 0;
    num_tasks = num;
    start_times = start_times_;
    finish_times = finish_times_;
    int num_threads = 10;
    unique = unique_;
    zmq::context_t context (1);
    pthread_t workers[num_threads];
    std::cout << "Creating threadpool of size: " << num_threads << std::endl;
    for (int i = 0; i < num_threads; i ++) {
        pthread_create (workers + i, NULL, pins_worker, &context);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(workers[i], NULL);
    }
    return 0;
}

void *pin_objects_worker(void *arg) {
    //  Prepare our context and socket
    zmq::context_t *context = (zmq::context_t *) arg;
    zmq::socket_t socket (*context, ZMQ_REQ);
    socket.connect ("ipc:///tmp/local_cache");
    zmq::socket_t rel_socket(*context, ZMQ_REQ);
    rel_socket.connect("ipc:///tmp/local_cache_release");

    int task;
    while ((task = get_task()) < num_tasks) {
        double *start = start_times + task;
        double *finish = finish_times + task;
        struct stat stat;
        int fd;
        long size;
        void *data;

        std::string key = std::string(keys[task]);
        int len = strlen(key.c_str()) + 1;
        zmq::message_t request (len + 1);
        char *msg = (char *) request.data();
        msg[0] = 0;
        struct timespec start_t, finish_t;
        clock_gettime(CLOCK_REALTIME, &start_t);
        *start = start_t.tv_sec + ((double) start_t.tv_nsec / 1e9);
        memcpy (msg + 1, key.c_str(), len);
        socket.send (request);

        //  Get the reply.
        zmq::message_t reply;
        socket.recv (&reply);

        std::string path = std::string((char *) reply.data());
        fd = open(path.c_str(), O_RDWR);
        fstat(fd, &stat);
        size = stat.st_size;
        data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

        clock_gettime(CLOCK_REALTIME, &finish_t);
        *finish = finish_t.tv_sec + ((double) finish_t.tv_nsec / 1e9);

        munmap(data, size);
        close(fd);
        zmq::message_t request2(len + 1);
        msg = (char *) request2.data();
        msg[0] = 1;
        memcpy (msg + 1, key.c_str(), len);
        rel_socket.send (request2);
        //  Get the reply.
        rel_socket.recv (&reply);

    }
}

int pin_objects(long num_objects, const char** keys_, double *start_times_, double *finish_times_, int num_threads)
{
    srand(time(0));
    curr_task = 0;
    num_tasks = num_objects;
    start_times = start_times_;
    finish_times = finish_times_;
    keys = keys_;
    zmq::context_t context (1);
    pthread_t workers[num_threads];
    std::cout << "Creating threadpool of size: " << num_threads << std::endl;
    for (int i = 0; i < num_threads; i ++) {
        pthread_create (workers + i, NULL, pin_objects_worker, &context);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(workers[i], NULL);
    }
    return 0;
}


void *release_objects_worker(void *arg) {
    //  Prepare our context and socket
    zmq::context_t *context = (zmq::context_t *) arg;
    zmq::socket_t rel_socket(*context, ZMQ_REQ);
    rel_socket.connect("ipc:///tmp/local_cache_release");

    int task;
    while ((task = get_task()) < num_tasks) {
        long size;
        void *data;

        std::string key = std::string(keys[task]);
        int len = strlen(key.c_str()) + 1;

        zmq::message_t request2(len + 1);
        char *msg = (char *) request2.data();
        msg[0] = 1;
        memcpy (msg + 1, key.c_str(), len);
        rel_socket.send (request2);
        //  Get the reply.
        zmq::message_t reply;
        rel_socket.recv (&reply);

    }
}

int release_objects(long num_objects, const char** keys_, int num_threads)
{
    srand(time(0));
    curr_task = 0;
    num_tasks = num_objects;
    keys = keys_;
    zmq::context_t context (1);
    pthread_t workers[num_threads];
    std::cout << "Creating threadpool of size: " << num_threads << std::endl;
    for (int i = 0; i < num_threads; i ++) {
        pthread_create (workers + i, NULL, release_objects_worker, &context);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(workers[i], NULL);
    }
    return 0;
}

void *pin_objects_worker2(void *arg) {
    //  Prepare our context and socket
    zmq::context_t *context = (zmq::context_t *) arg;
    zmq::socket_t socket (*context, ZMQ_REQ);
    socket.connect ("ipc:///tmp/local_cache");
    zmq::socket_t rel_socket(*context, ZMQ_REQ);
    rel_socket.connect("ipc:///tmp/local_cache_release");

    std::cout << "Contex acquired and connected to socket: " << std::endl;

    int task;
    while ((task = get_task()) < num_tasks) {
        //double *start = start_times + task;
        //double *finish = finish_times + task;
        //struct stat stat;
        //int fd;
        //long size;
        void *data;

        std::string key = std::string(keys[task]);
        int len = strlen(key.c_str()) + 1;
        zmq::message_t request (len + 1);
        char *msg = (char *) request.data();
        msg[0] = 0;
        //struct timespec start_t, finish_t;
        //clock_gettime(CLOCK_REALTIME, &start_t);
        //*start = start_t.tv_sec + ((double) start_t.tv_nsec / 1e9);
        memcpy (msg + 1, key.c_str(), len);

        std::cout << "On task: " << task << " out of " << num_tasks << std::endl;
        std::cout << "Requesting key: " << key << std::endl;

        socket.send (request);

        //  Get the reply.
        zmq::message_t reply;
        socket.recv (&reply);

        std::cout << "Received reply: " << std::endl;

        std::string path = std::string((char *) reply.data());

        std::cout << "Got path to file: " << path << std::endl;


        memcpy ((char *)rets[task], path.c_str(), 32);

        std::cout << "Copied path to file: " << (char *) rets[task] << std::endl;

        //fd = open(path.c_str(), O_RDWR);
        //fstat(fd, &stat);
        //size = stat.st_size;
        //data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

        //clock_gettime(CLOCK_REALTIME, &finish_t);
        //*finish = finish_t.tv_sec + ((double) finish_t.tv_nsec / 1e9);

        //munmap(data, size);
        //close(fd);
        //zmq::message_t request2(len + 1);
        //msg = (char *) request2.data();
        //msg[0] = 1;
        //memcpy (msg + 1, key.c_str(), len);
        //rel_socket.send (request2);
        //  Get the reply.
        //rel_socket.recv (&reply);

    }
}

int pin_objects2(long num_objects, const char** keys_, const char** rets_, int num_threads)
{
    srand(time(0));
    curr_task = 0;
    num_tasks = num_objects;
    keys = keys_;
    rets = rets_;
    zmq::context_t context (1);
    pthread_t workers[num_threads];
    std::cout << "Creating threadpool of size: " << num_threads << std::endl;
    for (int i = 0; i < num_threads; i ++) {
        pthread_create (workers + i, NULL, pin_objects_worker2, &context);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(workers[i], NULL);
    }
    return 0;
}

int run_benchmark_gets(int num)
{
    srand(time(0));
    std::string prefix = std::string("test");
    //  Prepare our context and socket
    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REQ);

    socket.connect ("ipc:///tmp/local_cache");
    struct stat stat;
    int fd;
    long size;
    void *data;

    for (int request_nbr = 0; request_nbr < num; request_nbr++) {
        std::string key = prefix + std::to_string(rand() % num);
        int len = strlen(key.c_str()) + 1;
        zmq::message_t request (len + 1);
        char *msg = (char *) request.data();
        msg[0] = 0;
        memcpy (msg + 1, key.c_str(), len);
        socket.send (request);

        //  Get the reply.
        zmq::message_t reply;
        socket.recv (&reply);

        std::string path = std::string((char *) reply.data());
        fd = open(path.c_str(), O_RDWR);
        fstat(fd, &stat);
        size = stat.st_size;
        data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        void *local = malloc(size);
        memcpy(local, data, size);
        munmap(data, size);

        zmq::message_t request2(len + 1);
        msg = (char *) request2.data();
        msg[0] = 1;
        memcpy (msg + 1, key.c_str(), len);
        socket.send (request2);

        //  Get the reply.
        socket.recv (&reply);
	free(local);
    }
    return 0;
}

int run_benchmark_puts(int num, long size)
{
    std::string prefix = std::string("test");
    //  Prepare our context and socket
    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REQ);

    socket.connect ("ipc:///tmp/local_cache");
    int fd;
    void *data;


    for (int request_nbr = 0; request_nbr < num; request_nbr++) {

        std::string key = prefix + std::to_string(request_nbr);
        std::string path = "/tmp/" + key;
        FILE *fp = fopen(path.c_str(), "w+");
        for (int i = 0; i < size; i++) {
            fputc(rand() % (1 << 8), fp);
        }
        fclose(fp);
        fd = open(path.c_str(), O_RDWR);
        data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        int len = strlen(key.c_str()) + 1;
        zmq::message_t request (len + 1);
        char *msg = (char *) request.data();
        msg[0] = 2;
        memcpy (msg + 1, key.c_str(), len);
        socket.send (request);

        //  Get the reply.
        zmq::message_t reply;
        socket.recv (&reply);

        munmap(data, size);
    }
    return 0;
}


