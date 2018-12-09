CXX=g++
CXXFLAGS=-g -O3 -std=c++11 -laws-cpp-sdk-core -laws-cpp-sdk-s3 -lpthread -lzmq
SOEXT ?= so

# all: fastio.$(SOEXT) cache_test
all: cache cache_client.$(SOEXT)

cache_client.$(SOEXT): cache_client.cpp
	 $(CXX) -fPIC -c -o cache_client.o $< $(CXXFLAGS)
	 $(CXX)  -dynamiclib -shared -o $@ cache_client.o $(CXXFLAGS)
	 aws s3 cp cache_client.$(SOEXT) s3://zehric-pywren-149/

cache: cache.cpp
	 $(CXX) -c -o cache.o $< $(CXXFLAGS)
	 $(CXX) -o $@ cache.o $(CXXFLAGS) -Wl,-rpath,/home/ec2-user/aws-sdk-cpp-build/.deps/install/lib
	 aws s3 cp cache s3://zehric-pywren-149/

clean:
	rm -rf *.o
	rm -rf *.so
	rm -f cache
