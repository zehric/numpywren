CXX=g++
CXXFLAGS=-g -O3 -std=c++11 -laws-cpp-sdk-core -laws-cpp-sdk-s3 -lpthread
SOEXT ?= so

all: fastio.$(SOEXT) fastio_test

fastio.$(SOEXT): fastio.cpp
	 $(CXX) -fPIC -c -o fastio_shared.o $< $(CXXFLAGS)
	 $(CXX)  -dynamiclib -shared -o $@ fastio_shared.o $(CXXFLAGS)
	 aws s3 cp fastio.$(SOEXT) s3://zehric-pywren-149/

fastio_test: fastio.cpp
	 $(CXX) -c -o fastio_test.o $< $(CXXFLAGS)
	 $(CXX) -o $@ fastio_test.o $(CXXFLAGS) -Wl,-rpath,/home/ec2-user/aws-sdk-cpp-build/.deps/install/lib

clean:
	rm -rf *.o
	rm -rf fastio_shared.o
	rm -rf fastio_test.o
