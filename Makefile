CC=gcc
NVCC=nvcc
CXXFLAGS= -std=c++11
CUDAFLAGS= -c -std=c++11
LIBS=-lgmp
LIBDIRS=-L/usr/local/cuda/lib64
INCDIRS=-L/usr/local/cuda/include

cudasquare.o: cudasquare.cu
	$(NVCC) $(CUDAFLAGS) cudasquare.cu

all: cudasquare.o
	$(CC) -o cudasquare cudasquare.o $(LIBDIRS) $(INCDIRS) $(LIBS) $(CXXFLAGS)

clean:
	rm -rf cudasquare *.o