CPPC=g++
NVCC=nvcc
BINS_DIR=bin

CV_CFLAGS = #-I/usr/include/opencv4
CV_LIBS = #-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

CFLAGS += $(CV_CFLAGS)
LIBS += $(CV_LIBS)
NVCCFLAGS = $(CFLAGS) -gencode arch=compute_50,code=sm_50

DEBUG_CFLAGS = $(CFLAGS) -g -fsanitize=address
DEBUG_NVCCFLAGS = $(NVCCFLAGS) -g -G

RELEASE_CFLAGS = $(CFLAGS) -O3
RELEASE_NVCCFLAGS = $(NVCCFLAGS) -O3

default: all

.PHONY: clean

bin:
	mkdir -p $(BINS_DIR)

test_utils : src/test_utils.cpp | bin
	$(CPPC) $(RELEASE_CFLAGS) -o $(BINS_DIR)/$@ $< src/utils.cpp $(LIBS)

serial_demo : src/serial_demo.cpp | bin
	$(CPPC) $(RELEASE_CFLAGS) -o $(BINS_DIR)/$@ $< src/serial.cpp src/utils.cpp $(LIBS)
        
parallel_demo : src/parallel_demo.cu | bin
	$(NVCC) $(RELEASE_NVCCFLAGS) -o $(BINS_DIR)/$@ $< src/parallel.cu src/utils.cpp $(LIBS)

shared_demo : src/shared_demo.cu | bin
	$(NVCC) $(RELEASE_NVCCFLAGS) -o $(BINS_DIR)/$@ $< src/shared.cu src/utils.cpp $(LIBS)

all: test_utils serial_demo parallel_demo shared_demo

clean:
	rm -rf $(BINS_DIR)
