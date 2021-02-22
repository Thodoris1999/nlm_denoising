CPPC=g++
NVCC=nvcc
BINS_DIR=bin

CV_CFLAGS = -I/usr/include/opencv4
CV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

CFLAGS += $(CV_CFLAGS)
LIBS += $(CV_LIBS)

DEBUG_CFLAGS = $(CFLAGS) -g -fsanitize=address
DEBUG_NVCCFLAGS = $(CFLAGS) -g -G

RELEASE_CFLAGS = $(CFLAGS) -O3

default: all

.PHONY: clean

bin:
	mkdir -p $(BINS_DIR)

test_utils : src/test_utils.cpp | bin
	$(CPPC) $(RELEASE_CFLAGS) -o $(BINS_DIR)/$@ $< src/utils.cpp $(LIBS)

serial : src/serial.cpp | bin
	$(CPPC) $(DEBUG_CFLAGS) -O3 -o $(BINS_DIR)/$@ $< src/utils.cpp $(LIBS)
        
parallel : src/parallel.cu | bin
	$(NVCC) $(DEBUG_NVCCFLAGS) -O3 -o $(BINS_DIR)/$@ $< src/utils.cpp $(LIBS)

all: test_utils serial parallel

clean:
	rm -rf $(BINS_DIR)
