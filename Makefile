CPPC=g++
BINS_DIR=bin

CV_CFLAGS = -I/usr/local/include/opencv4
CV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

CFLAGS += $(CV_CFLAGS)
LIBS += $(CV_LIBS)

DEBUG_CFLAGS = $(CFLAGS) -g #-fsanitize=address

default: all

.PHONY: clean

bin:
	mkdir -p $(BINS_DIR)

test_utils : src/test_utils.cpp | bin
	$(CPPC) $(DEBUG_CFLAGS) -o $(BINS_DIR)/$@ $< src/utils.cpp $(LIBS)

