#!/bin/make
CXX = g++
CXXFLAGS = -O3 -Wall -mtune=native -march=native -mfpmath=both #-pg
LDFLAGS = -lm -lboost_system
OBJS = skipgram

all: $(OBJS)

.PHONY: clean
clean:
	-rm $(OBJS)
