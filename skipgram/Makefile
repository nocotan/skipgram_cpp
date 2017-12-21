#!/bin/make
#
# Installing SkipGram with hSm
# @author Masanari Kimura
# @file Makefile
# @data 2017/12/20
#
#

CXX = g++
CXXFLAGS = -O3 -Wall -mtune=native -march=native -mfpmath=both #-pg
LDFLAGS = -lm -lboost_system
OBJS = skipgram

all: $(OBJS)

.PHONY: clean
clean:
	-rm $(OBJS)
