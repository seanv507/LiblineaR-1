R = ${R_HOME}/bin/R CMD SHLIB
CXX = g++
CC = gcc
LIBS = blas/blas.a

all: trainLinear predictLinear

trainLinear: tron.o linear.o trainLinear.c blas/blas.a
	$(R) trainLinear.c tron.o linear.o $(LIBS)

predictLinear: tron.o linear.o predictLinear.c blas/blas.a
	$(R) predictLinear.c tron.o linear.o $(LIBS)

tron.o: tron.cpp tron.h
	$(CXX) -c -o tron.o tron.cpp

linear.o: linear.cpp linear.h
	$(CXX) -c -o linear.o linear.cpp

blas/blas.a:
	cd blas; make CC='$(CC)';

clean:
	cd blas;	make clean
	rm -f *~ tron.o linear.o trainLinear.o predictLinear.o