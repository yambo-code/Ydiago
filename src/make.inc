CC                  :=  mpicc
FC                  :=  mpifort
AR                  :=  ar

CFLAGS       	    := -O1 -g -fsanitize=address -fno-omit-frame-pointer # for mkl add 
OPENMP_FLAGS   	    := #-DWITH_ELPA -D_DOUBLE -D_CUDA

ELPA_INC            :=  


# Notes on Cflags,
# Simply pass flags that are used in yambo. In case of elpa support, use -DWITH_ELPA
