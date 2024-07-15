CC                  :=  mpicc
FC                  :=  mpifort
AR                  :=  ar

CFLAGS       	    := -O1 -g -fsanitize=address -fno-omit-frame-pointer 
YAMBO_FLAGS   	    := #-DWITH_ELPA -D_DOUBLE -D_CUDA -D_OPENMP

ELPA_INC            :=  # include directory of elpa


# Notes on YAMBO_FLAGS,
# Simply pass flags that are used in yambo. In case of elpa support, use -DWITH_ELPA
