CC                  :=  mpicc
AR                  :=  ar

CFLAGS       	    := -O1 -g -fsanitize=address -fno-omit-frame-pointer # for mkl add 
OPENMP_FLAGS   	    := #-DWITH_ELPA #-DELPH_OMP_PARALLEL_BUILD ## uncomment for openmp build

ELPA_INC            :=  


#### Notes Extra CFLAGS
### add -DCOMPILE_ELPH_DOUBLE if you want to compile the code in double precession
### if you are using yambo <= 5.1.2, you need to add "-DYAMBO_LT_5_1" to cflags
### for openmp use -DELPH_OMP_PARALLEL_BUILD in CFLAGS and set -fopenmp in LD_FLAGS and CFLAGS
