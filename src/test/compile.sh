CC=mpicc
CFLAGS="-DNOPRINT"
OPS="-g -fsanitize=address,undefined,integer -fno-omit-frame-pointer -Wl,-no_compact_unwind"
LIBS="-L.. -ldiago -L/opt/homebrew/lib -lscalapack"
INC="-I/opt/homebrew/include"
MPIRUN="mpirun"

# $CC $OPS $CFLAGS $INC test1.c $LIBS -o test1
# $MPIRUN -n 8 ./test1 
# $MPIRUN -n 7 ./test1 
# $MPIRUN -n 6 ./test1 

# $CC $OPS $CFLAGS $INC test2.c $LIBS  -o test2
# $MPIRUN -n 8 ./test2 
# $MPIRUN -n 7 ./test2 
# $MPIRUN -n 6 ./test2 

# $CC $OPS $CFLAGS $INC test3.c $LIBS  -o test3
# $MPIRUN -n 8 ./test3 
# $MPIRUN -n 7 ./test3 
# $MPIRUN -n 6 ./test3 

#rm -rf test1 test2 test3 *.dSYM

$CC $OPS $CFLAGS $INC test_herm.c $LIBS  -o test
$MPIRUN -n 8 ./test 
$MPIRUN -n 7 ./test 
$MPIRUN -n 6 ./test 


$CC $OPS $CFLAGS $INC test_bse_solver.c $LIBS  -o test
$MPIRUN -n 8 ./test 
$MPIRUN -n 7 ./test 
$MPIRUN -n 6 ./test 
#
mpifort -c -g -fbounds-check -fbacktrace  -pedantic -Wall -std=f2003 test_herm_fortran.f90 -I.. 

mpicc $OPS -pedantic -Wall -std=f2003 test_herm_fortran.o -I.. -L.. -ldiago -L/opt/homebrew/lib -lscalapack -llapack -lblas -L/opt/homebrew/Cellar/gcc/14.1.0_1/lib/gcc/current -lgfortran -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -o test

$MPIRUN -n 8 ./test
$MPIRUN -n 7 ./test
$MPIRUN -n 6 ./test


