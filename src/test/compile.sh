CC=mpicc
CFLAGS="-DNOPRINT"
OPS="-g -fsanitize=address,undefined,integer -fno-omit-frame-pointer -Wl,-no_compact_unwind"
LIBS="-L.. -ldiago -L/opt/homebrew/lib -lscalapack"
INC="-I/opt/homebrew/include"


# $CC $OPS $CFLAGS $INC test1.c $LIBS -o test1
# mpirun -n 8 ./test1 
# mpirun -n 7 ./test1 
# mpirun -n 6 ./test1 

# $CC $OPS $CFLAGS $INC test2.c $LIBS  -o test2
# mpirun -n 8 ./test2 
# mpirun -n 7 ./test2 
# mpirun -n 6 ./test2 

# $CC $OPS $CFLAGS $INC test3.c $LIBS  -o test3
# mpirun -n 8 ./test3 
# mpirun -n 7 ./test3 
# mpirun -n 6 ./test3 

#rm -rf test1 test2 test3 *.dSYM

$CC $OPS $CFLAGS $INC test_herm.c $LIBS  -o test
mpirun -n 8 ./test 
#mpirun -n 7 ./test 
#mpirun -n 6 ./test 


$CC $OPS $CFLAGS $INC test_bse_solver.c $LIBS  -o test
mpirun -n 8 ./test 
#mpirun -n 7 ./test 
#mpirun -n 6 ./test 
