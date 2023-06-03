## inputs ###
#/* iF unleft it with detect by itself */
CC             = 'gcc'
CXX            = 'g++'
FC             = 'gfortran'

MPICC          = 'mpicc'
MPICXX         = 'mpicxx'
MPIFORT        = 'mpifort'

CXX_FLAGS      = '-Xpreprocessor -fopenmp'
CMAKE_C_FLAGS  = ' -Wno-implicit-function-declaration'
MPIC_FLAGS     = ' -Wno-implicit-function-declaration'
MPICXX_FLAGS   = '-Xpreprocessor -fopenmp'

BLAS_INCS      = '-I/opt/homebrew/Cellar/openblas/0.3.23/include'
BLAS_LIBS      = '-L/opt/homebrew/Cellar/openblas/0.3.23/lib -lopenblas'
LAPACK_LIBS    = '-L/opt/homebrew/Cellar/openblas/0.3.23/lib -lopenblas'
SCALAPACK_LIBS = '-L/opt/homebrew/lib -lscalapack'

NETCDF_INCS    = '-I/Users/murali/softwares/core/include'
NETCDF_LIBS    = '-L/Users/murali/softwares/core/lib -lnetcdf -lhdf5'

INSTALL_PREFIX = '.'
OPENMP_ON      = False

MAKECPUS       = 8
