import os 
from makediago import *



#### input end here


###### build starts here######


###### COMPILE ELEMENTAL #####
current_path = os.getcwd()

elemental_already_build = os.path.exists(current_path+'/external/Elemental/build')

if (not elemental_already_build):
    os.chdir(current_path+'/external/Elemental')
    os.mkdir('build')
    os.chdir(current_path+'/external/Elemental/build')

def check_str_pattern(input_str,check_str):
    found = False
    check_len = len(check_str) ## get length of check_str
    for i in range(len(input_str)):
        if input_str[i:i+check_len] == check_str:
            found = True
            break
    return found


CXX_FLAGS     += ' -O3 ' 
MPICXX_FLAGS  += ' -O3 '
MPIC_FLAGS    += ' -O3 '



cmake_str = ''

if check_str_pattern(BLAS_LIBS,"openblas"):
    cmake_str += ' -DEL_PREFER_OPENBLAS=TRUE '

if check_str_pattern(BLAS_LIBS,"blis"):
    cmake_str += ' -DEL_DISABLE_OPENBLAS=TRUE -DEL_PREFER_BLIS_LAPACK=TRUE '

if check_str_pattern(BLAS_LIBS,"mkl"):
    cmake_str += ' -DEL_PREFER_OPENBLAS=TRUE '

cmake_str += '-DCMAKE_INSTALL_PREFIX='+INSTALL_PREFIX.strip()+' '

cmake_str += '-DCXX_FLAGS="%s"' %(CXX_FLAGS.strip()) + ' '

cmake_str += '-DC_FLAGS="%s"' %(CMAKE_C_FLAGS.strip()) + ' '

cmake_str += '-DMPI_C_COMPILE_FLAGS="%s"' %(MPIC_FLAGS.strip()) + ' '

cmake_str += '-DMPICXX_FLAGS="%s"' %(MPICXX_FLAGS.strip())

if OPENMP_ON:
    cmake_str += '-DEL_HYBRID=TRUE '
else: 
    cmake_str += '-DEL_HYBRID=FALSE '

if len(CC.strip()) != 0:
    cmake_str += '-DCMAKE_C_COMPILER='+CC.strip() + ' '

if len(CXX.strip()) != 0:
    cmake_str += '-DCMAKE_CXX_COMPILER='+CXX.strip() + ' '

if len(FC.strip()) != 0:
    cmake_str += '-DCMAKE_Fortran_COMPILER='+FC.strip() + ' '

if len(MPICC.strip()) != 0:
    cmake_str += '-DMPI_C_COMPILER='+MPICC.strip() + ' '

if len(MPICXX.strip()) != 0:
    cmake_str += '-DMPI_CXX_COMPILER='+MPICXX.strip() + ' '

if len(MPIFORT.strip()) != 0:
    cmake_str += '-DMPI_Fortran_COMPILER='+MPIFORT.strip() + ' '

math_libs = SCALAPACK_LIBS.strip() + ' ' + LAPACK_LIBS.strip() + ' ' +  BLAS_LIBS.strip()

cmake_str += '-DMATH_LIBS=' + ('"%s"' %(math_libs.strip()))

# print(cmake_str)
if (not elemental_already_build):
    os.system("cmake .. -DCMAKE_BUILD_TYPE=Debug -DEL_TESTS=OFF -DEL_EXAMPLES=OFF  -DEL_DISABLE_QUAD=ON "+cmake_str)
    os.system("make -j%d all" %(MAKECPUS))
    os.system("make install")
    os.chdir(current_path)

## Compile nd_array

os.chdir(current_path+'/external/nd_array')
os.system("python3 compile.py")
os.chdir(current_path)

# ./build/CMakeCache.txt
###### COMPILE the DIAGONALIZATION CODE #####

files = ['El_diagonalize.c', 'Hbse.c', 'symmetries.c', 'deltaE.c', 'bse_diagonalize.c', 'bs_table.c', 'helper.c', 'netCDF4_io.c','libcfg.c','read_input.c',]

file_name_combined = ' '
for i in files:
    file_name_combined += ' ' + i + ' '

ELEMENTAL_INCS = ' '
ELEMENTAL_LIBS = ' '
with open(current_path+'/external/Elemental'+'/build/CMakeCache.txt',mode='r') as file:
    for line in file:
        line = line.rstrip()
        if (line.startswith("MPC_INCLUDES")):
            add_ele_path = line.split('=')[-1].strip()
            if (len(add_ele_path)>1): ELEMENTAL_INCS = ELEMENTAL_INCS + ' -I/'+ add_ele_path
        elif (line.startswith("MPC_LIBRARIES")):
            add_ele_path = line.split('=')[-1].strip()
            #if (len(add_ele_path)>1): ELEMENTAL_LIBS = ELEMENTAL_LIBS + ' '+ add_ele_path
        elif (line.startswith("MPFR_INCLUDES")):
            add_ele_path = line.split('=')[-1].strip()
            if (len(add_ele_path)>1): ELEMENTAL_INCS = ELEMENTAL_INCS + ' -I/'+ add_ele_path
        elif (line.startswith("MPFR_LIBRARIES")):
            add_ele_path = line.split('=')[-1].strip()
            #if (len(add_ele_path)>1): ELEMENTAL_LIBS = ELEMENTAL_LIBS + ' '+ add_ele_path
        elif (line.startswith("GMP_INCLUDES")):
            add_ele_path = line.split('=')[-1].strip()
            if (len(add_ele_path)>1): ELEMENTAL_INCS = ELEMENTAL_INCS + ' -I/'+ add_ele_path
        elif (line.startswith("GMP_LIBRARIES")):
            add_ele_path = line.split('=')[-1].strip()
            #print(add_ele_path)
            if (len(add_ele_path)>1): ELEMENTAL_LIBS = ELEMENTAL_LIBS + ' '+ add_ele_path


os.chdir(current_path+'/src')

## Compile the c++ file ElfullControlEig.cpp
os.system(MPICXX + " -c -std=c++11 " + MPICXX_FLAGS + " " +ELEMENTAL_INCS+' ' +' -I'+current_path+ \
            '/external/Elemental/build/include -I'+current_path+'/external/nd_array/src ' + NETCDF_INCS + ' ElfullControlEig.cpp ')

for ifiles in files:
    compile = MPICC + ' ' + MPIC_FLAGS + ' -c ' + ' -I'+current_path+'/external/Elemental/build/include -I'+ \
    current_path+'/external/nd_array/src ' + NETCDF_INCS + ' ' + ' ' + ifiles 
    print(compile)
    os.system(compile)

## some times libraries are written to lib64

is64lib = os.path.exists(current_path+'/external/Elemental/build/lib64')

if is64lib:
    os.chdir(current_path+'/external/Elemental/build/lib')
    os.system("ln -s ../lib64/* .")

os.chdir(current_path+'/src')
compile = MPICXX + ' ' + MPIC_FLAGS + ' *.o ' + SCALAPACK_LIBS + ' ' + LAPACK_LIBS +  ' ' + BLAS_LIBS + ' ' + NETCDF_LIBS+ ' ' + '-Wl,-rpath '+\
            current_path+'/external/Elemental/build/lib -Wl,-rpath ' + current_path+'/external/nd_array/src' \
        + '  -L'+current_path+'/external/Elemental/build/lib -lEl -lpmrrr -lElSuiteSparse -lparmetis -lmetis -L'+ current_path + '/external/nd_array/src -lnd_array -lm '  + ' -o ydiago ' 
## Compile Ydiago
print(compile)
os.system(compile)
os.system("rm *.o ")
os.chdir(current_path)
