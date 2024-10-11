#pragma once

#ifdef __STDC_NO_COMPLEX__
#error Your compiler does not C99 complex numbers, Please use a supported compiler.
#endif

#define SUPPORTED_ELPA_VERSION 20231705
//====== Requries ELPA vesrion >= ELPA 2023.11.001

#define SL_WORK_QUERY_FAC 1.0001
// This is factor introduced to workaround int to float error
// and float to int conversions that happen in SL work queries.
// This must be always > 1
// ======= add yambo defs ======
#ifdef _DOUBLE
#define WITH_DOUBLE
#endif

#if defined _CUDAF || defined _OPENACC
#define WITH_CUDA
#endif

#if defined _OPENMP_GPU && defined _HIP
#define WITH_HIP
#endif

#if defined _OPENMP_GPU && defined _MKLGPU
#define WITH_INTEL_GPU
#endif

#if defined(WITH_CUDA) || defined(WITH_HIP) || defined(WITH_INTEL_GPU)
#ifndef WITH_GPU
#define WITH_GPU
#endif
#endif

#ifdef _OPENMP
#define WITH_OPENMP
#endif

#ifdef _ELPA
#define WITH_ELPA
#endif

#if defined(WITH_GPU) && !defined(WITH_ELPA)
#error GPU support is available only when compiled with elpa
#endif

// =========
#ifdef WITH_DOUBLE
typedef double D_float;
typedef double _Complex D_Cmplx;
#define D_Cmplx_MPI_TYPE MPI_C_DOUBLE_COMPLEX
#define D_float_MPI_TYPE MPI_DOUBLE
#else
typedef float D_float;
typedef float _Complex D_Cmplx;
#define D_Cmplx_MPI_TYPE MPI_C_FLOAT_COMPLEX
#define D_float_MPI_TYPE MPI_FLOAT
#endif

typedef long long int D_LL_INT;
typedef int D_INT;
/* Warning: leave D_INT to int unless
   the scalapack library uses different type for int.
*/
#define D_INT_MPI_TYPE MPI_INT  // this this type according to D_INT
