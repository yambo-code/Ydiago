#pragma once

#include <stdlib.h>
#include <mpi.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include "common/error.h"

#define SUPPORTED_ELPA_VERSION 20221109
//====== Requries ELPA vesrion > ELPA 2022.11.001

#ifdef WITH_DOUBLE
typedef double D_float;
typedef double complex D_Cmplx;
#define D_Cmplx_MPI_TYPE MPI_C_DOUBLE_COMPLEX
#define D_float_MPI_TYPE MPI_DOUBLE

#define SL_FunCmplx(FUN_NAME) scalapack_fun_HIDDEN_Cmplx(FUN_NAME)
#define scalapack_fun_HIDDEN_Cmplx(FUN_NAME) pz##FUN_NAME##_

#define SL_FunFloat(FUN_NAME) scalapack_fun_HIDDEN_float(FUN_NAME)
#define scalapack_fun_HIDDEN_float(FUN_NAME) pd##FUN_NAME##_

#define Elpa_FunCmplx(FUN_NAME) ElpaCmplx_HIDDEN(FUN_NAME)
#define ElpaCmplx_HIDDEN(FUN_NAME) elpa_##FUN_NAME##_double_complex

#define Elpa_FunFloat(FUN_NAME) ElpaFloat_HIDDEN(FUN_NAME)
#define ElpaFloat_HIDDEN(FUN_NAME) elpa_##FUN_NAME##_double

#else
typedef float D_float;
typedef float complex D_Cmplx;
#define D_Cmplx_MPI_TYPE MPI_C_FLOAT_COMPLEX
#define D_float_MPI_TYPE MPI_FLOAT

#define SL_FunCmplx(FUN_NAME) scalapack_fun_HIDDEN_Cmplx(FUN_NAME)
#define scalapack_fun_HIDDEN_Cmplx(FUN_NAME) pc##FUN_NAME##_

#define SL_FunFloat(FUN_NAME) scalapack_fun_HIDDEN_float(FUN_NAME)
#define scalapack_fun_HIDDEN_float(FUN_NAME) ps##FUN_NAME##_

#define Elpa_FunCmplx(FUN_NAME) ElpaCmplx_HIDDEN(FUN_NAME)
#define ElpaCmplx_HIDDEN(FUN_NAME) elpa_##FUN_NAME##_float_complex

#define Elpa_FunFloat(FUN_NAME) ElpaFloat_HIDDEN(FUN_NAME)
#define ElpaFloat_HIDDEN(FUN_NAME) elpa_##FUN_NAME##_float

#endif

typedef long long int D_LL_INT;
typedef int D_INT;
/* Warning: leave D_INT to int unless
   the scalapack library uses different type for int.
*/
#define D_INT_MPI_TYPE MPI_INT // this this type according to D_INT

struct D_Matrix
{
    // struct for the block cyclic matrix
    D_Cmplx* data; // local data pointer
    D_LL_INT lda[2]; // local leading dimensions
    D_INT gdims[2]; // global dims of matrix
    D_INT ldims[2]; // local dims of matrix
    D_INT pgrid[2]; // processor grid
    D_INT pids[2]; // processor ids i.e grid indices of this processor
    D_INT block_size[2]; // block size in block cyclic layout
    D_INT blacs_ctxt; // blacs context
    MPI_Comm comm; // MPI communicator
    D_INT* mapP2iD; // (comm size) array that map process coordinates to mpi rank.
    void* SetQueuePtr; // Pointer to process set/get element Queues
    D_LL_INT nSetQueueElements; // number of elements in QueuePtr
    D_LL_INT iset; // number of elements filled in SetQueue
    void* GetQueuePtr; // Pointer to process set/get element Queues
    D_LL_INT nGetQueueElements; // number of elements in QueuePtr
    D_LL_INT iget; // number of elements filled in GetQueue
    bool cpu_engage; // This is true if the processor belongs to process grid, else false
    // This happens when mpi processes are more than the processor grid cpus.
};

struct MPIcxt
{
    D_INT pgrid[2]; // processor grid
    char layout; // 'C'/'R' coloumn or row major for processor grid
    MPI_Comm comm; // comm
    D_INT ctxt; // context
};

#if !defined(MIN)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#if !defined(MAX)
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif