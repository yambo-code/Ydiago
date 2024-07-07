#pragma once

// Here we define set of macros for errors
typedef int Err_INT;

#define DIAGO_SUCCESS 0
// This must be 0 always

#define BUF_ALLOC_FAILED -1
// Allocation of a buffer failed

#define MATRIX_INIT_FAILED -2
// Initiation of distbuted matrix failed

#define DIAGO_MPI_ERROR -3
// error from mpi routines

#define EQUAL_BLOCK_SIZE_ERROR -4
// the sizes of block size are not same

#define DESC_MISMATCH_ERROR -5
// mismatch of descriptors of two distributed arrays

#define OUT_OF_BOUNDS_ERROR -6
// error when the array is referenced out of its bounds

#define DESC_ERROR -7
// Failed to set descriptor for a block cyclic matrix

#define MATRIX_NOT_INIT -8
// Provided matrix pointer is NULL, i.e it is not initiated or already freed

#define MATRIX_NOT_SQUARE -9
// The matrix is not a square matrix

#define WRONG_NON_TDA_BSE_MAT -10
// The bse matrix provided is not a multiple of 2.

#define ERR_NULL_PTR_BUFFER -11
// Null pointer is passed instead of a valid buffer.

#define INCOMPATIBLE_BLOCK_SIZE_ERR -12
// Incompatible block size

#define INCOMPATIBLE_EIG_VECS_DIM_ERR
// Dimesion of eigen vectors not compatible with matrix

#define QUEUE_ALREADY_INIT 1
// Only one set and one get queue can be set at a time.
// If a previous queue is not processed before initiating
// a new queue, we see this error.

#define QUEUE_NOT_INIT 2
// Set/Get Queue is not initiated and set/getElement is called.

#define QUEUE_LIMIT_REACHED 3
// Limit has been reached when setting or Getting the Queue.

#define BLACS_GRID_INIT_ERROR 4
// Error when blacs grid initition failed

#define INVALID_BLACS_CXT 5
// invalid blacs context

#define CHOLESKY_FAILED -50
// CHOLESKY factorization failed

#define HERMITIAN_DIAGO_FAILED -51
// hermitian diagonalization failed

#define SL_WORK_SPACE_ERROR -52
// error computing the scalapack workspaces

#define SL_DIAGO_ERROR -53
// error from scalapack diagonalization routine

#define SL_SCHUR_ERROR -54
// error compute schur decomposiotn

#define SL_HESSENBERG_ERROR -55
// error Hessenberg reduction

#define SL_TREVC_ERROR -56
// Triangular eigenvector solver failed

#define SL_EIG_SORT_ERROR -57
// Soring of eigenvectors failed

#define ELPA_SETUP_ERROR -101
// error when setting up elpa

#define ELPA_UNSUPPORTED_ERROR -102
// unsupported version elpa

#define ELPA_UNINIT_ERROR -103
// failed to uninit elpa

#define ELPA_DEALLOC_ERROR -104
// failed to deallocated elpa

#define ELPA_SKEW_SYMM_DIAGO_ERROR -105
// Elpa skew symmetric solver failed

#define ELPA_HERM_MULTIPLY_ERROR -106