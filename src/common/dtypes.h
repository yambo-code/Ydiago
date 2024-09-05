#pragma once 
#include "../diago.h"
#include <mpi.h>
#include <stdbool.h>

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


