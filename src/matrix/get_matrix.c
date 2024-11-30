// THis file contains functions that gets elements from the
// global distribution matrix
//
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "../SL/scalapack_header.h"
#include "../common/dtypes.h"
#include "../common/error.h"
#include "../diago.h"
#include "matrix.h"

struct GetElement
{
    D_Cmplx* value;
    D_INT i;  // global row index
    D_INT j;  // global col index
};

static inline D_INT Get_idxG2iD(const D_INT* getCmpPrms, const D_INT i,
                                const D_INT j, const D_INT* Get_mapP2iD)
{
    // getCmpPrms is array[5]{cxt, BlkX, BlkY, NProcX, NProcY}
    D_INT prow = INDXG2P(i, getCmpPrms[1], 0, 0, getCmpPrms[3]);
    D_INT pcol = INDXG2P(j, getCmpPrms[2], 0, 0, getCmpPrms[4]);
    return Get_mapP2iD[prow * getCmpPrms[4] + pcol];
}

Err_INT initiateGetQueue(void* D_mat, const D_LL_INT nelements)
{
    if (!D_mat)
    {
        return MATRIX_NOT_INIT;  // error.
    }
    struct D_Matrix* mat = D_mat;

    int mpi_error = MPI_Barrier(mat->comm);  // make this a collective call
    if (mpi_error != MPI_SUCCESS)
    {
        return DIAGO_MPI_ERROR;
    }

    if (mat->GetQueuePtr)
    {
        return QUEUE_ALREADY_INIT;  // error. Queue already initated.
    }
    mat->nGetQueueElements = nelements;
    mat->iget = 0;

    mat->GetQueuePtr = malloc(sizeof(struct GetElement) * (nelements + 1));
    // Always alloc aleast one element to check if Queue is active
    if (!mat->GetQueuePtr)
    {
        return BUF_ALLOC_FAILED;
    }

    return DIAGO_SUCCESS;
}

Err_INT DMatGet(void* D_mat, const D_INT i, const D_INT j, D_Cmplx* value)
{
    if (!D_mat)
    {
        return MATRIX_NOT_INIT;  // error.
    }

    struct D_Matrix* mat = D_mat;

    if (i >= mat->gdims[0] || j >= mat->gdims[1])
    {
        // error : out of bounds
        return OUT_OF_BOUNDS_ERROR;
    }

    if (!mat->GetQueuePtr)
    {
        return QUEUE_NOT_INIT;  // error not initated
    }
    if (mat->iget >= mat->nGetQueueElements)
    {
        return QUEUE_LIMIT_REACHED;  // error out of bounds
    }
    struct GetElement* QueuePtr = mat->GetQueuePtr;

    QueuePtr[mat->iget].i = i;
    QueuePtr[mat->iget].j = j;
    QueuePtr[mat->iget].value = value;

    ++mat->iget;

    return 0;
}

Err_INT ProcessGetQueue(void* D_mat)
{
    // all comm cpus must call this

    Err_INT error = DIAGO_SUCCESS;

    if (!D_mat)
    {
        error = MATRIX_NOT_INIT;  // error.
        goto error_get_queue_0;
    }

    struct D_Matrix* mat = D_mat;
    if (!mat->GetQueuePtr)
    {
        error = QUEUE_NOT_INIT;  // error not initated
        goto error_get_queue_1;
    }

    const D_INT* Get_mapP2iD = mat->mapP2iD;

    // next we need to sort the GetQueue array based on the ranks
    const D_INT cmpGetupEle[5] = {mat->blacs_ctxt, mat->block_size[0],
                                  mat->block_size[1], mat->pgrid[0],
                                  mat->pgrid[1]};

    // some MPI related stuff
    int TotalCommCpus;

    int mpi_error = MPI_Comm_size(mat->comm, &TotalCommCpus);
    if (mpi_error != MPI_SUCCESS)
    {
        error = DIAGO_MPI_ERROR;
        goto error_get_queue_1;
    }

    int* counts_send = calloc(4 * TotalCommCpus, sizeof(*counts_send));
    if (!counts_send)
    {
        error = BUF_ALLOC_FAILED;  // error
        goto error_get_queue_1;
    }
    int* displacements_send = counts_send + TotalCommCpus * 1;
    int* counts_recv = counts_send + TotalCommCpus * 2;
    int* displacements_recv = counts_send + TotalCommCpus * 3;

    for (D_LL_INT i = 0; i < TotalCommCpus; ++i)
    {
        counts_recv[i] = 0;
        displacements_send[i] = 0;
    }

    struct GetElement* getQueue = mat->GetQueuePtr;
    struct GetElement* getQueue_sorted =
        malloc((mat->nGetQueueElements + 1) * sizeof(*getQueue_sorted));
    if (!getQueue_sorted)
    {
        error = BUF_ALLOC_FAILED;  // error
        goto error_get_queue_2;
    }

    // Use displacements_send as tmp buffer to first find the number of elements
    //
    for (D_LL_INT i = 0; i < mat->nGetQueueElements; ++i)
    {
        D_INT tmp_rank =
            Get_idxG2iD(cmpGetupEle, getQueue[i].i, getQueue[i].j, Get_mapP2iD);
        ++displacements_send[tmp_rank];
    }

    // compute cumulative sum
    for (D_INT i = 1; i < TotalCommCpus; ++i)
    {
        displacements_send[i] += displacements_send[i - 1];
    }
    if (TotalCommCpus > 1)
    {
        memmove(displacements_send + 1, displacements_send,
                (TotalCommCpus - 1) * sizeof(*displacements_send));
    }
    displacements_send[0] = 0;

    // Cluster them into groups by ranks
    for (D_LL_INT i = 0; i < mat->nGetQueueElements; ++i)
    {
        D_INT tmp_rank =
            Get_idxG2iD(cmpGetupEle, getQueue[i].i, getQueue[i].j, Get_mapP2iD);
        D_LL_INT tmp_disp = displacements_send[tmp_rank];
        memcpy(getQueue_sorted + tmp_disp + counts_recv[tmp_rank], getQueue + i,
               sizeof(*getQueue));
        ++counts_recv[tmp_rank];
    }

    // swap the pointer and free it to avoid copying it
    mat->GetQueuePtr = getQueue_sorted;
    free(getQueue);
    getQueue = getQueue_sorted;
    getQueue_sorted = NULL;

    mpi_error = MPI_Alltoall(counts_recv, 1, MPI_INT, counts_send, 1, MPI_INT,
                             mat->comm);
    if (mpi_error != MPI_SUCCESS)
    {
        error = DIAGO_MPI_ERROR;
        goto error_get_queue_2;
    }

    D_LL_INT total_send = 0;
    D_LL_INT total_recv = 0;
    for (D_LL_INT i = 0; i < TotalCommCpus; ++i)
    {
        displacements_send[i] = total_send;
        displacements_recv[i] = total_recv;
        total_send += counts_send[i];
        total_recv += counts_recv[i];
    }

    // create a datatype to carry 2 ints
    MPI_Datatype MPI_idxType;

    mpi_error = MPI_Type_contiguous(2, D_INT_MPI_TYPE, &MPI_idxType);
    if (mpi_error != MPI_SUCCESS)
    {
        error = DIAGO_MPI_ERROR;
        goto error_get_queue_2;
    }

    mpi_error = MPI_Type_commit(&MPI_idxType);
    if (mpi_error != MPI_SUCCESS)
    {
        error = DIAGO_MPI_ERROR;
        goto error_get_queue_2;
    }

    // we need to send the indices of receiving elements to senders
    // In buffers we add atleast one extra element to avoid implementation
    // defined behaviour
    D_INT* tmp_idx_buf_recv = malloc(2 * sizeof(D_INT) * (total_recv + 1));
    if (!tmp_idx_buf_recv)
    {
        error = BUF_ALLOC_FAILED;  // error
        goto error_get_queue_3;
    }

    // buffer to fill reciving element indices
    D_INT* tmp_idx_buf_send = malloc(2 * sizeof(D_INT) * (total_send + 1));
    if (!tmp_idx_buf_send)
    {
        error = BUF_ALLOC_FAILED;  // error
        free(tmp_idx_buf_recv);
        goto error_get_queue_3;
    }

    for (D_LL_INT i = 0; i < mat->nGetQueueElements; ++i)
    {
        tmp_idx_buf_recv[2 * i] = getQueue[i].i;
        tmp_idx_buf_recv[2 * i + 1] = getQueue[i].j;
    }

    // AlltoAllv
    mpi_error = MPI_Alltoallv(tmp_idx_buf_recv, counts_recv, displacements_recv,
                              MPI_idxType, tmp_idx_buf_send, counts_send,
                              displacements_send, MPI_idxType, mat->comm);

    if (mpi_error != MPI_SUCCESS)
    {
        error = DIAGO_MPI_ERROR;
        free(tmp_idx_buf_send);
        free(tmp_idx_buf_recv);
        goto error_get_queue_3;
    }

    free(tmp_idx_buf_recv);

    D_Cmplx* send_buf = malloc(sizeof(*send_buf) * (total_send + 1));
    if (!send_buf)
    {
        error = BUF_ALLOC_FAILED;  // error
        free(tmp_idx_buf_send);
        goto error_get_queue_3;
    }
    // pack the send buffer
    for (D_LL_INT i = 0; i < total_send; ++i)
    {
        D_INT loc_idx_row = INDXG2L(tmp_idx_buf_send[2 * i], mat->block_size[0],
                                    mat->pids[0], 0, mat->pgrid[0]);
        D_INT loc_idx_col =
            INDXG2L(tmp_idx_buf_send[2 * i + 1], mat->block_size[1],
                    mat->pids[1], 0, mat->pgrid[1]);
        send_buf[i] =
            mat->data[loc_idx_row * mat->lda[0] + loc_idx_col * mat->lda[1]];
    }
    free(tmp_idx_buf_send);

    // Now send back the required elements
    D_Cmplx* recv_buf = malloc(sizeof(*recv_buf) * (total_recv + 1));
    if (!recv_buf)
    {
        error = BUF_ALLOC_FAILED;  // error
        goto error_get_queue_4;
    }
    mpi_error = MPI_Alltoallv(send_buf, counts_send, displacements_send,
                              D_Cmplx_MPI_TYPE, recv_buf, counts_recv,
                              displacements_recv, D_Cmplx_MPI_TYPE, mat->comm);
    if (mpi_error != MPI_SUCCESS)
    {
        error = DIAGO_MPI_ERROR;
        goto error_get_queue_5;
    }

    // unpack the values
    for (D_LL_INT i = 0; i < mat->nGetQueueElements; ++i)
    {
        *getQueue[i].value = recv_buf[i];
    }

error_get_queue_5:
    free(recv_buf);
error_get_queue_4:
    free(send_buf);
error_get_queue_3:
    mpi_error = MPI_Type_free(&MPI_idxType);
    if (mpi_error != MPI_SUCCESS)
    {
        error = DIAGO_MPI_ERROR;
    }
error_get_queue_2:
    free(counts_send);
error_get_queue_1:
    // once Queue is processed, reset.
    free(mat->GetQueuePtr);
    mat->GetQueuePtr = NULL;
    mat->nGetQueueElements = 0;
    mat->iget = 0;
error_get_queue_0:
    return error;
}

// Fortran version of get element
Err_INT DMatGet_fortran(void* D_mat, const D_INT i, const D_INT j,
                        D_Cmplx* value)
{
    return DMatGet(D_mat, i - 1, j - 1, value);
}
