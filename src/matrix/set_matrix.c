// THis file contains functions that fills elements in the
// global distribution matrix
//
//
//
#include "matrix.h"
#include "../diago.h"
#include <mpi.h>
#include "../common/error.h"
#include "../common/dtypes.h"
#include <stdlib.h>
#include "../SL/scalapack_header.h"
#include <string.h>

#ifdef MPI_Aint_diff
#define MPI_Aint_diff_tmp(addr1, addr2) MPI_Aint_diff(addr1, addr2)
#else
#define MPI_Aint_diff_tmp(addr1, addr2) ((MPI_Aint)((char*)(addr1) - (char*)(addr2)))
#endif
// This is equvalent function of MPI_Aint_diff
// From MPI standard 3.1, there is MPI_Aint_diff.
// this created simple for backward compatibility

struct SetElement
{
    D_Cmplx value;
    D_INT i; // global row index
    D_INT j; // global col index
};

static D_INT* Set_mapP2iD; // maps (iproc,jproc)->mpirank. used in Set_idxG2iD
static int cmpSetElements(const void* a, const void* b);

static inline D_INT Set_idxG2iD(const D_INT* setCmpPrms, const D_INT i, const D_INT j)
{
    // setCmpPrms is array[5]{cxt, BlkX, BlkY, NProcX, NProcY}
    D_INT prow = INDXG2P(i, setCmpPrms[1], 0, 0, setCmpPrms[3]);
    D_INT pcol = INDXG2P(j, setCmpPrms[2], 0, 0, setCmpPrms[4]);
    return Set_mapP2iD[prow * setCmpPrms[4] + pcol];
}

Err_INT initiateSetQueue(void* D_mat, const D_LL_INT nelements)
{
    if (!D_mat)
    {
        return MATRIX_NOT_INIT; // error.
    }
    struct D_Matrix* mat = D_mat;

    int mpi_error = MPI_Barrier(mat->comm); // make this a collective call
    if (mpi_error != MPI_SUCCESS)
    {
        return DIAGO_MPI_ERROR;
    }

    if (mat->SetQueuePtr)
    {
        return QUEUE_ALREADY_INIT; // error. Queue already initated.
    }
    mat->nSetQueueElements = nelements;
    mat->iset = 0;

    mat->SetQueuePtr = malloc(sizeof(struct SetElement) * (nelements + 1));
    // Always alloc aleast one element to check if Queue is active
    if (!mat->SetQueuePtr)
    {
        return BUF_ALLOC_FAILED;
    }

    return DIAGO_SUCCESS;
}

Err_INT DMatSet(void* D_mat, const D_INT i, const D_INT j, const D_Cmplx value)
{
    if (!D_mat)
    {
        return MATRIX_NOT_INIT; // error.
    }

    struct D_Matrix* mat = D_mat;

    if (i >= mat->gdims[0] || j >= mat->gdims[1])
    {
        // error : out of bounds
        return OUT_OF_BOUNDS_ERROR;
    }

    if (!mat->SetQueuePtr)
    {
        return QUEUE_NOT_INIT; // error not initated
    }
    if (mat->iset >= mat->nSetQueueElements)
    {
        return QUEUE_LIMIT_REACHED; // error out of bounds
    }
    struct SetElement* QueuePtr = mat->SetQueuePtr;

    QueuePtr[mat->iset].i = i;
    QueuePtr[mat->iset].j = j;
    QueuePtr[mat->iset].value = value;

    ++mat->iset;

    return 0;
}

Err_INT ProcessSetQueue(void* D_mat)
{
    // all comm cpus must call this

    Err_INT error = DIAGO_SUCCESS;

    if (!D_mat)
    {
        error = MATRIX_NOT_INIT; // error.
        goto error_set_queue_0;
    }
    struct D_Matrix* mat = D_mat;

    if (!mat->SetQueuePtr)
    {
        error = QUEUE_NOT_INIT; // error not initated
        goto error_set_queue_1;
    }

    // First, set the Set_mapP2iD pointer;
    Set_mapP2iD = mat->mapP2iD;

    // next we need to sort the SetQueue array based on the ranks
    const D_INT cmpSetupEle[5] = { mat->blacs_ctxt, mat->block_size[0],
                                   mat->block_size[1], mat->pgrid[0],
                                   mat->pgrid[1] };
    if (mat->nSetQueueElements > 0)
    {
        // setup the comparison function
        cmpSetElements(NULL, cmpSetupEle);
        // Now sort
        qsort(mat->SetQueuePtr, mat->nSetQueueElements, sizeof(struct SetElement), cmpSetElements);
    }

    // some MPI related stuff
    int TotalCommCpus;
    int mpi_error = MPI_Comm_size(mat->comm, &TotalCommCpus);
    if (mpi_error != MPI_SUCCESS)
    {
        error = DIAGO_MPI_ERROR;
        goto error_set_queue_1;
    }

    int* counts_send = calloc(4 * TotalCommCpus, sizeof(*counts_send));
    if (!counts_send)
    {
        error = BUF_ALLOC_FAILED; // error
        goto error_set_queue_1;
    }
    int* displacements_send = counts_send + TotalCommCpus * 1;
    int* counts_recv = counts_send + TotalCommCpus * 2;
    int* displacements_recv = counts_send + TotalCommCpus * 3;

    for (D_LL_INT i = 0; i < TotalCommCpus; ++i)
    {
        counts_send[i] = 0;
    }

    struct SetElement* setQueue = mat->SetQueuePtr;
    for (D_LL_INT i = 0; i < mat->nSetQueueElements; ++i)
    {
        D_INT tmp_rank = Set_idxG2iD(cmpSetupEle, setQueue[i].i, setQueue[i].j);
        ++counts_send[tmp_rank];
    }
    mpi_error = MPI_Alltoall(counts_send, 1, MPI_INT, counts_recv, 1, MPI_INT, mat->comm);
    if (mpi_error != MPI_SUCCESS)
    {
        error = DIAGO_MPI_ERROR;
        goto error_set_queue_2;
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

    MPI_Datatype MPI_SetElement; // MPI type for SetElement

    int tmp_lengths[3] = { 1, 1, 1 };
    MPI_Datatype tmp_types[3] = { D_Cmplx_MPI_TYPE, D_INT_MPI_TYPE, D_INT_MPI_TYPE };
    MPI_Aint tmpdisps[3];
    MPI_Aint base_address;
    struct SetElement tmp_dummy;

    mpi_error = MPI_Get_address(&tmp_dummy, &base_address);
    if (mpi_error != MPI_SUCCESS)
    {
        error = DIAGO_MPI_ERROR;
        goto error_set_queue_2;
    }

    mpi_error = MPI_Get_address(&tmp_dummy.value, &tmpdisps[0]);
    if (mpi_error != MPI_SUCCESS)
    {
        error = DIAGO_MPI_ERROR;
        goto error_set_queue_2;
    }

    mpi_error = MPI_Get_address(&tmp_dummy.i, &tmpdisps[1]);
    if (mpi_error != MPI_SUCCESS)
    {
        error = DIAGO_MPI_ERROR;
        goto error_set_queue_2;
    }

    mpi_error = MPI_Get_address(&tmp_dummy.j, &tmpdisps[2]);
    if (mpi_error != MPI_SUCCESS)
    {
        error = DIAGO_MPI_ERROR;
        goto error_set_queue_2;
    }
    tmpdisps[0] = MPI_Aint_diff_tmp(tmpdisps[0], base_address);
    tmpdisps[1] = MPI_Aint_diff_tmp(tmpdisps[1], base_address);
    tmpdisps[2] = MPI_Aint_diff_tmp(tmpdisps[2], base_address);

    mpi_error = MPI_Type_create_struct(3, tmp_lengths, tmpdisps, tmp_types, &MPI_SetElement);
    if (mpi_error != MPI_SUCCESS)
    {
        error = DIAGO_MPI_ERROR;
        goto error_set_queue_2;
    }

    mpi_error = MPI_Type_commit(&MPI_SetElement);
    if (mpi_error != MPI_SUCCESS)
    {
        error = DIAGO_MPI_ERROR;
        goto error_set_queue_2;
    }

    struct SetElement* recv_buf = malloc(sizeof(*recv_buf) * (total_recv + 1));
    if (!recv_buf)
    {
        error = BUF_ALLOC_FAILED;
        goto error_set_queue_3;
    }

    mpi_error = MPI_Alltoallv(mat->SetQueuePtr, counts_send,
                              displacements_send, MPI_SetElement,
                              recv_buf, counts_recv, displacements_recv,
                              MPI_SetElement, mat->comm);
    if (mpi_error != MPI_SUCCESS)
    {
        error = DIAGO_MPI_ERROR;
        goto error_set_queue_4;
    }
    // unpack the data and set it locally
    for (D_LL_INT i = 0; i < total_recv; ++i)
    {
        D_INT loc_idx_row = INDXG2L(recv_buf[i].i, mat->block_size[0], mat->pids[0], 0, mat->pgrid[0]);
        D_INT loc_idx_col = INDXG2L(recv_buf[i].j, mat->block_size[1], mat->pids[1], 0, mat->pgrid[1]);
        mat->data[loc_idx_row * mat->lda[0] + loc_idx_col * mat->lda[1]] = recv_buf[i].value;
    }

error_set_queue_4:
    free(recv_buf);
error_set_queue_3:
    mpi_error = MPI_Type_free(&MPI_SetElement);
    if (mpi_error != MPI_SUCCESS)
    {
        error = DIAGO_MPI_ERROR;
    }
error_set_queue_2:
    free(counts_send);
error_set_queue_1:
    // once Queue is processed, reset.
    free(mat->SetQueuePtr);
    mat->SetQueuePtr = NULL;
    mat->nSetQueueElements = 0;
    mat->iset = 0;
error_set_queue_0:
    return error;
}

// Fortran interface
Err_INT DMatSet_fortran(void* D_mat, const D_INT i, const D_INT j, const D_Cmplx value)
{
    return DMatSet(D_mat, i - 1, j - 1, value);
}

// compartor for qsort
static int cmpSetElements(const void* a, const void* b)
{
    /*
    This is a compartor function used only in ProcessSetQueue
    function to sort the SetQueue elements according to the
    processor iD.
    Note : Before calling qsort. one must set the setCmpParams.
    THis can be done by setting a = NULL and passing D_INT[5] array
    to b
    */
    static D_INT setCmpParams[5];
    // {cxt, BlkX, BlkY, NProcX, NProcY}
    /*
    cxt : Context
    BlkX,BlkY : block cyclic laylout block size
    NProcX, NProcY : Total number of processors in the grid
    */
    if (!a)
    {
        if (b)
        {
            memcpy(setCmpParams, b, sizeof(D_INT) * 5);
        }
        return 0;
    }
    const struct SetElement* arg1 = a;
    const struct SetElement* arg2 = b;
    // find the processor ids of arg1 and arg2 and sort
    D_INT rank1 = Set_idxG2iD(setCmpParams, arg1->i, arg1->j);
    D_INT rank2 = Set_idxG2iD(setCmpParams, arg2->i, arg2->j);

    return (rank1 > rank2) - (rank1 < rank2);
}
