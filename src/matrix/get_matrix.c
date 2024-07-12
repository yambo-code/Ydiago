// THis file contains functions that gets elements from the
// global distribution matrix
//
#include "matrix.h"

struct GetElement
{
    D_Cmplx* value;
    D_INT i; // global row index
    D_INT j; // global col index
};

static D_INT* Get_mapP2iD; // maps (iproc,jproc)->mpirank. used in Get_idxG2iD
static int cmpGetElements(const void* a, const void* b);

static inline D_INT Get_idxG2iD(const D_INT* getCmpPrms, const D_INT i, const D_INT j)
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
        return MATRIX_NOT_INIT; // error.
    }
    struct D_Matrix* mat = D_mat;

    int mpi_error = MPI_Barrier(mat->comm); // make this a collective call
    if (mpi_error != MPI_SUCCESS)
    {
        return DIAGO_MPI_ERROR;
    }

    if (mat->GetQueuePtr)
    {
        return QUEUE_ALREADY_INIT; // error. Queue already initated.
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
        return MATRIX_NOT_INIT; // error.
    }

    struct D_Matrix* mat = D_mat;

    if (i >= mat->gdims[0] || j >= mat->gdims[1])
    {
        // error : out of bounds
        return OUT_OF_BOUNDS_ERROR;
    }

    if (!mat->GetQueuePtr)
    {
        return QUEUE_NOT_INIT; // error not initated
    }
    if (mat->iget >= mat->nGetQueueElements)
    {
        return QUEUE_LIMIT_REACHED; // error out of bounds
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
        error = MATRIX_NOT_INIT; // error.
        goto error_get_queue_0;
    }

    struct D_Matrix* mat = D_mat;
    if (!mat->GetQueuePtr)
    {
        error = QUEUE_NOT_INIT; // error not initated
        goto error_get_queue_1;
    }

    // First, set the Get_mapP2iD pointer;
    Get_mapP2iD = mat->mapP2iD;

    // next we need to sort the GetQueue array based on the ranks
    const D_INT cmpGetupEle[5] = { mat->blacs_ctxt, mat->block_size[0],
                                   mat->block_size[1], mat->pgrid[0],
                                   mat->pgrid[1] };
    if (mat->nGetQueueElements > 0)
    {
        // setup the comparison function
        cmpGetElements(NULL, cmpGetupEle);
        // Now sort
        qsort(mat->GetQueuePtr, mat->nGetQueueElements, sizeof(struct GetElement), cmpGetElements);
    }

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
        error = BUF_ALLOC_FAILED; // error
        goto error_get_queue_1;
    }
    int* displacements_send = counts_send + TotalCommCpus * 1;
    int* counts_recv = counts_send + TotalCommCpus * 2;
    int* displacements_recv = counts_send + TotalCommCpus * 3;

    for (D_LL_INT i = 0; i < TotalCommCpus; ++i)
    {
        counts_recv[i] = 0;
    }

    struct GetElement* getQueue = mat->GetQueuePtr;
    for (D_LL_INT i = 0; i < mat->nGetQueueElements; ++i)
    {
        D_INT tmp_rank = Get_idxG2iD(cmpGetupEle, getQueue[i].i, getQueue[i].j);
        ++counts_recv[tmp_rank];
    }

    mpi_error = MPI_Alltoall(counts_recv, 1, MPI_INT, counts_send, 1, MPI_INT, mat->comm);
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
        error = BUF_ALLOC_FAILED; // error
        goto error_get_queue_3;
    }

    // buffer to fill reciving element indices
    D_INT* tmp_idx_buf_send = malloc(2 * sizeof(D_INT) * (total_send + 1));
    if (!tmp_idx_buf_send)
    {
        error = BUF_ALLOC_FAILED; // error
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
                              MPI_idxType, tmp_idx_buf_send, counts_send, displacements_send,
                              MPI_idxType, mat->comm);

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
        error = BUF_ALLOC_FAILED; // error
        free(tmp_idx_buf_send);
        goto error_get_queue_3;
    }
    // pack the send buffer
    for (D_LL_INT i = 0; i < total_send; ++i)
    {
        D_INT loc_idx_row = INDXG2L(tmp_idx_buf_send[2 * i], mat->block_size[0], mat->pids[0], 0, mat->pgrid[0]);
        D_INT loc_idx_col = INDXG2L(tmp_idx_buf_send[2 * i + 1], mat->block_size[1], mat->pids[1], 0, mat->pgrid[1]);
        send_buf[i] = mat->data[loc_idx_row * mat->lda[0] + loc_idx_col * mat->lda[1]];
    }
    free(tmp_idx_buf_send);

    // Now send back the required elements
    D_Cmplx* recv_buf = malloc(sizeof(*recv_buf) * (total_recv + 1));
    if (!recv_buf)
    {
        error = BUF_ALLOC_FAILED; // error
        goto error_get_queue_4;
    }
    mpi_error = MPI_Alltoallv(send_buf, counts_send,
                              displacements_send, D_Cmplx_MPI_TYPE,
                              recv_buf, counts_recv, displacements_recv,
                              D_Cmplx_MPI_TYPE, mat->comm);
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
Err_INT DMatGet_fortran(void* D_mat, const D_INT i, const D_INT j, D_Cmplx* value)
{
    return DMatGet(D_mat, i - 1, j - 1, value);
}

// compartor for qsort
static int cmpGetElements(const void* a, const void* b)
{
    /*
    This is a compartor function used only in ProcessGetQueue
    function to sort the GetQueue elements according to the
    processor iD.
    Note : Before calling qsort. one must Get the GetCmpParams.
    THis can be done by Getting a = NULL and passing D_INT[5] array
    to b
    */
    static D_INT GetCmpParams[5];
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
            memcpy(GetCmpParams, b, sizeof(D_INT) * 5);
        }
        return 0;
    }
    const struct GetElement* arg1 = a;
    const struct GetElement* arg2 = b;
    // find the processor ids of arg1 and arg2 and sort
    D_INT rank1 = Get_idxG2iD(GetCmpParams, arg1->i, arg1->j);
    D_INT rank2 = Get_idxG2iD(GetCmpParams, arg2->i, arg2->j);

    return (rank1 > rank2) - (rank1 < rank2);
}
