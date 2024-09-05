// THis file contains functions that initiate the
// block cyclic matrix
//
//
//
#include "matrix.h"
#include "../SL/scalapack_header.h"
#include "../diago.h"
#include <mpi.h>
#include "../common/error.h"
#include "../common/dtypes.h"
#include <stdlib.h>
#include <ctype.h>

void* BLACScxtInit(char layout, MPI_Comm comm, D_INT ProcX, D_INT ProcY)
{
    // initiate a blacs context
    struct MPIcxt* contxt = malloc(sizeof(struct MPIcxt));
    if (!contxt)
    {
        return NULL;
    }
    contxt->comm = comm;
    contxt->pgrid[0] = ProcX;
    contxt->pgrid[1] = ProcY;
    contxt->ctxt = Csys2blacs_handle(comm);

    if (tolower(layout) == 'c')
    {
        Cblacs_gridinit(&contxt->ctxt, "C", ProcX, ProcY);
        contxt->layout = 'C';
    }
    else
    {
        Cblacs_gridinit(&contxt->ctxt, "R", ProcX, ProcY);
        contxt->layout = 'R';
    }
    return contxt;
};

void* BLACScxtInit_Fortran(char layout, D_INT comm, D_INT ProcX, D_INT ProcY)
{
    // fortran interface
    return BLACScxtInit(layout, MPI_Comm_f2c(comm), ProcX, ProcY);
};

void BLACScxtFree(void* mpicontxt)
{
    if (!mpicontxt)
    {
        return;
    }
    struct MPIcxt* mcxt = mpicontxt;
    int myid;
    int mpi_error = MPI_Comm_rank(mcxt->comm, &myid);
    if (myid < mcxt->pgrid[0] * mcxt->pgrid[1])
    {
        Cblacs_gridexit(mcxt->ctxt);
    }
    free(mpicontxt);
}

void* init_D_Matrix(D_INT Grows, D_INT Gcols,
                    D_INT blockX, D_INT blockY, void* mpicontxt)
{
    // this function must be called by all the CPUS participating
    // in the comm
    int comm_size;
    struct MPIcxt* mcxt = mpicontxt;
    if (!mcxt)
    {
        goto error_1;
    }

    MPI_Comm comm = mcxt->comm;

    D_INT ProcX = mcxt->pgrid[0];
    D_INT ProcY = mcxt->pgrid[1];

    int mpi_error = MPI_Comm_size(comm, &comm_size);
    if (mpi_error != MPI_SUCCESS)
    {
        goto error_1;
    }

    if (ProcX * ProcY > comm_size)
    {
        goto error_1;
    }

    // must free the data of the return value manually
    struct D_Matrix* mat = malloc(sizeof(*mat));

    if (!mat)
    {
        goto error_1;
    }

    mat->comm = mcxt->comm;
    mat->blacs_ctxt = mcxt->ctxt;

    mat->mapP2iD = malloc(ProcX * ProcY * sizeof(*mat->mapP2iD));
    if (!mat->mapP2iD)
    {
        goto error_2;
    }

    mat->cpu_engage = true;

    D_INT myid, myrow, mycol;

    mpi_error = MPI_Comm_rank(comm, &myid);
    if (mpi_error != MPI_SUCCESS)
    {
        goto error_3;
    }

    if (myid < (ProcX * ProcY))
    {
        D_INT tmp_ProcX, tmp_ProcY;
        Cblacs_gridinfo(mat->blacs_ctxt, &tmp_ProcX, &tmp_ProcY, &myrow, &mycol);
        if (tmp_ProcX != ProcX || tmp_ProcY != ProcY)
        {
            goto error_3;
        }
    }
    else
    {
        mat->cpu_engage = false; // cpu is not in th process grid operations
        myrow = -1;
        mycol = -1;
    }

    D_INT* tmp_row_arr = malloc(2 * comm_size * sizeof(D_INT));
    if (!tmp_row_arr)
    {
        goto error_3;
    }

    D_INT* tmp_col_arr = tmp_row_arr + comm_size;
    mpi_error = MPI_Allgather(&myrow, 1, D_INT_MPI_TYPE, tmp_row_arr, 1, D_INT_MPI_TYPE, comm);
    if (mpi_error != MPI_SUCCESS)
    {
        goto error_4;
    }

    mpi_error = MPI_Allgather(&mycol, 1, D_INT_MPI_TYPE, tmp_col_arr, 1, D_INT_MPI_TYPE, comm);
    if (mpi_error != MPI_SUCCESS)
    {
        goto error_4;
    }

    for (D_INT i = 0; i < (ProcX * ProcY); ++i)
    {
        D_INT irow = tmp_row_arr[i];
        D_INT jcol = tmp_col_arr[i];
        mat->mapP2iD[irow * ProcY + jcol] = i;
    }

    free(tmp_row_arr);

    mat->pgrid[0] = ProcX;
    mat->pgrid[1] = ProcY;

    mat->gdims[0] = Grows;
    mat->gdims[1] = Gcols;

    mat->block_size[0] = blockX;
    mat->block_size[1] = blockY;

    D_INT tmp_zero = 0;
    if (myrow < 0 || mycol < 0)
    {
        // This is important to set them to 0
        mat->ldims[0] = 0;
        mat->ldims[1] = 0;
    }
    else
    {
        mat->ldims[0] = numroc_(&Grows, &blockX, &myrow, &tmp_zero, &ProcX);
        mat->ldims[1] = numroc_(&Gcols, &blockY, &mycol, &tmp_zero, &ProcY);
    }

    mat->pids[0] = myrow;
    mat->pids[1] = mycol;

    // scalapack is coloumn major, so we always use coloumn major layout
    mat->lda[0] = 1;
    mat->lda[1] = mat->ldims[0];

    if (myrow < 0 || mycol < 0)
    {
        mat->lda[0] = 0;
        mat->lda[1] = 0;
    }

    mat->data = malloc((mat->ldims[0] * mat->ldims[1] + 1) * sizeof(D_Cmplx));
    // atleast allocate 1 element to check if the buffer if allocated
    if (!mat->data)
    {
        goto error_3;
    }
    mat->SetQueuePtr = NULL; // initiate the Queue pointer to NULL
    mat->GetQueuePtr = NULL; // initiate the Queue pointer to NULL
    mat->nSetQueueElements = 0;
    mat->nGetQueueElements = 0;
    mat->iset = 0;
    mat->iget = 0;

    return mat;

// errors
error_4:
    free(tmp_row_arr);
error_3:
    free(mat->mapP2iD);
error_2:
    free(mat);
error_1:
    return NULL;
}

void free_D_Matrix(void* D_mat)
{
    if (!D_mat)
    {
        return;
    }
    struct D_Matrix* mat = D_mat;

    free(mat->mapP2iD);
    mat->mapP2iD = NULL;

    free(mat->data);
    mat->data = NULL;

    free(mat->SetQueuePtr);
    mat->SetQueuePtr = NULL;

    free(mat->GetQueuePtr);
    mat->GetQueuePtr = NULL;

    free(D_mat);
}
