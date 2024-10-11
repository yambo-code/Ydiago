// Contain heev solvers

#include <complex.h>
#include <ctype.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "../common/dtypes.h"
#include "../common/error.h"
#include "../common/min_max.h"
#include "../diago.h"
#include "../matrix/matrix.h"
#include "../solvers.h"
#include "scalapack_header.h"

/*=== P?heevR ===*/
Err_INT Heev(void* DmatA, char ulpo, D_INT* neigs_range, D_float* eigval_range,
             D_Cmplx* eig_vals, void* Deig_vecs, D_INT* neigs_found)
{
    /*
    Note the eig_vals buffer must have the dimension of the matrix
    and eig_vecs must have same dimension as the input matrix.
    */
    // uses p?heevR solver

    if (!eig_vals)
    {
        return ERR_NULL_PTR_BUFFER;  // NUll pointer is passed
        // This is a fatal error, better return immediately!
    }

    Err_INT error = check_mat_diago(DmatA, false);
    D_INT err_code = 0;  // error code for scalapack
    // do basic checks
    if (error)
    {
        return error;  // Fatal error, return immediately.
    }

    struct D_Matrix* matA = DmatA;
    struct D_Matrix* eig_vecs = Deig_vecs;

    // MN : FIX ME : we need to check the Deig_vecs if present

    *neigs_found = 0;  // this will be reset by the scalapack function

    // zero out the eigenvalue buffer
    for (D_LL_INT i = 0; i < matA->gdims[0]; ++i)
    {
        eig_vals[i] = 0;
    }

    if (!matA->cpu_engage)
    {
        goto Heev_end;
    }

    char jobz = 'N';
    if (eig_vecs)
    {
        jobz = 'V';
        // if null ptr is passed to Deig_vecs, only eigvals are computed.
        // FIX ME : do check to eig_vec distributed matrix
        // zero out the buffer
        error = set_zero(eig_vecs);
        if (error)
        {
            goto Heev_end;
        }
    }

    D_float vl = 0.0, vu = 0.0;
    D_INT il = 0, iu = 0;

    char range = 'A';
    if (neigs_range)
    {
        range = 'I';
        il = MIN(neigs_range[0], neigs_range[1]);
        iu = MAX(neigs_range[0], neigs_range[1]);
        if (il < 1 || iu < 1)
        {  // compute all eigen values. This is wrong config
            range = 'A';
        }
    }
    else if (eigval_range)
    {
        range = 'V';
        vl = MIN(eigval_range[0], eigval_range[1]);
        vu = MAX(eigval_range[0], eigval_range[1]);
    }
    char ulpo_tmp = toupper(ulpo);

    D_INT izero = 1;  // scalapack indices start from 1

    D_INT desca[9], descz[9];

    error = set_descriptor(matA, desca);
    if (error)
    {
        goto Heev_end;
    }

    error = set_descriptor(eig_vecs, descz);
    if (error)
    {
        goto Heev_end;
    }

    D_float* eig_tmp = calloc(matA->gdims[0], sizeof(D_float));
    if (!eig_tmp)
    {
        error = BUF_ALLOC_FAILED;
        goto Heev_end;
    }

    D_INT nevecs_found = 0;  // this will be reset by the scalapack function

    D_INT lwork = -1, lrwork = -1, liwork = -1;

    // Query request
    D_Cmplx work_tmp[4];
    D_float rwork_tmp[4];
    D_INT iwork_tmp[4];

    SL_FunCmplx(heevr)(&jobz, &range, &ulpo_tmp, matA->gdims, matA->data,
                       &izero, &izero, desca, &vl, &vu, &il, &iu, neigs_found,
                       &nevecs_found, eig_tmp, eig_vecs->data, &izero, &izero,
                       descz, work_tmp, &lwork, rwork_tmp, &lrwork, iwork_tmp,
                       &liwork, &err_code);

    if (err_code)
    {
        error = SL_WORK_SPACE_ERROR;
        goto Heev_end1;
    }

    lwork = rint(creal(work_tmp[0]) * SL_WORK_QUERY_FAC);
    lrwork = rint(rwork_tmp[0] * SL_WORK_QUERY_FAC);
    liwork = iwork_tmp[0];

    D_Cmplx* work = calloc(lwork, sizeof(*work));
    D_float* rwork = calloc(lrwork, sizeof(*rwork));
    D_INT* iwork = calloc(liwork, sizeof(*iwork));

    if (work && rwork && iwork)
    {
        SL_FunCmplx(heevr)(&jobz, &range, &ulpo_tmp, matA->gdims, matA->data,
                           &izero, &izero, desca, &vl, &vu, &il, &iu,
                           neigs_found, &nevecs_found, eig_tmp, eig_vecs->data,
                           &izero, &izero, descz, work, &lwork, rwork, &lrwork,
                           iwork, &liwork, &err_code);
    }
    else
    {
        err_code = 0;
        error = BUF_ALLOC_FAILED;
    }

    free(iwork);
    free(rwork);
    free(work);

    if (error || err_code)
    {
        if (err_code)
        {
            error = SL_DIAGO_ERROR;
        }
        nevecs_found = 0;
        goto Heev_end1;
    }

    *neigs_found = MIN(*neigs_found, nevecs_found);

    for (D_INT i = 0; i < *neigs_found; ++i)
    {
        eig_vals[i] = eig_tmp[i];
    }

Heev_end1:;
    free(eig_tmp);
Heev_end:;
    err_code = MPI_Bcast(neigs_found, 1, D_INT_MPI_TYPE, 0, matA->comm);
    if (!error && err_code)
    {
        // return the true error
        error = DIAGO_MPI_ERROR;
    }
    err_code =
        MPI_Bcast(eig_vals, *neigs_found, D_Cmplx_MPI_TYPE, 0, matA->comm);
    if (!error && err_code)
    {
        // return the true error
        error = DIAGO_MPI_ERROR;
    }

    return error;
}

// /*=== P?HEEVX ===*/
// static Err_INT HeevX(void * DmatA, char ulpo, D_INT * neigs_range,
//             D_float * eigval_range, D_Cmplx * eig_vals, void * Deig_vecs)
// {
//     // uses p?heevX solver

//     int error;
//     // do basic checks
//     if (!DmatA)
//     {   // error
//         return -1;
//     }
//     struct D_Matrix * matA = DmatA;
//     struct D_Matrix * eig_vecs = Deig_vecs;

//     D_INT err_code = 0;
//     D_INT neigs_found = 0; // this will be reset by the scalapack function

//     if (matA->gdims[0] != matA->gdims[1])
//     {
//         return -1; // error not a square matrix
//     }

//     if (matA->block_size[0] != matA->block_size[1])
//     {
//         return -1; // error block size must be square
//     }

//     if (matA->pids[0] < 0 || matA->pids[1] < 0 )
//     {
//         goto end_HeevX ;// cpu not participating in diago
//     }
//     char jobz = 'N';
//     if (eig_vecs)
//     {
//         jobz = 'V' ;
//         // if null ptr is passed to Deig_vecs, only eigvals are computed.
//     }

//     D_float vl = 0.0, vu = 0.0;
//     D_INT il = 0, iu = 0;

//     char range = 'A';
//     if (neigs_range)
//     {
//         range = 'I';
//         il = MIN(neigs_range[0],neigs_range[1]);
//         iu = MAX(neigs_range[0],neigs_range[1]);
//     }
//     else if (eigval_range)
//     {
//         range = 'V';
//         vl = MIN(eigval_range[0],eigval_range[1]);
//         vu = MAX(eigval_range[0],eigval_range[1]);
//     }
//     char ulpo_tmp = toupper(ulpo);

//     D_INT izero = 1; // scalapack indices start from 1

//     D_INT desca[9], descz[9];

//     if (set_descriptor(matA, desca))
//     {
//         return -3;
//     }
//     if (set_descriptor(eig_vecs,descz))
//     {
//         return -3;
//     }

//     D_float abstol = 0.0; // ! MN : FIX ME! set it to the 2*machine_prec

//     D_float* eig_tmp = calloc(matA->gdims[0],sizeof(D_float));

//     D_INT nevecs_found = 0; // this will be reset by the scalapack function

//     D_float orfac = 1e-6; // ! MN : FIX ME

//     D_INT nproc_grid = matA->pgrid[0]*matA->pgrid[1];

//     D_float* gap = calloc(nproc_grid, sizeof(*gap));
//     D_INT* iclustr = calloc(2*nproc_grid,sizeof(*iclustr));
//     D_INT* ifail = calloc(matA->gdims[0],sizeof(*ifail));

//     D_INT lwork = -1, lrwork = -1, liwork = -1;

//     // Query request
//     D_Cmplx  work_tmp[4];
//     D_float rwork_tmp[4];
//     D_INT   iwork_tmp[4];

//     SL_FunCmplx(heevx)(&jobz, &range, &ulpo_tmp, matA->gdims,
//         matA->data, &izero, &izero, desca, &vl, &vu, &il, &iu, &abstol,
//         &neigs_found, &nevecs_found, eig_tmp, &orfac, eig_vecs->data, &izero,
//         &izero, descz, work_tmp, &lwork, rwork_tmp, &lrwork, iwork_tmp,
//         &liwork, ifail, iclustr, gap, &err_code);

//     lwork  = rint(creal(work_tmp[0])*SL_WORK_QUERY_FAC);
//     lrwork = rint(rwork_tmp[0]*SL_WORK_QUERY_FAC);
//     liwork = iwork_tmp[0];

//     D_Cmplx*  work  = calloc(lwork ,sizeof(*work));
//     D_float* rwork  = calloc(lrwork,sizeof(*rwork));
//     D_INT*   iwork  = calloc(liwork,sizeof(*iwork));

//     SL_FunCmplx(heevx)(&jobz, &range, &ulpo_tmp, matA->gdims,
//         matA->data, &izero, &izero, desca, &vl, &vu, &il, &iu, &abstol,
//         &neigs_found, &nevecs_found, eig_tmp, &orfac, eig_vecs->data, &izero,
//         &izero, descz, work, &lwork, rwork, &lrwork, iwork, &liwork, ifail,
//         iclustr, gap, &err_code);

//     free(iwork);
//     free(rwork);
//     free(work);

//     free(ifail);
//     free(iclustr);
//     free(gap);

//     for (D_INT i = 0; i < neigs_found; ++i)
//     {
//         eig_vals[i] = eig_tmp[i];
//     }

//     free(eig_tmp);

//     end_HeevX :
//         ;

//     error = MPI_Bcast(&neigs_found, 1, D_INT_MPI_TYPE, 0, matA->comm);
//     error = MPI_Bcast(eig_vals, neigs_found, D_Cmplx_MPI_TYPE, 0,
//     matA->comm);

//     return 0;

// }

// /*=== P?HEEVD ===*/
// static Err_INT HeevD(void * DmatA, char ulpo, D_INT * neigs_range,
//             D_float * eigval_range, D_Cmplx * eig_vals, void * Deig_vecs)
// {
//     // uses p?heevD solver

//     int error;
//     // do basic checks
//     if (!DmatA)
//     {   // error
//         return -1;
//     }
//     struct D_Matrix * matA = DmatA;
//     struct D_Matrix * eig_vecs = Deig_vecs;

//     D_INT err_code = 0;

//     if (matA->gdims[0] != matA->gdims[1])
//     {
//         return -1; // error not a square matrix
//     }

//     if (matA->block_size[0] != matA->block_size[1])
//     {
//         return -1; // error block size must be square
//     }

//     if (matA->pids[0] < 0 || matA->pids[1] < 0 )
//     {
//         goto end_HeevD ;// cpu not participating in diago
//     }
//     char jobz = 'N';
//     if (eig_vecs)
//     {
//         jobz = 'V' ;
//         // if null ptr is passed to Deig_vecs, only eigvals are computed.
//     }

//     char ulpo_tmp = toupper(ulpo);

//     D_INT izero = 1; // scalapack indices start from 1

//     D_INT desca[9], descz[9];

//     if (set_descriptor(matA, desca))
//     {
//         return -3;
//     }
//     if (set_descriptor(eig_vecs,descz))
//     {
//         return -3;
//     }

//     D_float* eig_tmp = calloc(matA->gdims[0],sizeof(D_float));

//     D_INT lwork = -1, lrwork = -1, liwork = -1;

//     D_Cmplx  work_tmp[4];
//     D_float rwork_tmp[4];
//     D_INT   iwork_tmp[4];

//     // Query request
//     SL_FunCmplx(heevd)(&jobz, &ulpo_tmp, matA->gdims, matA->data,
//         &izero, &izero, desca, eig_tmp, eig_vecs->data,
//         &izero, &izero, descz, work_tmp,
//         &lwork, rwork_tmp, &lrwork, iwork_tmp, &liwork, &err_code);

//     lwork  = rint(creal(work_tmp[0])*SL_WORK_QUERY_FAC);
//     lrwork = rint(rwork_tmp[0]*SL_WORK_QUERY_FAC);
//     liwork = iwork_tmp[0];

//     D_Cmplx*  work  = calloc(lwork ,sizeof(*work));
//     D_float* rwork  = calloc(lrwork,sizeof(*rwork));
//     D_INT*   iwork  = calloc(liwork,sizeof(*iwork));

//     SL_FunCmplx(heevd)(&jobz, &ulpo_tmp, matA->gdims, matA->data,
//         &izero, &izero, desca, eig_tmp, eig_vecs->data,
//         &izero, &izero, descz, work,
//         &lwork, rwork, &lrwork, iwork, &liwork, &err_code);

//     free(iwork);
//     free(rwork);
//     free(work);

//     for (D_INT i = 0; i < matA->gdims[0]; ++i)
//     {
//         eig_vals[i] = eig_tmp[i];
//     }

//     free(eig_tmp);

//     end_HeevD :
//         ;

//     error = MPI_Bcast(eig_vals, matA->gdims[0], D_Cmplx_MPI_TYPE, 0,
//     matA->comm);

//     return 0;

// }
