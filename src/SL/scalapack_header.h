#pragma once
#include <mpi.h>
#include "../diago.h"

// pblacs routines
void Cblacs_pinfo(D_INT*, D_INT*);
void Cblacs_get(D_INT, D_INT, D_INT*);
void Cblacs_gridinit(D_INT*, const char*, D_INT, D_INT);
void Cblacs_gridinfo(D_INT, D_INT*, D_INT*, D_INT*, D_INT*);
void Cblacs_pcoord(D_INT, D_INT, D_INT*, D_INT*);
void Cblacs_gridexit(D_INT);
D_INT Cblacs_pnum(D_INT ictxt, D_INT prow, D_INT pcol);
D_INT Csys2blacs_handle(MPI_Comm);
D_INT numroc_(D_INT*, D_INT*, D_INT*, D_INT*, D_INT*);

#define INDXG2L(iglob, nb, iproc, isrcproc, nprocs) \
    ((nb) * (((iglob)) / ((nb) * (nprocs))) + (((iglob)) % (nb)))

#define INDXL2G(iloc, nb, iproc, isrcproc, nprocs) \
    ((nprocs) * (nb) * (((iloc)) / (nb)) + (((iloc)) % (nb)) + (((nprocs) + (iproc) - (isrcproc)) % (nprocs)) * (nb))

#define INDXG2P(iglob, nb, iproc, isrcproc, nprocs) \
    (((isrcproc) + ((iglob)) / (nb)) % (nprocs))

void SL_FunFloat(gemr2d)(D_INT* m, D_INT* n, D_float* a,
                         D_INT* ia, D_INT* ja, D_INT* desca,
                         D_float* b, D_INT* ib, D_INT* jb,
                         D_INT* descb, D_INT* ictxt);

void SL_FunFloat(trmr2d)(char* uplo, char* diag, D_INT* m, D_INT* n,
                         D_float* a, D_INT* ia, D_INT* ja, D_INT* desca,
                         D_float* b, D_INT* ib, D_INT* jb, D_INT* descb,
                         D_INT* ictxt);

void SL_FunFloat(trmm)(char* side, char* uplo, char* transa,
                       char* diag, D_INT* m, D_INT* n,
                       D_float* alpha, D_float* a, D_INT* ia,
                       D_INT* ja, D_INT* desca, D_float* b,
                       D_INT* ib, D_INT* jb, D_INT* descb);

void SL_FunFloat(tran)(D_INT* m, D_INT* n, D_float* alpha, D_float* a,
                       D_INT* ia, D_INT* ja, D_INT* desca, D_float* beta,
                       D_float* c, D_INT* ic, D_INT* jc, D_INT* descc);

void SL_FunFloat(potrf)(char* uplo, D_INT* n, D_float* a,
                        D_INT* ia, D_INT* ja, D_INT* desca, D_INT* info);

void SL_FunFloat(lasrt)(char* id, D_INT* n, D_float* d, D_float* q,
                        D_INT* iq, D_INT* jq, D_INT* descq, D_float* work,
                        D_INT* lwork, D_INT* iwork, D_INT* liwork, D_INT* info);

void SL_FunFloat(tradd)(char* uplo, char* trans, D_INT* m, D_INT* n,
                        D_float* alpha, D_float* a, D_INT* ia, D_INT* ja,
                        D_INT* desca, D_float* beta, D_float* c, D_INT* ic,
                        D_INT* jc, D_INT* descc);

void SL_FunCmplx(gemm)(char* transa, char* transb, D_INT* m, D_INT* n,
                       D_INT* k, D_Cmplx* alpha, D_Cmplx* a, D_INT* ia,
                       D_INT* ja, D_INT* desca, D_Cmplx* b, D_INT* ib,
                       D_INT* jb, D_INT* descb, D_Cmplx* beta, D_Cmplx* c,
                       D_INT* ic, D_INT* jc, D_INT* descc);

void SL_FunCmplx(heevx)(char* jobz, char* range, char* uplo, D_INT* n,
                        D_Cmplx* a, D_INT* ia, D_INT* ja, D_INT* desca, D_float* vl,
                        D_float* vu, D_INT* il, D_INT* iu, D_float* abstol, D_INT* m,
                        D_INT* nz, D_float* w, D_float* orfac, D_Cmplx* z, D_INT* iz,
                        D_INT* jz, D_INT* descz, D_Cmplx* work, D_INT* lwork,
                        D_float* rwork, D_INT* lrwork, D_INT* iwork,
                        D_INT* liwork, D_INT* ifail, D_INT* iclustr,
                        D_float* gap, D_INT* info);

void SL_FunCmplx(heevd)(char* jobz, char* uplo, D_INT* n, D_Cmplx* a,
                        D_INT* ia, D_INT* ja, D_INT* desca, D_float* w, D_Cmplx* z,
                        D_INT* iz, D_INT* jz, D_INT* descz, D_Cmplx* work,
                        D_INT* lwork, D_float* rwork, D_INT* lrwork,
                        D_INT* iwork, D_INT* liwork, D_INT* info);

void SL_FunCmplx(heevr)(char* jobz, char* range, char* uplo, D_INT* n,
                        D_Cmplx* a, D_INT* ia, D_INT* ja, D_INT* desca, D_float* vl,
                        D_float* vu, D_INT* il, D_INT* iu, D_INT* m, D_INT* nz,
                        D_float* w, D_Cmplx* z, D_INT* iz, D_INT* jz,
                        D_INT* descz, D_Cmplx* work, D_INT* lwork,
                        D_float* rwork, D_INT* lrwork, D_INT* iwork,
                        D_INT* liwork, D_INT* info);

void SL_FunCmplx(gehrd)(D_INT* n, D_INT* ilo, D_INT* ihi,
                        D_Cmplx* a, D_INT* ia, D_INT* ja, D_INT* desca,
                        D_Cmplx* tau, D_Cmplx* work, D_INT* lwork, D_INT* info);

void SL_FunCmplx(unmhr)(char* side, char* trans, D_INT* m, D_INT* n,
                        D_INT* ilo, D_INT* ihi, D_Cmplx* a, D_INT* ia, D_INT* ja,
                        D_INT* desca, D_Cmplx* tau, D_Cmplx* c, D_INT* ic,
                        D_INT* jc, D_INT* descc, D_Cmplx* work, D_INT* lwork, D_INT* info);

void SL_FunCmplx(lahqr)(D_INT* wantt, D_INT* wantz, D_INT* n, D_INT* ilo, D_INT* ihi,
                        D_Cmplx* a, D_INT* desca, D_Cmplx* w, D_INT* iloz, D_INT* ihiz,
                        D_Cmplx* z, D_INT* descz, D_Cmplx* work, D_INT* lwork,
                        D_INT* iwork, D_INT* ilwork, D_INT* info);

void SL_FunCmplx(trevc)(char* side, char* howmny, D_INT* select, D_INT* n,
                        D_Cmplx* t, D_INT* desct, D_Cmplx* vl, D_INT* descvl,
                        D_Cmplx* vr, D_INT* descvr, D_INT* mm, D_INT* m, D_Cmplx* work,
                        D_float* rwork, D_INT* info);

void SL_FunCmplx(scal)(D_INT* n, D_Cmplx* a, D_Cmplx* x, D_INT* ix,
                       D_INT* jx, D_INT* descx, D_INT* incx);

void SL_FunCmplx(elset)(D_Cmplx* a, D_INT* i, D_INT* j, D_INT* desca, D_Cmplx* alpha);

void SL_FunCmplx(getrf)(D_INT* m, D_INT* n, D_Cmplx* a, D_INT* ia, D_INT* ja, 
                        D_INT* desca, D_INT* ipiv, D_INT* info);

void SL_FunCmplx(getri)(D_INT* n, D_Cmplx* a, D_INT* ia, D_INT* ja, D_INT* desca, 
                        D_INT* ipiv, D_Cmplx* work, D_INT* lwork, D_INT* iwork, 
                        D_INT* liwork, D_INT* info);


#ifdef WITH_DOUBLE
#define SLvec_norm2 pdznrm2_
#else
#define SLvec_norm2 pscnrm2_
#endif
void SLvec_norm2(D_INT* n, D_float* norm2, D_Cmplx* x,
                 D_INT* ix, D_INT* jx, D_INT* descx, D_INT* incx);
