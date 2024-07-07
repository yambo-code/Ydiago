// sort real eigen vectors based on eigenvalues

Err_INT sort_evecs(D_float* evals, D_float* evecs_real,
                   D_float* evecs_imag, D_INT neigs)
{

    pslasrt(const char* id, const MKL_INT* n, float* d,
            const float* q, const MKL_INT* iq, const MKL_INT* jq,
            const MKL_INT* descq, float* work, const MKL_INT* lwork,
            MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info);
}