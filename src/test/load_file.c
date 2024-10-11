#include <complex.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tests.h"

static D_LL_INT get_doubles(char* str, D_float* out);
static Err_INT read_complex_from_file(const char* filename, D_Cmplx* out_arr,
                                      D_INT nmax);

// int main(void)
// {
//     D_Cmplx * out = malloc(sizeof(*out) * 100);

//     read_complex_from_file("Herm_100_eigs.mat", out,100);

//     for (int i =0; i <100; ++i)
//     {
//         printf("%.5f %.5f\n",creal(out[i]), cimag(out[i]));
//     }

//     free(out);
//     return 0;
// }

Err_INT load_mat_file(const char* mat_file, const char* eig_file, void* DmatA,
                      D_Cmplx* eig_vals, bool bse_mat)
{
    struct D_Matrix* matA = DmatA;
    D_INT dim = matA->gdims[0];

    D_INT error = 0;

    D_Cmplx* tmp_mat = NULL;
    D_LL_INT nset = 0;
    D_INT nmax = dim;
    if (bse_mat) nmax /= 2;

    int myid;
    MPI_Comm_rank(matA->comm, &myid);

    if (!myid)
    {
        nset = dim * dim;

        tmp_mat = malloc(sizeof(*tmp_mat) * nset);
        check_ptr(tmp_mat);

        error = read_complex_from_file(mat_file, tmp_mat, nset);
        check_error(error);

        error = read_complex_from_file(eig_file, eig_vals, nmax);
        check_error(error);
    }

    error = initiateSetQueue(DmatA, nset);
    check_error(error);

    if (!myid)
    {
        for (D_LL_INT iset = 0; iset < nset; ++iset)
        {
            D_INT i = iset / dim;
            D_INT j = iset % dim;
            // printf("%d %d\n",i,j);
            error = DMatSet(DmatA, i, j, tmp_mat[iset]);
            check_error(error);
        }
    }

    error = ProcessSetQueue(DmatA);
    check_error(error);

    error = MPI_Bcast(eig_vals, nmax, D_Cmplx_MPI_TYPE, 0, matA->comm);
    check_error(error);

    free(tmp_mat);
    return 0;
}

Err_INT read_complex_from_file(const char* filename, D_Cmplx* out_arr,
                               D_INT nmax)
{
    FILE* file = fopen(filename, "r"); /* should check the result */
    if (!file)
    {
        return -50;
    }
    char line[128];

    D_LL_INT i = 0;
    while (fgets(line, sizeof(line), file) && i < nmax)
    {
        char* hash_char = strchr(line, '#');
        if (hash_char)
        {
            *hash_char = '\0';
        }

        D_float out_tmp[16];  // to avoid any overflow if
        // input has more than 2 double present
        D_LL_INT nparsed = get_doubles(line, out_tmp);
        if (!nparsed)
        {
            continue;
        }
        if (nparsed != 2)
        {
            return -40;  // error
        }
        out_arr[i] = out_tmp[0] + I * out_tmp[1];
        ++i;
    }
    fclose(file);
    return 0;
}

static D_LL_INT get_doubles(char* str, D_float* out)
{
    /*
    Taken from :
    https://github.com/muralidhar-nalabothula/LetzElPhC/blob/main/src/common/string_func.c
    Extract all float values from given string

    if out == NULL, it return number of float it parsed
    */
    char* p = str;
    char* q;
    double temp_val;
    D_LL_INT count = 0;

    while (*p)
    {
        if (isdigit(*p) || ((*p == '-' || *p == '+') && isdigit(*(p + 1))))
        {
            temp_val = strtod(p, &q);

            if (p == q)
            {
                break;
            }
            else
            {
                p = q;
            }

            if (out != NULL)
            {
                out[count] = temp_val;
            }
            ++count;
        }
        else
        {
            ++p;
        }
    }
    return count;
}
