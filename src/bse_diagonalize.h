#pragma once
#include <El.h>
#include <netcdf.h>
#include <netcdf_par.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <nd_array.h>
#include "ElfullControlEig.h"

#define NC_CHUCK_WRITE_SIZE 512 // in KiloBytes
#define EL_BLOCK_SIZE 64

#define EL_type(FUN_NAME, TYPE_SMALL)                   EL_type_HIDDEN(FUN_NAME, TYPE_SMALL)
#define EL_type_HIDDEN(FUN_NAME, TYPE_SMALL)            FUN_NAME ## _ ## TYPE_SMALL

#define EL_Function(FUN_NAME, TYPE_SMALL)          EL_Function_HIDDEN(FUN_NAME, TYPE_SMALL)
#define EL_Function_HIDDEN(FUN_NAME, TYPE_SMALL)   FUN_NAME ## _ ## TYPE_SMALL

#define Function(FUN_NAME, TYPE_SMALL)                  Function_HIDDEN(FUN_NAME, TYPE_SMALL)
#define Function_HIDDEN(FUN_NAME, TYPE_SMALL)           p ## TYPE_SMALL ## FUN_NAME

#if defined(COMPILE_ND_DOUBLE_COMPLEX)
    typedef double BS_float;
    typedef double complex BS_cmplx;
    #define Nd_floatS d
    #define Nd_cmplxS z
    #define NC_WRITE_TYPE NC_DOUBLE
    #define MPI_FLOAT_TYPE              MPI_DOUBLE
    #define MPI_COMPLEX_FLOAT_TYPE      MPI_C_DOUBLE_COMPLEX
#else
    typedef float BS_float;
    typedef float complex BS_cmplx;
    #define Nd_floatS s
    #define Nd_cmplxS c
    #define NC_WRITE_TYPE NC_FLOAT
    #define MPI_FLOAT_TYPE          MPI_FLOAT
    #define MPI_COMPLEX_FLOAT_TYPE  MPI_C_FLOAT_COMPLEX
#endif


#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);}

void Function(Symmetrize,Nd_cmplxS) (EL_type(ElDistMatrix,Nd_cmplxS) block, char ulpo, char block_type);

void Function(AddDeltaE2Diag,Nd_cmplxS) (EL_type(ElDistMatrix,Nd_cmplxS) Kernal, BS_cmplx * delta_energies, \
                        BS_cmplx alpha, BS_cmplx beta, bool make_diag_real);

void Function(load_full_kernal_block,Nd_cmplxS) (EL_type(ElDistMatrix,Nd_cmplxS) Kernal, EL_type(ElDistMatrix,Nd_cmplxS) block,\
    ElInt shift_x, ElInt shift_y, bool transpose, bool conjugate, BS_cmplx alpha, BS_cmplx beta);

// Eig solvers
void Function(heev,Nd_cmplxS) (char uplo, EL_type(ElDistMatrix,Nd_cmplxS) A, EL_type(ElDistMatrix,Nd_floatS) W, \
                            EL_type(ElDistMatrix,Nd_cmplxS) Z, bool eig_vals_only, bool show_progress, \
                            ND_int * n_eig_idx_range, BS_float * eig_val_range);
                            
void Function(geev,Nd_cmplxS) (EL_type(ElDistMatrix,Nd_cmplxS) A, EL_type(ElDistMatrix,Nd_cmplxS) W,
                            EL_type(ElDistMatrix,Nd_cmplxS) Z_left, EL_type(ElDistMatrix,Nd_cmplxS) Z_right);

void Function(Pencilheev,Nd_cmplxS) (ElPencil pencil, char uplo, EL_type(ElDistMatrix,Nd_cmplxS) A, \
                                EL_type(ElDistMatrix,Nd_cmplxS) B, EL_type(ElDistMatrix,Nd_floatS) W, \
                                EL_type(ElDistMatrix,Nd_cmplxS) Z_right, bool eig_vals_only, bool show_progress, \
                                ND_int * n_eig_idx_range, BS_float * eig_val_range);

// IO
/* serial_read,serial_write are take raw data pointers instead of array. THis is the main difference between nd_readP  */
void Function(serial_read,Nd_cmplxS)(char* file_name, char* var_name, void * data_out, MPI_Comm comm);
void Function(serial_read,Nd_floatS)(char* file_name, char* var_name, void * data_out, MPI_Comm comm);
void Function(serial_write,Nd_cmplxS)(char* file_name, char mode, char * var_name, char ** dim_names, size_t rank, size_t dims,  void * data_write, MPI_Comm comm);

void Function(nc_readMat,Nd_cmplxS) (char * filename, char * var_name, bool conjugate, EL_type(ElDistMatrix,Nd_cmplxS) A, ElGrid grid);
void Function(nc_writeMat,Nd_cmplxS) (char * filename, char mode, char *var_name, char ** dim_names, bool conjugate, EL_type(ElDistMatrix,Nd_cmplxS) A, ElGrid grid);

void Function(nd_readP, Nd_cmplxS) (const char* file_name, const char* var_name, ND_array(Nd_cmplxS) * nd_arr_in, MPI_Comm comm);
void Function(nd_readP, Nd_floatS) (const char* file_name, const char* var_name, ND_array(Nd_floatS) * nd_arr_in, MPI_Comm comm);
/** Function related to Yambo */

void Function(YamboDeltaE, Nd_cmplxS) (ND_array(Nd_floatS) * energies_ibz, nd_arr_i * kmap, nd_arr_i * KplusQidxs, nd_arr_i * KminusQidxs, \
                                    ND_array(Nd_floatS) * bse_table ,ND_array(Nd_cmplxS) * delta_energies, bool anti_res_symm);

void Function(BZ_expand, Nd_cmplxS) (ND_array(Nd_floatS) * ibz_kpts, ND_array(Nd_floatS) * sym_mats, \
                                    ND_array(Nd_floatS) * lat_vec,ND_array(Nd_floatS) * kpoints, nd_arr_i * kmap);

void Function(get_KplusQ_idxs , Nd_cmplxS) (ND_array(Nd_floatS) * kpoints, nd_arr_i * KplusQidxs , \
                            BS_float * Q_pt, ND_array(Nd_floatS) * lat_vec, bool Qincrystal);

void Function(BS_table, i) (ND_array(Nd_floatS) * bse_table, nd_arr_i * kmap, ND_int nibz, ND_int vmin, ND_int vmax, \
                            ND_int cmin, ND_int cmax, ND_int nspin, bool anti_res_symm, bool magnons );


/** Helper Functions*/
void Function(Real2Imag, Nd_cmplxS) (EL_type(ElDistMatrix,Nd_cmplxS) Complex_mat, EL_type(ElDistMatrix,Nd_floatS) Real_mat);

void parse_inputs(  int argc, char* argv[], char * bse_report_file, int * nq, ND_int * eig_num_range, \
                    BS_float * eig_val_range, ND_int * eig_num_range_anti, BS_float * eig_val_range_anti, \
                    char * bse_job_name, bool * eig_num_present, bool * eig_val_present, bool * eig_num_present_anti, \
                    bool * eig_val_present_anti);

bool get_GW_from_report_file(const char * report_file, ND_int nspin, int * calc_type, bool * has_inv,\
                            bool * is_metal, int * metal_bands, ND_array(Nd_floatS) * energies_ibz);


void MPI_Bcast_input_variables(int root, MPI_Comm comm, int * nq, ND_int * eig_num_range, BS_float * eig_val_range,\
                             ND_int * eig_num_range_anti, BS_float * eig_val_range_anti, char * bse_job_name,\
                             bool * eig_num_present, bool * eig_val_present, bool * eig_num_present_anti, bool * eig_val_present_anti);

void MPI_Bcast_report_variables(int root, MPI_Comm comm, int * calc_type, bool * has_inv,\
                            bool * is_metal, int * metal_bands, ND_array(Nd_floatS) * energies_ibz);