#include "bse_diagonalize.h"
#include <time.h>
#include <unistd.h>
#include <stdint.h>

/*
Current issues to be solved

** Implemention for metalic case **

Other than metalic case, following needs to be fixed ASAP

1) Interface to QP corrections.
if (rank == 0) printf("Debug 1 \n");
MPI_Abort( MPI_COMM_WORLD, EXIT_SUCCESS );

*/


/***************/
/***************/

enum { NS_PER_SECOND = 1000000000 };
void sub_timespec(struct timespec t1, struct timespec t2, struct timespec *td);

int main( int argc, char* argv[] )
{
    ElError error = ElInitialize( &argc, &argv );

    int rank;

    ElMPIWorldRank(&rank);

    struct timespec start, finish, delta; // timing vars

    if( error != EL_SUCCESS ) MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );

    ElGrid grid;

    error = ElGridCreate( MPI_COMM_WORLD, EL_ROW_MAJOR, &grid );  
    EL_ABORT_ON_ERROR( error );

    error = ElSetBlocksize(EL_BLOCK_SIZE);
    EL_ABORT_ON_ERROR( error );

    int bse_dim, bse_kcvs, NQ;
    int8_t coupling, anti_res_symm, anti_coupling, imag_deltaE, bse_completed, magnons;

    char * DB_FILE          = "SAVE/ns.db1" ;
    char * BS_HEADER        = calloc(3200,sizeof(char));
    char * BS_PAR           = BS_HEADER + 1000;
    char * bse_report_file  = BS_PAR + 1000;
    char * bse_job_name     = bse_report_file + 1000 ;

    bool eig_num_present,eig_val_present,eig_num_present_anti,eig_val_present_anti ;

    ND_int eig_num_range_temp[4] = {0,0,0,0};
    BS_float eig_val_range_temp[4] = {0,0,0,0};

    ND_int *            eig_num_range      = &eig_num_range_temp[0];
    BS_float *     eig_val_range      = &eig_val_range_temp[0];
    ND_int *            eig_num_range_anti = &eig_num_range_temp[2];
    BS_float *     eig_val_range_anti = &eig_val_range_temp[2];

    if (rank == 0) parse_inputs(argc, argv, bse_report_file, &NQ, eig_num_range, eig_val_range, eig_num_range_anti,eig_val_range_anti, \
                bse_job_name, &eig_num_present, &eig_val_present, &eig_num_present_anti, &eig_val_present_anti);
    
    MPI_Bcast_input_variables(0, MPI_COMM_WORLD, &NQ, eig_num_range, eig_val_range, eig_num_range_anti,eig_val_range_anti, \
                            bse_job_name, &eig_num_present, &eig_val_present, &eig_num_present_anti, &eig_val_present_anti);

    if (!eig_num_present)       eig_num_range      = NULL ;
    if (!eig_val_present)       eig_val_range      = NULL ;
    if (!eig_num_present_anti)  eig_num_range_anti = NULL ;
    if (!eig_val_present_anti)  eig_val_range_anti = NULL ;

    sprintf(BS_HEADER, "%s/ndb.BS_head_Q%d", bse_job_name,NQ) ; 
    sprintf(BS_PAR,    "%s/ndb.BS_PAR_Q%d",  bse_job_name,NQ) ; 

    ND_int vmin, vmax, cmax, nspin, nspinor; // valance band max, min, conduction band max, min, number of spin polarizations, spinor components
    
    bool time_rev, inv_sym, is_metal;

    int calc_type = 0;

    int metal_bands[4] = {0,0,0,0};

    BS_float DIMENSIONS[18];

    Function(serial_read,Nd_floatS)(DB_FILE, "DIMENSIONS", DIMENSIONS, MPI_COMM_WORLD);

    
    nspinor = (ND_int)rint(DIMENSIONS[11]);
    nspin   = (ND_int)rint(DIMENSIONS[12]);
    vmax    = (ND_int)rint(DIMENSIONS[14]);
    
    time_rev = (bool)((ND_int)rint(DIMENSIONS[9]));

    //inv_sym  = true ; // *** FIX ME ***

    ND_array(Nd_floatS) energies_ibz;
    ND_function(init,Nd_floatS) (&energies_ibz,    0, NULL);

    Function(nd_readP, Nd_floatS) (DB_FILE, "EIGENVALUES", &energies_ibz, MPI_COMM_WORLD); // read Energies

    if (rank == 0)
    {
        bool gw_present = get_GW_from_report_file(bse_report_file, nspin, &calc_type, &inv_sym, &is_metal, metal_bands, &energies_ibz);

        if (gw_present){ printf("GW Corrections found from report file. Applying Corrections \n"); fflush(stdout);}
    }

    MPI_Bcast_report_variables(0, MPI_COMM_WORLD, &calc_type, &inv_sym, &is_metal, metal_bands, &energies_ibz);

    if (is_metal) 
    {
        printf("Error : Diagonalization for metallic systems is not implemented yet ;( \n");
        MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
    }

    if (nspinor == 1) vmax = vmax/2 ;

    anti_res_symm = 1 ; // resonant - anti-resonant symmetry i.e the anti-resonant block is A = -R* and coupling is symmetric matrix
    anti_coupling = 0 ; // if True, anti-coupling block is read instead of coupling block. For now disabled as yambo only outputs coupling block


    imag_deltaE   = 0 ; // If there is any lifetime corrections to be added. Note it will non Hermitian if this is turned on
    
    if (calc_type == 3) magnons = 1 ; // Magnon case. **** FIX ME **** read from yambo outputs
    else magnons = 0;

    if (NQ != 1 && !time_rev && !inv_sym)  anti_res_symm = 0;
    // if (photoLumen)                     anti_res_symm = 0;
    if (magnons && nspin == 2)             anti_res_symm = 0;

    Function(serial_read,Nd_cmplxS)(BS_HEADER, "Dimension", &bse_dim, MPI_COMM_WORLD);
    Function(serial_read,Nd_cmplxS)(BS_HEADER, "COUPLING", &coupling, MPI_COMM_WORLD);
    Function(serial_read,Nd_cmplxS)(BS_HEADER, "BSE_KERNEL_COMPLETE", &bse_completed, MPI_COMM_WORLD);

    if (!bse_completed)
    {
        printf("BSE Calculation is incomplete! Please redo the calculation completely \n");
        MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
    }

    bse_kcvs = bse_dim;

    if (coupling) bse_dim = bse_dim*2;

    ND_int n_bse_blocks = 1;

    if (!anti_res_symm) n_bse_blocks = 2 ;

    EL_Function(ElDistMatrix,Nd_cmplxS) kernel, W, Z_left, Z_right; // declare varibles 
    EL_Function(ElDistMatrix,Nd_floatS) W_real ;
    
    error = EL_Function(ElDistMatrixCreateSpecific,Nd_cmplxS) ( EL_MR, EL_MC, grid, &kernel );
    EL_ABORT_ON_ERROR( error );

    error = EL_Function(ElDistMatrixCreateSpecific,Nd_cmplxS) ( EL_MR, EL_MC, grid, &Z_right );
    EL_ABORT_ON_ERROR( error );

    error = EL_Function(ElDistMatrixCreateSpecific,Nd_cmplxS) ( EL_MR, EL_MC, grid, &W );
    EL_ABORT_ON_ERROR( error );

    error = EL_Function(ElDistMatrixResize,Nd_cmplxS) ( kernel, bse_dim, bse_dim );  
    EL_ABORT_ON_ERROR( error );

    error = EL_Function(ElZeroDist,Nd_cmplxS)  (kernel);
    EL_ABORT_ON_ERROR( error );

    error = EL_Function(ElDistMatrixCreateSpecific,Nd_cmplxS) ( EL_MR, EL_MC, grid, &Z_left );
    EL_ABORT_ON_ERROR( error );

    /**************/
    

    //ElInt i, j;

    MPI_Barrier(MPI_COMM_WORLD);

    clock_gettime(CLOCK_REALTIME, &start);

    /***** Read QP stuff */

    ND_array(Nd_cmplxS)       delta_energies;
    ND_array(Nd_floatS) ibz_kpts, sym_mats, lat_vec, kpoints, bse_table;
    nd_arr_i kmap, KplusQidxs, KminusQidxs;

    BS_float kpts_param[4];
    
    int bse_bands[2];

    Function(serial_read,Nd_floatS)(BS_HEADER, "HEAD_R_LATT", kpts_param, MPI_COMM_WORLD);
    Function(serial_read,Nd_floatS)(BS_HEADER, "Bands", bse_bands, MPI_COMM_WORLD);

    vmin = bse_bands[0];
    cmax = bse_bands[1];

    ND_int Nkbz = (ND_int)kpts_param[3] ;
    
    ND_function(init,Nd_floatS) (&ibz_kpts,  0, NULL);
    ND_function(init,Nd_floatS) (&sym_mats,  0, NULL);
    ND_function(init,Nd_floatS) (&lat_vec,   0, NULL);
    
    


    nd_init_i                           (&kmap,              2, nd_idx{Nkbz,2});
    nd_init_i                           (&KplusQidxs,        1, nd_idx{Nkbz}); 
    nd_init_i                           (&KminusQidxs,       1, nd_idx{Nkbz});
    ND_function(init,Nd_floatS) (&kpoints,           2, nd_idx{Nkbz,3});

    ND_int bs_table_dim = n_bse_blocks*bse_kcvs ;
    
    ND_function(init,Nd_cmplxS)       (&delta_energies,    1, nd_idx{bs_table_dim});
    ND_function(init,Nd_floatS) (&bse_table,         2, nd_idx{5,bs_table_dim});

    nd_malloc_i                           (&kmap);
    nd_malloc_i                           (&KplusQidxs);
    nd_malloc_i                           (&KminusQidxs);
    ND_function(malloc,Nd_floatS) (&kpoints);
    ND_function(malloc,Nd_cmplxS)       (&delta_energies);
    ND_function(malloc,Nd_floatS) (&bse_table);

    Function(nd_readP, Nd_floatS) (DB_FILE, "K-POINTS",           &ibz_kpts,     MPI_COMM_WORLD); // read kpoints
    Function(nd_readP, Nd_floatS) (DB_FILE, "SYMMETRY",           &sym_mats,     MPI_COMM_WORLD); // read kpoints
    Function(nd_readP, Nd_floatS) (DB_FILE, "LATTICE_VECTORS",    &lat_vec,      MPI_COMM_WORLD); // read lattice vectors

    BS_float lat_parameters[3], Q_pt[3]; // lattice scaling parameters and Q point 

    if (NQ == 1)
    {
        Q_pt[0] = 0; Q_pt[1] = 0; Q_pt[2] = 0;
    }
    else
    {
        Function(serial_read,Nd_cmplxS)(BS_HEADER, "Q-point", Q_pt, MPI_COMM_WORLD); // Read the BSE Q point
    }
    

    Function(serial_read,Nd_floatS)(DB_FILE, "LATTICE_PARAMETER", lat_parameters, MPI_COMM_WORLD); // lattice constants

    //FUNCTION(nd_readP, Nd_floatS) (DIAGO_FILE, "BS_TABLE",    &bse_table,     MPI_COMM_WORLD); // read bse_table

    /* Convert kpoints in yambo units to cartisian units*/
    ND_int Nikbz = ibz_kpts.dims[1];
    for (ND_int i = 0 ; i<Nikbz ; ++i )
    {
        ibz_kpts.data[i + 0*Nikbz] = ibz_kpts.data[i + 0*Nikbz]/lat_parameters[0];
        ibz_kpts.data[i + 1*Nikbz] = ibz_kpts.data[i + 1*Nikbz]/lat_parameters[1];
        ibz_kpts.data[i + 2*Nikbz] = ibz_kpts.data[i + 2*Nikbz]/lat_parameters[2];
    }

    /** negative sign is added because later we need k-Q, but the Functions are written for K+Q*/
    Q_pt[0] = Q_pt[0]/lat_parameters[0] ;
    Q_pt[1] = Q_pt[1]/lat_parameters[1] ;
    Q_pt[2] = Q_pt[2]/lat_parameters[2] ;
    
    /* Compute Delta E*/
    Function(BZ_expand, Nd_cmplxS) (&ibz_kpts, &sym_mats, &lat_vec, &kpoints, &kmap);

    /* Compute the BS_table*/
    /************************ FIX ME *******************************/
    /** FIX ME, the these bands needs to be fixed in mettalic case*/
    Function(BS_table, i) (&bse_table, &kmap, Nikbz, vmin, vmax, vmax+1, cmax, nspin, anti_res_symm, magnons);
    
    Function(get_KplusQ_idxs , Nd_cmplxS) (&kpoints, &KplusQidxs, Q_pt, &lat_vec, false); /* get K+Q indices*/

    /** negative sign is added because we now need k-Q*/
    Q_pt[0] = -Q_pt[0];
    Q_pt[1] = -Q_pt[1];
    Q_pt[2] = -Q_pt[2];

    /*Get K+Q indices. since we set Q=-Q, we are actually getting K-Q indices*/
    /* Now Q is in crystal coordinates as previous get_KplusQ_idxs call changed it from cart to crystal*/
    Function(get_KplusQ_idxs , Nd_cmplxS) (&kpoints, &KminusQidxs, Q_pt, &lat_vec, true); /* get K+Q indices*/

    /* Campute the transistion energies*/
    Function(YamboDeltaE, Nd_cmplxS) (&energies_ibz, &kmap, &KplusQidxs, &KminusQidxs, &bse_table ,&delta_energies, anti_res_symm);

    if (rank ==0) ND_function(write,Nd_floatS)("bs_table.nc", "bs_table", &bse_table, (char * [4]) {"k_v_c_ispc_ispv", "bse_dim"},NULL);

    if (rank ==0) ND_function(write,Nd_floatS)("kpoints.nc", "kpoints", &kpoints, (char * [4]) {"nk", "coordinates"},NULL);

    if (rank ==0) nd_write_i("kmap.nc", "map", &kmap, (char * [4]) {"Nkbz", "ibz_sym"},NULL);

    /** Free stuff as they are no longer needed*/
    ND_function(free,Nd_floatS) (&ibz_kpts);
    ND_function(free,Nd_floatS) (&sym_mats);
    ND_function(free,Nd_floatS) (&lat_vec);
    ND_function(free,Nd_floatS) (&bse_table);
    ND_function(free,Nd_floatS) (&energies_ibz);
    nd_free_i                           (&kmap);
    nd_free_i                           (&KplusQidxs); 
    nd_free_i                           (&KminusQidxs);
    ND_function(free,Nd_floatS) (&kpoints);
    

    ND_function(uninit,Nd_floatS) (&ibz_kpts);
    ND_function(uninit,Nd_floatS) (&sym_mats);
    ND_function(uninit,Nd_floatS) (&lat_vec);
    ND_function(uninit,Nd_floatS) (&bse_table);
    ND_function(uninit,Nd_floatS) (&energies_ibz);
    nd_uninit_i                           (&kmap);
    nd_uninit_i                           (&KplusQidxs);
    nd_uninit_i                           (&KminusQidxs);
    ND_function(uninit,Nd_floatS) (&kpoints);
    
    
    /**********/
    /**********************************/
    if (!coupling) 
    {   
        //printf("Resonant \n");

        Function(nc_readMat,Nd_cmplxS) (BS_PAR,"BSE_RESONANT",true,kernel,grid);
    
        Function(AddDeltaE2Diag,Nd_cmplxS) (kernel, delta_energies.data, 1.0f, 1.0, true);

        error = EL_Function(ElDistMatrixCreateSpecific,Nd_floatS) ( EL_MR, EL_MC, grid, &W_real );
        EL_ABORT_ON_ERROR( error );
    }

    else
    {   /** Read all blocks */
        /*
            | R   C |
        K = |       |
            |-C* -R*|
        */
        /** Create a buffer block to load subblocks **/
        //printf("Coupling \n");
        EL_Function(ElDistMatrix,Nd_cmplxS) buffer_block;

        error = EL_Function(ElDistMatrixCreateSpecific,Nd_cmplxS) ( EL_MR, EL_MC, grid, &buffer_block );
        EL_ABORT_ON_ERROR( error );

        error = EL_Function(ElDistMatrixResize,Nd_cmplxS) ( buffer_block, bse_dim/2, bse_dim/2 );  
        EL_ABORT_ON_ERROR( error );

        error = EL_Function(ElZeroDist,Nd_cmplxS)  (buffer_block);
        EL_ABORT_ON_ERROR( error );

        /** Setting resonant blocks*/

        // 1. Set resonant block
        Function(nc_readMat,Nd_cmplxS) (BS_PAR,"BSE_RESONANT",true,buffer_block,grid);

        Function(Symmetrize,Nd_cmplxS) (buffer_block, 'L', 'H');

        Function(AddDeltaE2Diag,Nd_cmplxS) (buffer_block, delta_energies.data, 1.0f, 1.0, true);
        
        Function(load_full_kernal_block,Nd_cmplxS) (kernel, buffer_block, 0, 0, false, false, 1.0, 0.0);
        
        // 2. Set anti-resonant block
        if (!anti_res_symm)
        {   
            /* Anti resonant block is still hermitian*/

            Function(nc_readMat,Nd_cmplxS) (BS_PAR,"BSE_ANTI-RESONANT",true,buffer_block,grid);

            Function(Symmetrize,Nd_cmplxS) (buffer_block, 'L', 'H');
            
            Function(AddDeltaE2Diag,Nd_cmplxS) (buffer_block, delta_energies.data + bse_kcvs, 1.0f, 1.0, true);

            Function(load_full_kernal_block,Nd_cmplxS) (kernel, buffer_block, bse_dim/2, bse_dim/2, false, false, 1.0, 0.0);
        }

        else
        {   
            Function(load_full_kernal_block,Nd_cmplxS) (kernel, buffer_block, bse_dim/2, bse_dim/2, false, true, -1.0, 0.0);
        }

        /** Setting coupling blocks*/

        if (!anti_coupling)
        {   
            // 3. Read the coupling block
            Function(nc_readMat,Nd_cmplxS) (BS_PAR,"BSE_COUPLING",false,buffer_block,grid);

            /* Coupling mat is symmetric when there is res-antires symmetry*/
            if (anti_res_symm) Function(Symmetrize,Nd_cmplxS) (buffer_block, 'L', 'S');
        
            Function(load_full_kernal_block,Nd_cmplxS) (kernel, buffer_block, 0, bse_dim/2, false, false, 1.0, 0.0);
        
            // 4. Set anti-coupling block. Q = -C^H
            Function(load_full_kernal_block,Nd_cmplxS) (kernel, buffer_block, bse_dim/2, 0, true, true, -1.0, 0.0);

        }
        else
        {   
            // 3. Read the ant-coupling block
            Function(nc_readMat,Nd_cmplxS) (BS_PAR,"BSE_ANTI-COUPLING",false,buffer_block,grid);

            /* Anti-Coupling mat is symmetric when there is res-antires symmetry*/
            if (anti_res_symm) Function(Symmetrize,Nd_cmplxS) (buffer_block, 'L', 'S');
        
            Function(load_full_kernal_block,Nd_cmplxS) (kernel, buffer_block, bse_dim/2, 0, false, false, 1.0, 0.0); 
        
            // 4. Set coupling block. C = -Q^H
            Function(load_full_kernal_block,Nd_cmplxS) (kernel, buffer_block, 0, bse_dim/2, true, true, -1.0, 0.0);
            
        }

        EL_Function(ElDistMatrixEmpty,Nd_cmplxS) (buffer_block,true);
    }

    /**********************************/
    MPI_Barrier(MPI_COMM_WORLD);
    
    clock_gettime(CLOCK_REALTIME, &finish); //timing end:
    
    sub_timespec(start, finish, &delta); // timing end:
    
    if (rank == 0) { printf("Time taken for Read : %d.%.9ld s\n", (int)delta.tv_sec, delta.tv_nsec); fflush(stdout);}

    
    MPI_Barrier(MPI_COMM_WORLD);
    
    clock_gettime(CLOCK_REALTIME, &start);
    
    /**********************************/

    //FUNCTION(nc_writeMat,Nd_cmplxS) ("kernal.nc",'w',"eig_vecs", (char *[]){"evec1","evec2","evec_re_im"},'N',kernel, grid);

    //scale matrix for better eigenvalues

    if (coupling) Function(geev,Nd_cmplxS) (kernel, W, Z_left, Z_right);

    else 
    {   
        if (!imag_deltaE)
        {   
            Function(heev,Nd_cmplxS) ('L',kernel,W_real,Z_right, false, true, eig_num_range,eig_val_range); // Hermitian diago 
            
            /* Copy to W*/

            /* Get dims of W_real*/
            ElInt eig_bse_dimH, eig_bse_dimHW;

            error = EL_Function(ElDistMatrixHeight,Nd_floatS) (W_real,&eig_bse_dimH);
            EL_ABORT_ON_ERROR( error );

            error = EL_Function(ElDistMatrixWidth,Nd_floatS)  (W_real,&eig_bse_dimHW);
            EL_ABORT_ON_ERROR( error );

            error = EL_Function(ElDistMatrixResize,Nd_cmplxS) ( W, eig_bse_dimH, eig_bse_dimHW );  //A.Height(), A.Width()
            EL_ABORT_ON_ERROR( error );

            error = EL_Function(ElZeroDist,Nd_cmplxS)  (W);
            EL_ABORT_ON_ERROR( error );

            Function(Real2Imag, Nd_cmplxS) (W, W_real);
        }
        else
        {   
            Function(Symmetrize,Nd_cmplxS) (kernel, 'L', 'H');

            Function(geev,Nd_cmplxS) (kernel, W, Z_left, Z_right); // Non-hermitian diago
        }
    }
    /**********************************/
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    clock_gettime(CLOCK_REALTIME, &finish); //timing end:
    
    sub_timespec(start, finish, &delta); // timing end:
    
    if (rank == 0) { printf("Time taken for Diagonalize : %d.%.9ld s\n", (int)delta.tv_sec, delta.tv_nsec); fflush(stdout);}


    MPI_Barrier(MPI_COMM_WORLD);
    
    clock_gettime(CLOCK_REALTIME, &start);
    
    /**********************************/

    // Now store the transpose of eigen vectors in kernal
    
    EL_Function(ElTransposeDist,Nd_cmplxS) (Z_right,kernel);

    Function(nc_writeMat,Nd_cmplxS) ("out.nc",'w',"BS_EIGENSTATES", (char *[]){"n_evec","kcv","evec_re_im"},false,kernel, grid);
    Function(nc_writeMat,Nd_cmplxS) ("out.nc",'a',"BS_Energies", (char *[]){"neig","None","eval_re_im"},false,W, grid);


    /* Anti-resonant and  no coupling*/
    if (!coupling && !anti_res_symm) 
    {   
        Function(nc_readMat,Nd_cmplxS) (BS_PAR,"BSE_ANTI-RESONANT",true,kernel,grid);
    
        Function(AddDeltaE2Diag,Nd_cmplxS) (kernel, delta_energies.data + bse_kcvs, 1.0f, 1.0, true);

        if (!imag_deltaE)
        {   
            Function(heev,Nd_cmplxS) ('L',kernel,W_real,Z_right, false, false, eig_num_range_anti,eig_val_range_anti); // Hermitian diago

            Function(Real2Imag, Nd_cmplxS) (W, W_real);
        }
        else
        {   
            Function(Symmetrize,Nd_cmplxS) (kernel, 'L', 'H');

            Function(geev,Nd_cmplxS) (kernel, W, Z_left, Z_right); // Non-hermitian diago
        }

        // Now store the transpose of eigen vectors in kernal
        EL_Function(ElTransposeDist,Nd_cmplxS) (Z_right,kernel);

        /*Write anti-res part to file*/
        Function(nc_writeMat,Nd_cmplxS) ("out.nc",'a',"BS_EIGENSTATES_ANTI", (char *[]){"n_evec_anti_res","kcv_anti_res","evec_re_im_anti_res"},false,kernel, grid);
        Function(nc_writeMat,Nd_cmplxS) ("out.nc",'a',"BS_Energies_ANTI", (char *[]){"neig_anti_res","1_anti_res","eval_re_im_anti_res"},false,W, grid);

    }

    // ElPrintDist_c( W, "Eigen_values" );
    // /**********************************/
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    clock_gettime(CLOCK_REALTIME, &finish); //timing end:
    
    sub_timespec(start, finish, &delta); // timing end:
    
    if (rank == 0) { printf("Time taken for Write : %d.%.9ld s\n", (int)delta.tv_sec, delta.tv_nsec); fflush(stdout);}

    EL_Function(ElDistMatrixEmpty,Nd_cmplxS) (kernel,true);
    
    EL_Function(ElDistMatrixEmpty,Nd_cmplxS) (W,true);

    if (!coupling) EL_Function(ElDistMatrixEmpty,Nd_floatS) (W_real,true);
    
    EL_Function(ElDistMatrixEmpty,Nd_cmplxS) (Z_left,true);
    
    EL_Function(ElDistMatrixEmpty,Nd_cmplxS) (Z_right,true);

    ND_function(free,Nd_cmplxS)       (&delta_energies); // free delta E
    ND_function(uninit,Nd_cmplxS)       (&delta_energies);

    free(BS_HEADER);

    if (rank == 0) { printf(" Done :) \n"); fflush(stdout);}

    error = ElFinalize();

    if( error != EL_SUCCESS ) MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );

    return 0;

}


void sub_timespec(struct timespec t1, struct timespec t2, struct timespec *td)
{
    td->tv_nsec = t2.tv_nsec - t1.tv_nsec;
    td->tv_sec  = t2.tv_sec - t1.tv_sec;
    if (td->tv_sec > 0 && td->tv_nsec < 0)
    {
        td->tv_nsec += NS_PER_SECOND;
        td->tv_sec--;
    }
    else if (td->tv_sec < 0 && td->tv_nsec > 0)
    {
        td->tv_nsec -= NS_PER_SECOND;
        td->tv_sec++;
    }
}



    