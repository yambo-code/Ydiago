#include "bse_diagonalize.h"


static void get_sub_block_indices(ElInt GlobalMatrows, ElInt GlobalMatcols, ElInt Grows, ElInt Gcols, \
                ElInt myrow, ElInt mycol, ElRange_i *rowRange, ElRange_i *colRange, size_t * block_dims);

static void Function(set_block,Nd_cmplxS) (EL_type(ElDistMatrix,Nd_cmplxS) A, BS_cmplx * A_sub, \
                                        ElRange_i rowRange, ElRange_i colRange, bool conjugate);

static void Function(get_block,Nd_cmplxS) (EL_type(ElDistMatrix,Nd_cmplxS) A, BS_cmplx * A_sub, \
                                        ElRange_i rowRange, ElRange_i colRange, bool conjugate);


void Function(nc_readMat,Nd_cmplxS) (char * filename, char * var_name, bool conjugate, EL_type(ElDistMatrix,Nd_cmplxS) A, ElGrid grid)
{
    /* Reads data from netCDF files to distMatrix 
        grid is grid of A
    */

    ElError error;

    int ncid, varid, dimid[3], retval;

    MPI_Comm comm ;
    
    error = EL_Function(ElDistMatrixDistComm,Nd_cmplxS) (A, &comm);
    EL_ABORT_ON_ERROR( error );

    ElInt grows, gcols; // global rows and cols of A

    /* Get the global block dims */
    error = EL_Function(ElDistMatrixHeight,Nd_cmplxS) (A,&grows);
    EL_ABORT_ON_ERROR( error );

    error = EL_Function(ElDistMatrixWidth,Nd_cmplxS)  (A,&gcols);
    EL_ABORT_ON_ERROR( error );

    int Gx, Gy; // Grid dimensions

    error = ElGridHeight(grid,&Gx);
    EL_ABORT_ON_ERROR( error );

    error = ElGridWidth(grid,&Gy);
    EL_ABORT_ON_ERROR( error );

    int col_rank, row_rank;

    error = EL_Function(ElDistMatrixRowRank,Nd_cmplxS)  (A,&row_rank);
    EL_ABORT_ON_ERROR( error );

    error = EL_Function(ElDistMatrixColRank,Nd_cmplxS)  (A,&col_rank);
    EL_ABORT_ON_ERROR( error );

    ElRange_i rowRange, colRange;

    size_t block_dims[3] = {0,0,2};

    get_sub_block_indices(grows, gcols, Gx, Gy, row_rank, col_rank, &rowRange, &colRange, block_dims);

    if ((retval = nc_open_par(filename, NC_NOWRITE, comm, MPI_INFO_NULL, &ncid))) ERR(retval);
    
    if ((retval = nc_inq_varid(ncid, var_name, &varid))) ERR(retval); // get the id of the req variable

    if ((retval = nc_var_par_access(ncid, varid, NC_COLLECTIVE))) ERR(retval); // NC_COLLECTIVE or NC_INDEPENDENT

    BS_cmplx * buffer = malloc(block_dims[0]*block_dims[1]*sizeof(BS_cmplx));

    if ((retval = nc_get_vara(ncid, varid,  (size_t[]){rowRange.beg,colRange.beg,0}, block_dims, buffer))) ERR(retval); 

    error = EL_Function(ElZeroDist,Nd_cmplxS)  (A);
    EL_ABORT_ON_ERROR( error );

    Function(set_block,Nd_cmplxS) (A, buffer,rowRange, colRange, conjugate);

    free(buffer);

    MPI_Barrier(comm);

    if ((retval =nc_close(ncid))) ERR(retval);
}


void Function(nc_writeMat,Nd_cmplxS) (char * filename, char mode, char *var_name, char ** dim_names, bool conjugate, EL_type(ElDistMatrix,Nd_cmplxS) A, ElGrid grid)
{
    /* Writes data from netCDF files to distMatrix 
    mode = 'a' append
    mode = 'w' or anyother. overwrite and create a new file
    grid is grid of A
    */
    // set strips for ex, incase of lustre : "lfs setstripe -c -1 path/to/dir"
    
    ElError error;

    int ncid, varid, dimid[3], retval, old_fill_mode;

    MPI_Comm comm ;

    error = EL_Function(ElDistMatrixDistComm,Nd_cmplxS) (A, &comm);
    EL_ABORT_ON_ERROR( error );

    ElInt grows, gcols; // global rows and cols of A

    /* Get the global block dims */
    error = EL_Function(ElDistMatrixHeight,Nd_cmplxS) (A,&grows);
    EL_ABORT_ON_ERROR( error );
    error = EL_Function(ElDistMatrixWidth,Nd_cmplxS)  (A,&gcols);
    EL_ABORT_ON_ERROR( error );

    if (mode == 'a')
    {
        if ((retval = nc_open_par(filename, NC_WRITE, comm, MPI_INFO_NULL, &ncid))) ERR(retval);
    }
    else
    {
        if ((retval= nc_create_par(filename, NC_CLOBBER | NC_NETCDF4, comm, MPI_INFO_NULL, &ncid))) ERR(retval);
    }

    retval = nc_set_fill(ncid, NC_NOFILL, &old_fill_mode); // Avoid filling automatically. This doubles the write cost
    
    if ((retval= nc_def_dim(ncid, dim_names[0], grows, dimid))) ERR(retval);

    if ((retval= nc_def_dim(ncid, dim_names[1], gcols, dimid+1))) ERR(retval);
    
    
    #if defined(COMPILE_ND_DOUBLE_COMPLEX) || defined(COMPILE_ND_SINGLE_COMPLEX)
    if ((retval= nc_def_dim(ncid, dim_names[2], 2, dimid+2))) ERR(retval);
    if ((retval =nc_def_var(ncid, var_name, NC_WRITE_TYPE, 3, dimid, &varid))) ERR(retval);
    #else
    if ((retval =nc_def_var(ncid, var_name, NC_WRITE_TYPE, 2, dimid, &varid))) ERR(retval);
    #endif

    
    size_t chuck_size = sqrt((NC_CHUCK_WRITE_SIZE*1024)/sizeof(BS_cmplx));

    size_t chuck_size_x = (chuck_size < grows) ? chuck_size : grows ;
    size_t chuck_size_y = (chuck_size < gcols) ? chuck_size : gcols ;

    if ((retval = nc_def_var_chunking(ncid, varid, NC_CHUNKED, (size_t[]){chuck_size_x, chuck_size_y, 2} ))) ERR(retval);

    if ((retval = nc_var_par_access(ncid, varid, NC_COLLECTIVE))) ERR(retval); // NC_COLLECTIVE or NC_INDEPENDENT

    if ((retval =nc_enddef(ncid))) ERR(retval);

    /* Fix me: Check the dims*/

    int Gx, Gy; // Grid dimensions

    error = ElGridHeight(grid,&Gx);
    EL_ABORT_ON_ERROR( error );

    error = ElGridWidth(grid,&Gy);
    EL_ABORT_ON_ERROR( error );


    ElInt lrows, lcols; // local block dims of the distributed matrix A

    /* Get the local block dims */
    error = EL_Function(ElDistMatrixLocalHeight,Nd_cmplxS) (A,&lrows);
    EL_ABORT_ON_ERROR( error );

    error = EL_Function(ElDistMatrixLocalWidth,Nd_cmplxS)  (A,&lcols);
    EL_ABORT_ON_ERROR( error );

    int col_rank, row_rank;

    error = EL_Function(ElDistMatrixRowRank,Nd_cmplxS)  (A,&row_rank);
    EL_ABORT_ON_ERROR( error );

    error = EL_Function(ElDistMatrixColRank,Nd_cmplxS)  (A,&col_rank);
    EL_ABORT_ON_ERROR( error );

    ElRange_i rowRange, colRange;

    size_t block_dims[3] = {0,0,2};

    get_sub_block_indices(grows, gcols, Gx, Gy, row_rank, col_rank, &rowRange, &colRange, block_dims);
    
    BS_cmplx * buffer = malloc(block_dims[0]*block_dims[1]*sizeof(BS_cmplx));

    Function(get_block,Nd_cmplxS) (A, buffer,rowRange, colRange, conjugate);

    if ((retval = nc_put_vara(ncid, varid,  (size_t[]){rowRange.beg,colRange.beg,0}, block_dims, buffer))) ERR(retval);

    free(buffer);
    
    MPI_Barrier(comm);

    if ((retval =nc_close(ncid))) ERR(retval);

}


/* Function to read from netCDF */
/*Complex version*/
void Function(nd_readP, Nd_cmplxS) (const char* file_name, const char* var_name, ND_array(Nd_cmplxS) * nd_arr_in, MPI_Comm comm)
{   
    /* Note this Function create the ND_array(Nd_cmplxS), so Function(free, Nd_cmplxS) () must be called to free the memory
        DO NOT pass a pointer which already has data. this leads to memory leak
        Ex: Function(read, Nd_cmplxS) ("ndb.BS_elph", "exc_elph", &temp_array);
    */
    //
    if (nd_arr_in->data != NULL) error_msg("Input array is uninitialized or has data."); // check if there is data or uninitialized

    if (nd_arr_in->rank != NULL) free(nd_arr_in->rank); // if not null, free the memeory

    //
    int ncid, var_id, retval, nc_rank; // variables 

    if ((retval = nc_open_par(file_name, NC_NOWRITE, comm, MPI_INFO_NULL, &ncid))) ERR(retval); // open the file

    if ((retval = nc_inq_varid(ncid, var_name, &var_id))) ERR(retval); // get the id of the req variable

    if ((retval = nc_inq_var(ncid, var_id, NULL, NULL, &nc_rank, NULL, NULL ))) ERR(retval); // get rank

    int * dim_ids             = malloc(nc_rank*sizeof(int));
    ND_int * nd_dims      = malloc(nc_rank*sizeof(ND_int));
    size_t * nd_dims_temp     = malloc(nc_rank*sizeof(size_t));

    if ((retval = nc_inq_var(ncid, var_id, NULL, NULL, NULL, dim_ids, NULL ))) ERR(retval); // get dims
    //
    for (ND_int i = 0; i < nc_rank; ++i)
    {
        if ((retval = nc_inq_dimlen(ncid, dim_ids[i], nd_dims_temp + i))) ERR(retval);
        nd_dims[i] = nd_dims_temp[i] ;
    }

    #if defined(COMPILE_ND_DOUBLE_COMPLEX) || defined(COMPILE_ND_SINGLE_COMPLEX)

        if ((nc_rank < 1) || nd_dims[(ND_int) (nc_rank-1)] != 2 ) error_msg("Cannot convert a real to complex array.") ; 

        ND_function(init, Nd_cmplxS) (nd_arr_in, ((ND_int) (nc_rank-1)), nd_dims); 

    #else

        ND_function(init, Nd_cmplxS) (nd_arr_in, ((ND_int) nc_rank), nd_dims); 

    #endif

    ND_function(malloc, Nd_cmplxS) (nd_arr_in);  // this must be free outside else memory leak

    if ((retval = nc_get_var(ncid, var_id, nd_arr_in->data))) ERR(retval); //get data in floats

    if ((retval = nc_close(ncid))) ERR(retval); // close the file

    // free all temp arrays
    free(dim_ids);
    free(nd_dims);
    free(nd_dims_temp);
}
/*Float version*/
void Function(nd_readP, Nd_floatS) (const char* file_name, const char* var_name, ND_array(Nd_floatS) * nd_arr_in, MPI_Comm comm)
{   
    /* Note this Function create the ND_array(Nd_floatS), so Function(free, Nd_floatS) () must be called to free the memory
        DO NOT pass a pointer which already has data. this leads to memory leak
        Ex: Function(read, Nd_floatS) ("ndb.BS_elph", "exc_elph", &temp_array);
    */
    //
    if (nd_arr_in->data != NULL) error_msg("Input array is uninitialized or has data."); // check if there is data or uninitialized

    if (nd_arr_in->rank != NULL) free(nd_arr_in->rank); // if not null, free the memeory

    //
    int ncid, var_id, retval, nc_rank; // variables 

    if ((retval = nc_open_par(file_name, NC_NOWRITE, comm, MPI_INFO_NULL, &ncid))) ERR(retval); // open the file

    if ((retval = nc_inq_varid(ncid, var_name, &var_id))) ERR(retval); // get the id of the req variable

    if ((retval = nc_inq_var(ncid, var_id, NULL, NULL, &nc_rank, NULL, NULL ))) ERR(retval); // get rank

    int * dim_ids             = malloc((nc_rank)*sizeof(int));
    ND_int * nd_dims      = malloc((nc_rank)*sizeof(ND_int));
    size_t * nd_dims_temp     = malloc((nc_rank)*sizeof(size_t));

    if ((retval = nc_inq_var(ncid, var_id, NULL, NULL, NULL, dim_ids, NULL ))) ERR(retval); // get dims
    //
    for (ND_int i = 0; i < nc_rank; ++i)
    {
        if ((retval = nc_inq_dimlen(ncid, dim_ids[i], nd_dims_temp + i))) ERR(retval);
        nd_dims[i] = nd_dims_temp[i] ;
    }

    ND_function(init, Nd_floatS) (nd_arr_in, nc_rank, nd_dims); 

    ND_function(malloc, Nd_floatS) (nd_arr_in);  // this must be free outside else memory leak

    if ((retval = nc_get_var(ncid, var_id, nd_arr_in->data))) ERR(retval); //get data in floats

    if ((retval = nc_close(ncid))) ERR(retval); // close the file

    // free all temp arrays
    free(dim_ids);
    free(nd_dims);
    free(nd_dims_temp);
}

// Read data from netcdf to all processes
/* Complex version */
void Function(serial_read,Nd_cmplxS)(char* file_name, char* var_name, void * data_out, MPI_Comm comm)
{   // input (Filename, Variable name, data pointer)
    /*  Serial read
        Note: the data_out will be in all processes
    */
    int ncid, varid, retval;

    if ((retval = nc_open_par(file_name, NC_NOWRITE, comm, MPI_INFO_NULL, &ncid))) ERR(retval); // open the file
    
    if ((retval = nc_inq_varid(ncid, var_name, &varid))) ERR(retval); // get the varible id of the file
    
    if ((retval = nc_get_var(ncid, varid, data_out))) ERR(retval); //get data in floats
    
    if ((retval = nc_close(ncid))) ERR(retval); // close the file

}

// Read data from netcdf to all processes
/* Real version */
void Function(serial_read,Nd_floatS)(char* file_name, char* var_name, void * data_out, MPI_Comm comm)
{   // input (Filename, Variable name, data pointer)
    /*  Serial read
        Note: the data_out will be in all processes
    */
    int ncid, varid, retval;

    if ((retval = nc_open_par(file_name, NC_NOWRITE, comm, MPI_INFO_NULL, &ncid))) ERR(retval); // open the file
    
    if ((retval = nc_inq_varid(ncid, var_name, &varid))) ERR(retval); // get the varible id of the file
    
    if ((retval = nc_get_var(ncid, varid, data_out))) ERR(retval); //get data in floats
    
    if ((retval = nc_close(ncid))) ERR(retval); // close the file

}


static void get_sub_block_indices(ElInt GlobalMatrows, ElInt GlobalMatcols, ElInt Grows, ElInt Gcols, \
                ElInt myrow, ElInt mycol, ElRange_i *rowRange, ElRange_i *colRange, size_t * block_dims)
{   
    /* Helper Function to get contigoues subblock indices*/

    ElInt row_size = GlobalMatrows/Grows;
    ElInt col_size = GlobalMatcols/Gcols;

    rowRange->beg = row_size*myrow ;
    colRange->beg = col_size*mycol ;

    if (myrow != Grows-1) rowRange->end = row_size*(1+myrow) - 1 ;
    else rowRange->end = GlobalMatrows-1;

    if (mycol != Gcols-1) colRange->end = col_size*(1+mycol) - 1 ;
    else colRange->end = GlobalMatcols-1;
    
    if ((rowRange->end - rowRange->beg + 1 < 0) || (colRange->end - colRange->beg + 1 < 0) )
    {
        printf("Error : Wrong block_dimensions \n"); 
        MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
    }
    block_dims[0] = rowRange->end - rowRange->beg + 1 ;
    block_dims[1] = colRange->end - colRange->beg + 1 ;

}

static void Function(set_block,Nd_cmplxS) (EL_type(ElDistMatrix,Nd_cmplxS) A, BS_cmplx * A_sub, \
                                        ElRange_i rowRange, ElRange_i colRange, bool conjugate)
{   
    ElError error;

    ElInt block_row_dims = rowRange.end - rowRange.beg + 1 ;
    ElInt block_col_dims = colRange.end - colRange.beg + 1 ;

    EL_Function(ElDistMatrixReserve,Nd_cmplxS) (A, block_row_dims*block_col_dims);

    for (ElInt i =0 ; i<block_row_dims; ++i)
    {   
        ElInt iglobal = i+rowRange.beg;
        for (ElInt j =0 ; j<block_col_dims; ++j)
        {   
            ElInt jglobal = j+colRange.beg;

            BS_cmplx element = A_sub[j+block_col_dims*i];

            if (conjugate) element = conj(element);

            //printf("%f + %fI \n", creal(element),cimag(element));
            //printf("%d , %d \n", iglobal, jglobal);
            
            error = EL_Function(ElDistMatrixQueueUpdate,Nd_cmplxS) (A, iglobal, jglobal, element);

            EL_ABORT_ON_ERROR( error );
        }
    }
    EL_Function(ElDistMatrixProcessQueues,Nd_cmplxS) (A);

}

static void Function(get_block,Nd_cmplxS) (EL_type(ElDistMatrix,Nd_cmplxS) A, BS_cmplx * A_sub, \
                                        ElRange_i rowRange, ElRange_i colRange, bool conjugate)
{   
    ElError error;

    ElInt block_row_dims = rowRange.end - rowRange.beg + 1 ;
    ElInt block_col_dims = colRange.end - colRange.beg + 1 ;

    EL_Function(ElDistMatrixReservePulls,Nd_cmplxS) (A, block_row_dims*block_col_dims);

    for (ElInt i =0 ; i<block_row_dims; ++i)
    {   
        ElInt iglobal = i+rowRange.beg;
        for (ElInt j =0 ; j<block_col_dims; ++j)
        {   
            ElInt jglobal = j+colRange.beg;

            BS_cmplx element;
            
            error = EL_Function(ElDistMatrixQueuePull,Nd_cmplxS) (A, iglobal, jglobal);

            //printf("%f + %fI \n", creal(element),cimag(element));

            A_sub[j+block_col_dims*i] = element;
            
            EL_ABORT_ON_ERROR( error );
        }
    }

    EL_Function(ElDistMatrixProcessPullQueue,Nd_cmplxS) (A, A_sub);

    if (conjugate)
    {   
        ND_int block_size = block_row_dims*block_col_dims;

        for (ND_int i =0; i<block_size; ++i)
        {
            A_sub[i] = conj(A_sub[i]);
        }
    }

}


