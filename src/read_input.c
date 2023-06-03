#include "bse_diagonalize.h"
#include "libcfg.h"
#include <ctype.h>


#define DEFAULT_CONF_FILE       "yambo_diago.in"
/* Priority of parameters from different sources. */
#define PRIOR_CMD               5
#define PRIOR_FILE              1
/* Print the warning and error messages. */
#define PRINT_WARNING   cfg_pwarn(cfg, stderr, "\x1B[35;1mWarning:\x1B[0m");
#define PRINT_ERROR     {                                               \
    cfg_perror(cfg, stderr, "\x1B[31;1mError:\x1B[0m");                   \
    cfg_destroy(cfg);                                                     \
    MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );                            \
}


#define READ_FGETS_MAX_LEN 1200

static void sort_int_arr(ND_int * arr_in);
static void sort_double_arr(BS_float * arr_in);

static ND_int parser_doubles_from_string(char * str, double * out);

static char *str_reverse_in_place(char *str);
static bool string_start_with(char * str, char * compare_str, bool trim);
static bool string_end_with  (char * str, char * compare_str, bool trim);

static bool check_file_exists(const char *fname);



void parse_inputs(  int argc, char* argv[], char * bse_report_file, int * nq, ND_int * eig_num_range, \
                    BS_float * eig_val_range, ND_int * eig_num_range_anti, BS_float * eig_val_range_anti, \
                    char * bse_job_name, bool * eig_num_present, bool * eig_val_present, bool * eig_num_present_anti, \
                    bool * eig_val_present_anti)
{   
    /******* Setting up parameters *******/
    
    const int npar = 8;
    int n = 0, optidx = 0;
    int nq1;
    int * eig_num_rang1;
    double * eig_val_range1;
    int * eig_num_range_anti1;
    double * eig_val_range_anti1;
    char * bse_job_name1; 
    char * input_file;
    char * bse_report_file1;

    *eig_num_present        = true;
    *eig_val_present        = true;
    *eig_num_present_anti   = true;
    *eig_val_present_anti   = true;

    cfg_param_t * params = malloc(npar*sizeof(cfg_param_t));
    
    /** Define parameters **/
    params[0] =     (cfg_param_t){   'F',       "inp",                     "INPUT_FILE",            CFG_DTYPE_STR,  &input_file             };
    params[1] =     (cfg_param_t){   'J',       "job",                     "JOB_NAME",              CFG_DTYPE_STR,  &bse_job_name1          };
    //------------------------------------- Below read from input file --------------------------
    params[2] =     (cfg_param_t){   'R',       "bse_report_file",         "bse_report_file",       CFG_DTYPE_STR,  &bse_report_file1        };
    params[3] =     (cfg_param_t){   'q',       "nq",                      "nq",                    CFG_DTYPE_INT,  &nq1                    };
    params[4] =     (cfg_param_t){   'n',       "neig",                    "neig",                  CFG_ARRAY_INT,  &eig_num_rang1          };
    params[5] =     (cfg_param_t){   'v',       "eig_val_range",           "eig_val_range",         CFG_ARRAY_DBL,  &eig_val_range1         };
    params[6] =     (cfg_param_t){   'N',       "neig_anti",               "neig_anti",             CFG_ARRAY_INT,  &eig_num_range_anti1    };
    params[7] =     (cfg_param_t){   'V',       "eig_range_anti",          "eig_range_anti",        CFG_ARRAY_DBL,  &eig_val_range_anti1    };
    

    //------------------------------------- Below read from report file -------------------------
    
    /* Initialise the configuations. */
    cfg_t *cfg = cfg_init();

    if (!cfg)
    {
        printf("Error: failed to initlise the configurations.\n");
        MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
    }

    /* Set parameters for cfg*/
    if (cfg_set_params(cfg, params, npar)) PRINT_ERROR; PRINT_WARNING;

    /** Read command line options -J anf -F*/
    if (cfg_read_opts(cfg, argc, argv, PRIOR_CMD, &optidx)) PRINT_ERROR; PRINT_WARNING;
    
    /** Check if input file "-F" is given*/
    if (!cfg_is_set(cfg, &input_file)) input_file = DEFAULT_CONF_FILE;
    
    /* Read stuff from input_file*/
    if (check_file_exists(input_file))
    {   
        if (cfg_read_file(cfg, input_file, PRIOR_FILE)) PRINT_ERROR;
    }
    /* Read JOb name of BSE*/
    if (cfg_is_set(cfg, &bse_job_name1))
    {
        strcpy(bse_job_name,bse_job_name1);
    }
    else
    {
        strcpy(bse_job_name,"SAVE");
    }

    /* Check if report file is given in the input*/
    if (!cfg_is_set(cfg, &bse_report_file1))
    {
        printf("Error: Please provide report file of BSE calculation with -fatlog in the input file.\n");
        MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
    } 

    /* Read NQ (qpoint)*/
    if (cfg_is_set(cfg, &nq1)) *nq = nq1;
    else *nq = 1;

    /** Read eig_num range from input*/
    if (cfg_is_set(cfg, &eig_num_rang1)) 
    {
        n = cfg_get_size(cfg, &eig_num_rang1);

        if (n == 1)
        {   
            eig_num_range[0] = 0;

            if (eig_num_rang1[0] > 1) eig_num_range[1] = eig_num_rang1[0]-1;
            
            else if (eig_num_rang1[0] == 1) eig_num_range[1] = eig_num_rang1[0];
            
            else *eig_num_present = false;
        }
        else
        {   
            if (eig_num_range[0] > 0 && eig_num_range[1] > 1)
            {
                eig_num_range[0] = eig_num_rang1[0]-1;
                eig_num_range[1] = eig_num_rang1[1]-1;
            }
            else
            {
                *eig_num_present = false;
            }
        }

    }
    else
    {
        *eig_num_present = false;
    }

    /** Read eig_value range from input*/
    if (cfg_is_set(cfg, &eig_val_range1)) 
    {   
        n = cfg_get_size(cfg, &eig_val_range1);

        if ( n == 2)
        {
            eig_val_range[0] = eig_val_range1[0];
            eig_val_range[1] = eig_val_range1[1];
        }
        else
        {
            *eig_val_present  = false;
        }
    }
    else
    {
        *eig_val_present  = false;
    }

    //....
    /** Read eig_num for anti resonant from input*/
    if (cfg_is_set(cfg, &eig_num_range_anti1 )) 
    {
        n = cfg_get_size(cfg, &eig_num_range_anti1 );

        if (n == 1)
        {   
            eig_num_range_anti[0] = 0;

            if (eig_num_range_anti1[0] > 1) eig_num_range_anti[1] = eig_num_range_anti1[0]-1;
            
            else if (eig_num_range_anti1[0] == 1) eig_num_range_anti[1] = eig_num_range_anti1[0];
            
            else *eig_num_present_anti  = false;
        }
        else
        {   
            if (eig_num_range_anti1[0] > 0 && eig_num_range_anti1[1] > 1)
            {
                eig_num_range_anti[0] = eig_num_range_anti1[0]-1;
                eig_num_range_anti[1] = eig_num_range_anti1[1]-1;
            }
            else
            {
                *eig_num_present_anti  = false;
            }
        }

    }
    else
    {
        *eig_num_present_anti  = false;
    }

    /** Read eig_value range anti-resonant from input*/
    if (cfg_is_set(cfg, &eig_val_range_anti1)) 
    {   
        n = cfg_get_size(cfg, &eig_val_range_anti1);

        if ( n == 2)
        {
            eig_val_range_anti[0] = eig_val_range_anti1[0];
            eig_val_range_anti[1] = eig_val_range_anti1[1];
        }
        else
        {
            *eig_val_present_anti   = false;
        }
    }
    else
    {
        *eig_val_present_anti   = false;
    }

    strcpy( bse_report_file, bse_report_file1);

    /** Sort the eigen_range and eigen_value */
    if (*eig_num_present)       sort_int_arr(eig_num_range);
    if (*eig_num_present_anti)  sort_int_arr(eig_num_range_anti);
    if (*eig_val_present)       sort_double_arr(eig_val_range);
    if (*eig_val_present_anti)  sort_double_arr(eig_val_range_anti);

    if (cfg_is_set(cfg, &eig_num_rang1))        free(eig_num_rang1);
    if (cfg_is_set(cfg, &eig_val_range1))       free(eig_val_range1);
    if (cfg_is_set(cfg, &eig_num_range_anti1))  free(eig_num_range_anti1);
    if (cfg_is_set(cfg, &eig_val_range_anti1))  free(eig_val_range_anti1);

    if (cfg_is_set(cfg, &bse_job_name1))        free(bse_job_name1); 
    if (cfg_is_set(cfg, &bse_report_file1))     free(bse_report_file1);

    /* Cleanup */
    if (cfg_is_set(cfg, &input_file)) free(input_file);
    /** Cleanup */
    cfg_destroy(cfg);

    free(params);

    // Either eig_num or eig_val should be given, when both read eig_val
    if (*eig_num_present && *eig_val_present ) *eig_num_present = false ;
    if (*eig_num_present_anti && *eig_val_present_anti ) *eig_num_present_anti = false;

}



static ND_int parser_doubles_from_string(char * str, double * out)
{
    char * p = str;
    char * q;
    double temp_val;
    ND_int count = 0;
    
    while (*p)
    {
        if ( isdigit(*p) || ( (*p=='-'||*p=='+') && isdigit(*(p+1)) ))
        {   
            temp_val = strtod(p, &q);
            
            if (p == q) break;
            else p = q ;
            
            if (out != NULL) out[count] = temp_val;
            ++count;
        }
        else ++p;
    }
    return count;
}




static char *str_reverse_in_place(char *str)
{   
    ND_int len = strlen(str);
    
    if (len == 0) return str;

    char *p1 = str;
    
    char *p2 = str + len - 1;

    while (p1 < p2) {
        char tmp = *p1;
        *p1++ = *p2;
        *p2-- = tmp;
    }
    return str;
}


static bool string_start_with(char * str, char * compare_str, bool trim)
{   
    char * a;
    char * b;
    a = str;
    b = compare_str;
    if (trim)
    {
        while(isspace(*a)) ++a;
        while(isspace(*b)) ++b;
    }

    return !strncmp(a, b, strlen(b));
}

static bool string_end_with(char * str, char * compare_str, bool trim)
{   
    char * temp_str = malloc(sizeof(char) * ( strlen(str) + strlen(compare_str) + 2 ));
    char * a = temp_str;
    char * b = temp_str+strlen(str)+1;
    
    strcpy(a,str);
    strcpy(b,compare_str);

    str_reverse_in_place(a);
    str_reverse_in_place(b);
    if (trim)
    {
        while(isspace(*a)) ++a;
        while(isspace(*b)) ++b;
    }

    bool ret_value = !strncmp(a, b, strlen(b));
    free(temp_str);
    return ret_value;
}

/******************************** Sorting Functions ******************************************/

static void sort_int_arr(ND_int * arr_in)
{
    if (arr_in[0]<=arr_in[1]) return ;
    else
    {
        ND_int temp_x = arr_in[1];
        arr_in[1] = arr_in[0];
        arr_in[0] = temp_x;
    }
}


static void sort_double_arr(BS_float * arr_in)
{
    if (arr_in[0]<=arr_in[1]) return ;
    else
    {
        BS_float temp_x = arr_in[1];
        arr_in[1] = arr_in[0];
        arr_in[0] = temp_x;
    }
}

/*********************************************************************************************/

static bool check_file_exists(const char *fname)
{
    FILE *file;
    if ((file = fopen(fname, "r")))
    {
        fclose(file);
        return true;
    }
    return false;
}

/*********************************************************************************************/

bool get_GW_from_report_file(const char * report_file, ND_int nspin, int * calc_type, bool * has_inv,\
                            bool * is_metal, int * metal_bands, ND_array(Nd_floatS) * energies_ibz)
{
    char * qp_str;

    if (nspin == 2) qp_str = "Eqp [up] @ K" ;
    else qp_str = "Eqp @ K";

    FILE* fp = fopen(report_file, "r");

    if (fp == NULL)
    {
        printf("Error : Cannot open the following report file : %s \n",report_file);
        MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
    }
    char * line = malloc(sizeof(char)*(3*READ_FGETS_MAX_LEN));

    char * line_parse_temp = line + READ_FGETS_MAX_LEN ;

    char * line_parse_temp2 = line + 2*READ_FGETS_MAX_LEN ;

    bool E_found = false;

    double temp_vals[8]; // Yambo writes 8 values 

    bool read_lines = false;

    ND_int nEqp_per_line = 0;

    int ik=0, ispin = 0, ibnd = 0 ;

    double ik_temp;

    *is_metal = false;

    while (fgets(line ,READ_FGETS_MAX_LEN, fp))
    {
        if (string_start_with(line, "| BSEprop", true) && (sscanf(line, "%*[^\"]\"%31[^\"]\"", line_parse_temp2) == 1))
        {   
            sscanf(line_parse_temp2, "%s", line_parse_temp);
            if      (!(strcmp(line_parse_temp,"abs")))                                                            *calc_type = 0;
            else if (!(strcmp(line_parse_temp,"jdos")))                                                           *calc_type = 1;
            else if (!(strcmp(line_parse_temp,"kerr")))                                                           *calc_type = 2;
            else if (!(strcmp(line_parse_temp,"magn")))                                                           *calc_type = 3;
            else if (!(strcmp(line_parse_temp,"dich")))                                                           *calc_type = 4;
            else if (!(strcmp(line_parse_temp,"photolum")))                                                       *calc_type = 5;
            else if (!(strcmp(line_parse_temp,"esrt")))                                                           *calc_type = 6;
            else { printf("Warning : BSEprop was not found in the report file, assuming it to be 'abs' \n"); *calc_type = 0;}
        }

        if (string_start_with(line, "Inversion symmetry", true))
        {
            // Read Inversion symmetry 
            if (sscanf(line, "%*[^:]:%10[^\n]", line_parse_temp2) == 1)
            {   
                sscanf(line_parse_temp2, "%s", line_parse_temp);
                if      (!(strcmp(line_parse_temp,"yes")))  *has_inv = true;
                else if (!(strcmp(line_parse_temp,"no")))   *has_inv = false;
                else
                {
                    printf("Error : Inversion symmetry details not found in the provided report !! : %s \n",report_file);
                    MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
                }
            }
        }
        // Write for spin case
        if (string_start_with(line, "[X] Metallic Bands", true))
        {
            *is_metal = true;

            if (nspin == 2)
            {
                if      (string_start_with(line, "[X] Metallic Bands                       [spin UP]", true))
                {
                    sscanf(line, "%*[^:]:%30[^\n]", line_parse_temp);
                    sscanf(line_parse_temp, "%d %d", metal_bands,metal_bands+1);
                }
                else if (string_start_with(line, "[X] Metallic Bands                       [spin DN]", true))
                {
                    sscanf(line, "%*[^:]:%30[^\n]", line_parse_temp);
                    sscanf(line_parse_temp, "%d %d", metal_bands+2,metal_bands+3);
                }
                else
                {
                    printf("Error : System is metallic but metal bands not found in the report file !! : %s \n",report_file);
                    MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
                }
            }
            else
            {
                sscanf(line, "%*[^:]:%30[^\n]", line_parse_temp);
                sscanf(line_parse_temp, "%d %d", metal_bands,metal_bands+1);
            }
        }
        // Check if system is metal
        if (string_start_with(line, "[WARNING][X] Metallic system", true)) *is_metal = true;

        /***** Parse GW Correction if any*/

        if (string_start_with(line, qp_str, true))
        {
            read_lines = true;
            E_found = true;
            ispin = 0;
            ibnd = 0;
            ND_int temp_k_num =  parser_doubles_from_string(line, &ik_temp);;
            if (temp_k_num != 1)
            {
                printf("Error : Parsing GW energies failed due to bad report file. \n");
                MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
            }
            ik = (int) ik_temp - 1;
        }

        if (nspin == 2)
        {
            if (string_start_with(line, "Eqp [dn] @ K", true)) { ibnd = 0 ; ispin = 1 ;}
        }

        if (string_start_with(line, "occ", true) || string_start_with(line, "", true)) read_lines = false;

        if (read_lines && !string_start_with(line, "Eqp", true))
        {
            nEqp_per_line = parser_doubles_from_string(line, temp_vals);

            for (ND_int ib = ibnd; ib < ibnd + nEqp_per_line ; ++ib ) 
            {   
                printf("%d %d %zu \n",ispin,ik,ib);
                *ND_function(ele, Nd_floatS) (energies_ibz, nd_idx{ispin,ik,ib}) = 0.0367493*temp_vals[ib-ibnd];
            }
            ibnd = ibnd + nEqp_per_line ;
        }


    }


    free(line);

    return E_found;
}


//printf("Debug - 1 \n");
void MPI_Bcast_input_variables(int root, MPI_Comm comm, int * nq, ND_int * eig_num_range, BS_float * eig_val_range,\
                             ND_int * eig_num_range_anti, BS_float * eig_val_range_anti, char * bse_job_name,\
                             bool * eig_num_present, bool * eig_val_present, bool * eig_num_present_anti, bool * eig_val_present_anti)
{
    int rankiD,bse_job_name_len,error_mpi;

    bse_job_name_len = 1;

    ElMPIWorldRank(&rankiD);

    if (rankiD == root) bse_job_name_len = strlen(bse_job_name)+1;
    
    error_mpi = MPI_Bcast( &bse_job_name_len,    1,                         MPI_INT ,          root, comm);
    if (error_mpi != MPI_SUCCESS ) {printf("Error : MPI_Bcast failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}

    if (MPI_Barrier(comm) != MPI_SUCCESS ) {printf("Error : MPI_Barrier failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}

    int temp_num[4] = {eig_num_range[0], eig_num_range[1], eig_num_range_anti[0], eig_num_range_anti[1]};

    error_mpi = MPI_Bcast( nq,                   1,                         MPI_INT ,          root, comm);
    if (error_mpi != MPI_SUCCESS ) {printf("Error : MPI_Bcast failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}

    error_mpi = MPI_Bcast( temp_num,             4,                         MPI_INT,           root, comm);
    if (error_mpi != MPI_SUCCESS ) {printf("Error : MPI_Bcast failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}
    
    error_mpi = MPI_Bcast( eig_val_range,        2,                         MPI_FLOAT_TYPE,    root, comm);
    if (error_mpi != MPI_SUCCESS ) {printf("Error : MPI_Bcast failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}

    error_mpi = MPI_Bcast( eig_val_range_anti,   2,                         MPI_FLOAT_TYPE,    root, comm);
    if (error_mpi != MPI_SUCCESS ) {printf("Error : MPI_Bcast failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}

    error_mpi = MPI_Bcast( bse_job_name,         bse_job_name_len,          MPI_CHAR,          root, comm);
    if (error_mpi != MPI_SUCCESS ) {printf("Error : MPI_Bcast failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}

    error_mpi = MPI_Bcast( eig_num_present,      1,                         MPI_C_BOOL,        root, comm);
    if (error_mpi != MPI_SUCCESS ) {printf("Error : MPI_Bcast failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}

    error_mpi = MPI_Bcast( eig_val_present,      1,                         MPI_C_BOOL,        root, comm);
    if (error_mpi != MPI_SUCCESS ) {printf("Error : MPI_Bcast failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}

    error_mpi = MPI_Bcast( eig_num_present_anti, 1,                         MPI_C_BOOL,        root, comm);
    if (error_mpi != MPI_SUCCESS ) {printf("Error : MPI_Bcast failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}

    error_mpi = MPI_Bcast( eig_val_present_anti, 1,                         MPI_C_BOOL,        root, comm);
    if (error_mpi != MPI_SUCCESS ) {printf("Error : MPI_Bcast failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}

    eig_num_range[0] = temp_num[0]; eig_num_range[1] = temp_num[1] ; eig_num_range_anti[0] = temp_num[2] ; eig_num_range_anti[1] = temp_num[3] ; 

    if (MPI_Barrier(comm) != MPI_SUCCESS ){printf("Error : MPI_Barrier failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}
}



void MPI_Bcast_report_variables(int root, MPI_Comm comm, int * calc_type, bool * has_inv,\
                            bool * is_metal, int * metal_bands, ND_array(Nd_floatS) * energies_ibz)
{   
    if (MPI_Barrier(comm) != MPI_SUCCESS ) {printf("Error : MPI_Barrier failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}

    int error_mpi;

    error_mpi = MPI_Bcast( calc_type,            1,                                                  MPI_INT ,          root, comm);
    if (error_mpi != MPI_SUCCESS ) {printf("Error : MPI_Bcast failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}

    error_mpi = MPI_Bcast( has_inv,              1,                                                  MPI_C_BOOL,        root, comm);
    if (error_mpi != MPI_SUCCESS ) {printf("Error : MPI_Bcast failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}

    error_mpi = MPI_Bcast( is_metal,             1,                                                  MPI_C_BOOL,        root, comm);
    if (error_mpi != MPI_SUCCESS ) {printf("Error : MPI_Bcast failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}

    error_mpi = MPI_Bcast( metal_bands,          4,                                                  MPI_INT ,          root, comm);
    if (error_mpi != MPI_SUCCESS ) {printf("Error : MPI_Bcast failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}

    error_mpi = MPI_Bcast( energies_ibz->data,   ND_function(size, Nd_floatS)(energies_ibz), MPI_FLOAT_TYPE,    root, comm);
    if (error_mpi != MPI_SUCCESS ) {printf("Error : MPI_Bcast failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}

    if (MPI_Barrier(comm) != MPI_SUCCESS ) {printf("Error : MPI_Barrier failed !. \n"); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );}

}