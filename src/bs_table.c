#include "bse_diagonalize.h"

static ND_int Function(get_kstar, i)( nd_arr_i * kmap , ND_int ibz , ND_int * kpoints);


void Function(BS_table, i) (ND_array(Nd_floatS) * bse_table, nd_arr_i * kmap, ND_int nibz, ND_int vmin, ND_int vmax, \
                            ND_int cmin, ND_int cmax, ND_int nspin, bool anti_res_symm, bool magnons )
{
    /*
    Creates BS_table

    OUTPUT: bs_table (5,bse_dim). (i_k, i_v, i_c, i_s_c, i_s_v )

    INPUT:
    kmap (nbz,2)  --> kmap related to ibz kpoints, sym and bz kpoints
    (vmin, vmax)  --> valance bands
    (cmin, cmax)  --> conduction bands
    nspin       -->  number of spins
    anti_res_symm --> if anti-res part can be obtained from res part.
    */
    
    ND_int n_bse_blocks = 1;

    if (!anti_res_symm) n_bse_blocks = 2 ;

    ND_int nvbands = vmax-vmin +1;
    ND_int ncbands = cmax-cmin +1;

    ND_int bse_counter = 0; 

    ND_int * kstar = malloc(sizeof(ND_int)*(kmap->dims[0]));

    for (ND_int iblock = 1; iblock < n_bse_blocks+1; ++iblock)
    {
        for (ND_int ibz = 0; ibz < nibz; ++ibz )
        {
            for (ND_int iv = 0; iv < nvbands; ++iv )
            {
                for (ND_int ic = 0; ic < ncbands; ++ic )
                {
                    for (ND_int ispinc = 1; ispinc < nspin+1; ++ispinc )
                    {   
                        /* For excitons */
                        ND_int ispinv = ispinc;

                        /* For magnons */
                        if (magnons && nspin==2)
                        {
                            if(iblock==1 && ispinc==1) continue ;

                            if(iblock==2 && ispinc==2) continue ;

                            ispinv = (ispinc%nspin) + 1 ;
                        }

                        ND_int nkstar = Function(get_kstar, i)(kmap, ibz, kstar);

                        for (ND_int ik = 0 ; ik < nkstar; ++ik)
                        {   
                            //bs_table (5,bse_dim). (i_k, i_v, i_c, i_s_c, i_s_v )

                            *ND_function(ele, Nd_floatS) (bse_table, nd_idx{0,bse_counter}) = kstar[ik] + 1;

                                *ND_function(ele, Nd_floatS) (bse_table, nd_idx{1,bse_counter}) = iv + vmin;
                                *ND_function(ele, Nd_floatS) (bse_table, nd_idx{2,bse_counter}) = ic + cmin;
                                *ND_function(ele, Nd_floatS) (bse_table, nd_idx{3,bse_counter}) = ispinc;
                                *ND_function(ele, Nd_floatS) (bse_table, nd_idx{4,bse_counter}) = ispinv;

                            if (iblock == 2)
                            {
                                *ND_function(ele, Nd_floatS) (bse_table, nd_idx{1,bse_counter}) = ic + cmin;
                                *ND_function(ele, Nd_floatS) (bse_table, nd_idx{2,bse_counter}) = iv + vmin;
                                *ND_function(ele, Nd_floatS) (bse_table, nd_idx{3,bse_counter}) = ispinv;
                                *ND_function(ele, Nd_floatS) (bse_table, nd_idx{4,bse_counter}) = ispinc;
                            }

                            ++bse_counter;

                        } // end k star loop
                    } // end spin loop
                } // end ic loop
            } // end iv loop
        } // end ibz loop
    } // end block loop

    free(kstar);
} // end of Function




/* Function to get the kstar of a particular kpoint in iBZ*/
static ND_int Function(get_kstar, i)( nd_arr_i * kmap , ND_int ibz , ND_int * kpoints)
{
    /*
    INPUT: kmap ---> kmap as described above
            ibz --> ibz number( starts from 0)
    OUTPUT:
        return: number of kpoints in this kstar
        kpoints --> array containing kpoint numbers (starts from 0)
    */

    ND_int Nbz = kmap->dims[0] ;

    ND_int counter = 0;

    for (ND_int i = 0 ; i<Nbz ; ++i)
    {
        if (*nd_ele_i(kmap,nd_idx{i,0}) == ibz )
        {   
            if (kpoints != NULL) kpoints[counter] = i;

            ++counter;
        }
    }

    return counter;

}