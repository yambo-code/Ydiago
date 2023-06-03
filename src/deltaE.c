#include "bse_diagonalize.h"

void Function(YamboDeltaE, Nd_cmplxS) (ND_array(Nd_floatS) * energies_ibz, nd_arr_i * kmap, nd_arr_i * KplusQidxs, nd_arr_i * KminusQidxs, \
                                    ND_array(Nd_floatS) * bse_table ,ND_array(Nd_cmplxS) * delta_energies, bool anti_res_symm)
{
    /*
    Computed Delta E from energies
    INPUT:
    energies_ibz --> Energies for QP band gaps (nspin, nkibz, nbands)
    kmap         --> Kmap (nbz,2)
    bse_table    --> (5,bse_dim). (i_k, i_v, i_c, i_s_c, i_s_v )
    KplusQidxs --> K+Q map

    OUTPUT:
    delta_energies --> (bse_dim) 
    
    */
    
    ND_int bse_dim = delta_energies->dims[0];

    ND_int k_dim = bse_dim;

    if (!anti_res_symm) k_dim = k_dim/2 ;

    for (ND_int i =0 ; i<bse_dim; ++i )
    {
        ND_int i_k, i_v, i_c, i_s_c, i_s_v, k_ibz, ik_c, ik_v, k_ibz_c,k_ibz_v;

        i_k   = ((ND_int)*ND_function(ele, Nd_floatS) (bse_table, nd_idx{0,i})) - 1;
        i_v   = ((ND_int)*ND_function(ele, Nd_floatS) (bse_table, nd_idx{1,i})) - 1;
        i_c   = ((ND_int)*ND_function(ele, Nd_floatS) (bse_table, nd_idx{2,i})) - 1;
        i_s_c = ((ND_int)*ND_function(ele, Nd_floatS) (bse_table, nd_idx{3,i})) - 1;
        i_s_v = ((ND_int)*ND_function(ele, Nd_floatS) (bse_table, nd_idx{4,i})) - 1;

        k_ibz = *nd_ele_i(kmap,nd_idx{i_k,0}) ;

        if (i<k_dim)
        {
            ik_c = KminusQidxs->data[i_k];

            k_ibz_v = *nd_ele_i(kmap,nd_idx{ik_c,0}) ;

            k_ibz_c = k_ibz;
        }
        else
        {
            ik_c = KplusQidxs->data[i_k];

            k_ibz_c = *nd_ele_i(kmap,nd_idx{ik_c,0}) ;

            k_ibz_v = k_ibz;
        }
        
        delta_energies->data[i] = *ND_function(ele, Nd_floatS) (energies_ibz, nd_idx{i_s_c,k_ibz_c,i_c}) \
                                - *ND_function(ele, Nd_floatS) (energies_ibz, nd_idx{i_s_v,k_ibz_v,i_v}); // Ekc-E(k-Q)v
    }

}


