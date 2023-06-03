#pragma once
#include <El.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single Hermitian Mats */
ElError ElHermitianEigPairControlDist_s( ElUpperOrLower uplo, ElDistMatrix_s A, ElDistMatrix_s w, ElDistMatrix_s Z, ElHermitianEigCtrl_s ctrl );

ElError ElHermitianEigPairControlDist_d( ElUpperOrLower uplo, ElDistMatrix_d A, ElDistMatrix_d w, ElDistMatrix_d Z, ElHermitianEigCtrl_d ctrl );

ElError ElHermitianEigPairControlDist_c( ElUpperOrLower uplo, ElDistMatrix_c A, ElDistMatrix_s w, ElDistMatrix_c Z, ElHermitianEigCtrl_c ctrl );

ElError ElHermitianEigPairControlDist_z( ElUpperOrLower uplo, ElDistMatrix_z A, ElDistMatrix_d w, ElDistMatrix_z Z, ElHermitianEigCtrl_z ctrl );


/* GeneralDef*/
ElError ElHermitianGenDefEigPairControlDist_s(ElPencil pencil, ElUpperOrLower uplo, ElDistMatrix_s A, ElDistMatrix_s B, ElDistMatrix_s w, ElDistMatrix_s Z, ElHermitianEigCtrl_s ctrl);

ElError ElHermitianGenDefEigPairControlDist_d(ElPencil pencil, ElUpperOrLower uplo, ElDistMatrix_d A, ElDistMatrix_d B, ElDistMatrix_d w, ElDistMatrix_d Z, ElHermitianEigCtrl_d ctrl);

ElError ElHermitianGenDefEigPairControlDist_c(ElPencil pencil, ElUpperOrLower uplo, ElDistMatrix_c A, ElDistMatrix_c B, ElDistMatrix_s w, ElDistMatrix_c Z, ElHermitianEigCtrl_c ctrl);

ElError ElHermitianGenDefEigPairControlDist_z(ElPencil pencil, ElUpperOrLower uplo, ElDistMatrix_z A, ElDistMatrix_z B, ElDistMatrix_d w, ElDistMatrix_z Z, ElHermitianEigCtrl_z ctrl);


#ifdef __cplusplus
}
#endif