import numpy as np

np.set_printoptions(suppress=True)

ndim =100



def Matrix_herm(n):
    a = np.random.rand(n,n) + 1j*np.random.rand(n,n)
    a = a + a.T.conj()
    a = 5j+ 2.0 + a/2.0
    a *= np.tri(*a.shape)
    a = a@a.T.conj()
    a = a + a.T.conj()
    a = a/2.0
    return a.astype(np.csingle)

def Matrix_symm(n):
    a = np.random.rand(n,n) + 1j*np.random.rand(n,n)
    a = a + a.T
    a = a/2.0
    return a.astype(np.csingle)

def write_to_file(file_name,arrrr):
    np.savetxt(file_name,np.c_[arrrr.real,arrrr.imag])

def Matrix_BSE(n):
    if n%2 == 1: n = n + 1

    bse_mat = np.zeros((n,n),dtype=complex)
    nhalf = n//2
    AA = Matrix_herm(nhalf)
    CC = Matrix_symm(nhalf)*1
    bse_mat[:nhalf,:nhalf] = AA
    bse_mat[nhalf:,nhalf:] = -AA.T
    bse_mat[:nhalf,nhalf:] = CC
    bse_mat[nhalf:,:nhalf] = -CC.conj()
    return bse_mat.astype(np.csingle)

#print(Matrix_BSE(4).real)



bse_matrix = Matrix_BSE(ndim)
her_matrix = Matrix_herm(ndim)

bse_eigs = np.sort(np.linalg.eig(bse_matrix)[0])
her_eigs = np.sort(np.linalg.eig(her_matrix)[0])

write_to_file('BSE_100.mat', bse_matrix.reshape(-1))
write_to_file('BSE_100_eigs.mat',bse_eigs[ndim//2:].reshape(-1))


write_to_file('Herm_100.mat',her_matrix.reshape(-1))
write_to_file('Herm_100_eigs.mat',her_eigs.reshape(-1))



