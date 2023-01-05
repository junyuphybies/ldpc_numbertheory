from scipy import sparse
import numpy as np 

def BinaryRepMat(mat):
    rows = [int(''.join(map(str, list(row))), 2) for row in mat]
#     for row in mat:
#         binary_str = ''.join(map(str, list(row)))
#         rows.append(int(binary_str, 2))
        
    return rows

def gf2_rank(rows):
    """
    Find rank of a matrix over GF2.

    The rows of the matrix are given as nonnegative integers, thought
    of as bit-strings.

    This function modifies the input list. Use gf2_rank(rows.copy())
    instead of gf2_rank(rows) to avoid modifying rows.
    """
    rows = BinaryRepMat(rows)
    rank = 0
    while rows:
        pivot_row = rows.pop()
        if pivot_row:
            rank += 1
            lsb = pivot_row & -pivot_row
            for index, row in enumerate(rows):
                if row & lsb:
                    rows[index] = row ^ pivot_row
    return rank


def local_check(m, delta):
    check = np.random.randint(low=0, high=2, size=(m, delta))
    flag = 0 
    while gf2_rank(check) != m or flag < 10000: 
        check = np.random.randint(low=0, high=2, size=(m, delta))
        flag = 1
    return check





class BuildqcssCodes: 
    def __init__(self,ma:int, 
                mb:int, 
                delta:int, 
                basedim:int, 
                ) -> None:
        self.ma = ma
        self.mb =mb 
        self.delta = delta
        self.basedim = basedim 
        self.edgedim = delta * basedim
        self.facedim = delta * delta * basedim

    
    def build_codes(self, TGVEhor_idx:list, TGVEver_idx:list, TGEFhor_idx:list, TGEFver_idx:list):
        '''
        Build codes for the sparse index given in the matrix form
        (newest)
        '''
        TGEFhor = np.zeros((2 * self.ma * self.edgedim ,self.facedim ), dtype=int)
        TGEFver = np.zeros((2 * self.mb * self.edgedim ,self.facedim ), dtype=int)
        TGVEhor = np.zeros((4 * self.ma * self.mb * self.basedim,2 * self.ma * self.edgedim),dtype=int)
        TGVEver = np.zeros((4 * self.ma * self.mb * self.basedim, 2 * self.mb * self.edgedim),dtype=int)
        for idx in TGEFhor_idx:
            idx = self._reindex(idx)
            # print(idx)
            # print(tuple(idx))
            TGEFhor[idx] = 1
        for idx in TGEFver_idx:
            idx = self._reindex(idx) 
            # print(idx)
            TGEFver[idx] = 1 
        for idx in TGVEhor_idx:
            idx = self._reindex(idx) 
            TGVEhor[idx] = 1 
        for idx in TGVEver_idx:
            idx = self._reindex(idx) 
            TGVEver[idx] = 1  
        self.TGEFhor = TGEFhor
        self.TGEFver = TGEFver
        self.TGVEhor = TGVEhor
        self.TGVEver = TGVEver
        TGEF = np.concatenate([self.TGEFhor, self.TGEFver], axis=0)
        TGVE = np.concatenate([self.TGVEhor, self.TGVEver], axis=1)
        return sparse.csr_matrix(TGEF), sparse.csr_matrix(TGVE)


    def build_chains(self, idTGVE:list, idTGEF:list): 
        
        '''
        Given derived graph code, we build explicit the X, Z parity checks. 
        '''

        TGEF = np.zeros((2 * (self.ma * self.edgedim + self.mb * self.edgedim), self.facedim), dtype=int)
        TGVE = np.zeros((4 * self.ma * self.mb * self.basedim, 2 * (self.ma * self.edgedim + self.mb * self.edgedim)), dtype=int)
        for idx in idTGVE:
            idx = self._reindex(idx)
            TGVE[idx] = 1
        for idx in idTGEF:
            idx = self._reindex(idx)
            TGEF[idx] = 1 
        return TGVE, TGEF.transpose()

    def check_exactseq(self, TGEF:sparse.csr_matrix, TGVE: sparse.csr_matrix):
        '''
        check if XZ^T = 0 mod 2
        '''
        T = TGVE._mul_sparse_matrix(TGEF).toarray()
        # T = T.toarray()
        # np.where(T==1)
        # np.where(T>2)
        print(f'the number of entry return 1 after mod 2: {print(T[T % 2 ==1])}')
        assert len(T[T % 2 ==1]) == 0 # check if The CSS condition is fulfilled. 
    
    def check_exactseq2(self):
        '''
        check if XZ^T = 0 mod 2
        '''
        Thor = sparse.csr_matrix(self.TGVEhor)._mul_sparse_matrix(sparse.csr_matrix(self.TGEFhor)).toarray()
        Tver = sparse.csr_matrix(self.TGVEver)._mul_sparse_matrix(sparse.csr_matrix(self.TGEFver)).toarray()
        T = Thor + Tver 
        # T = T.toarray()
        # np.where(T==1)
        # np.where(T>2)
        print(len(T[T % 2 ==1]))
        assert len(T[T % 2 ==1]) == 0 # THe CSS condition is fulfilled.

    def check_lowdensity(self, Z: np.array, X: np.array):
        Z_row = np.count_nonzero(Z, axis=1)
        Z_col = np.count_nonzero(Z, axis=0)
        # print(len(Z_row))
        print(f'the maximum weight given Z stabilizer: {np.max(Z_row)}, should be bounded by {4 * max(self.ma, self.mb)} by Cayley Complexes ')
        print(f'the maximum number of Z stabilizers on a given qubit: {np.max(Z_col)}, should be bounded by {self.delta}')

        X_row = np.count_nonzero(X, axis=1)
        X_col = np.count_nonzero(X, axis=0)
        print(f'the maximum weight given X stabilizer: {np.max(X_row)}, should be bounded by {2 * self.delta}')
        print(f'the maximum number of X stabilizers on a given qubit: {np.max(X_col)}, should be bounded by {2 * max(self.ma, self.mb)}')
    
    def _reindex(self, index:list):
        new_index = []
        for i in index: 
            new_index.append(i -1)
        return tuple(new_index)
    




# HA = local_check(3, 14)
# HB = local_check(8, 14)
# print(HA)
# print(HB)

    
