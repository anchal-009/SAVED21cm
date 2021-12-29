import numpy as np

class Basis:
    def __init__(self, nu):
        self.nu = nu
    
    def wgtSVDbasis(self, modset, covmat, opt, nLST=1, ants=1):
        """This methods returns the noise covariance weighted modes obtained from
        the Singular Value Decomposition of the training sets.

        Args:
            modset (array): Modelling set
            covmat (array): noise covariance matrix
            opt (string): option to choose ('FG' or '21')
            nLST (int, optional): number of time bins to use in the analysis. Defaults to 1.
            ants (int, optional): number of antenna designs. Defaults to 1.

        Returns:
            array: noise covariance weighted SVD modes
        """
        print('Basis: Performing SVD of %s modelling set...'%opt, end='', flush=True)
        if opt == 'FG':
            F, _, _ = np.linalg.svd(modset, full_matrices=False)
            F = self.mulDiag(np.sqrt(covmat), F)

        if opt == '21':
            F, _, _ = np.linalg.svd(modset, full_matrices=False)
            exp21 = np.identity(len(self.nu))
            exp21 = np.tile(exp21, (nLST*ants, 1))
            covmatInv = self.invDiag(covmat)
            covmat21Inv = np.matmul(exp21.T, self.mulDiag(covmatInv, exp21))
            covmat21 = self.invDiag(covmat21Inv)
            F = self.mulDiag(np.sqrt(covmat21), F)
        print('Done!')
        return F
    
    def unwgtSVDbasis(self, modset):
        """This method returns the (unweighted) modes simply obtained from the 
        Singular Value Decomposition of the modelling set.

        Args:
            modset (array): modelling set

        Returns:
            array: unweighted SVD modes
        """
        F, _, _ = np.linalg.svd(modset, full_matrices=False)
        return F

    @staticmethod
    def mulDiag(A, B):
        return np.multiply(np.diag(A)[:, None], B)

    @staticmethod
    def invDiag(A):
        B = np.zeros(A.shape)
        np.fill_diagonal(B, 1./np.diag(A))
        return B
