import numpy as np
from numpy.core.fromnumeric import shape

class Dic:
    def __init__(self, nu, nLST, ants):
        """This class contains methods for minimizing Deviance Information Criterion (DIC).

        Args:
            nu (array): frequency range
            nLST (int): number of time bins to use in the analysis
            ants (int): number of antenna designs
        """
        self.nu = nu
        self.nLST = nLST
        self.ants = ants
    
    def gridinfo(self, modesFg, modes21, wgtBasisFg, wgtBasis21, covmatInv, mockObs, file):
        """This method calculates the DIC over a grid of foreground and 21cm modes,
        and save the data into a text file.

        Args:
            modesFg (int): number of foreground modes for the grid
            modes21 (int): number of 21cm modes for the grid
            wgtBasisFg (array): foreground modes
            wgtBasis21 (array): 21cm modes
            covmatInv (array): inverse of noise covariance matrix
            mockObs (array): mock observation
            file (string): filename to save the information
        """
        print('Info Criterion: Calculating info criterion over grid...', end='', flush=True)
        gridFg = np.linspace(1, modesFg, num=modesFg, dtype=int).tolist()
        grid21 = np.linspace(1, modes21, num=modes21, dtype=int).tolist()
        
        exp21 = np.identity(len(self.nu))
        exp21 = np.tile(exp21, (self.nLST*self.ants, 1))     
           
        fp = open(file, 'w')
        
        for i in range(len(gridFg)):
            for j in range(len(grid21)):
                F_Fg = np.zeros(shape=(len(self.nu)*self.nLST*self.ants, gridFg[i]))
                F_21 = np.zeros(shape=(len(self.nu), grid21[j]))
                
                for mm in range(gridFg[i]):
                    F_Fg.T[mm] = wgtBasisFg[:, mm]
                for nn in range(grid21[j]):
                    F_21.T[nn] = wgtBasis21[:, nn]
                    
                F_x21 = np.matmul(exp21, F_21)
                F = np.concatenate((F_Fg, F_x21), axis=-1)
                
                S = np.matmul(F.T, self.mulDiag(covmatInv, F))
                S = np.linalg.inv(S)
                
                Xi = np.matmul(S, np.matmul(F.T, self.mulDiag(covmatInv, mockObs)))
                reconsObs = np.matmul(F, Xi)
                diff = reconsObs - mockObs
                
                dic = np.matmul(diff.T, self.mulDiag(covmatInv, diff)) +\
                    2 * (gridFg[i] + grid21[j])
                
                fp.write(str(gridFg[i]))
                fp.write("\t")
                fp.write(str(grid21[j]))
                fp.write("\t")
                fp.write(str(dic[0][0]))
                fp.write("\n")
        fp.close()
        print('Done!')
    
    def searchMinima(self, file):
        """This method searches for the minima of the information criterion
        from the gridded data.

        Args:
            file (string): filename that contains the gridded IC data.

        Returns:
            int: number of foreground modes that minimize the IC
                 number of 21cm modes that minimize the IC
                 minimized IC value
        """
        print('Info Critetion: Searching for minima over the grid...', end='', flush=True)
        data = np.loadtxt(file)
        Vals = list(data[:, 2])
        minIndex = Vals.index(min(Vals))
        modesFg = int(data[minIndex, 0])
        modes21 = int(data[minIndex, 1])
        minima = data[minIndex, 2]
        print('Done!')
        print('Info Criterion: IC is minimzed for %d FG and %d signal modes.'%(modesFg, modes21))
        return modesFg, modes21, minima
                        
    @staticmethod
    def mulDiag(A, B):
        return np.multiply(np.diag(A)[:, None], B)
    