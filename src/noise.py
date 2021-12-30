import numpy as np

class Noise:
    def __init__(self, nu, nLST, ants, power, deltaNu, deltaT):
        self.nu = nu
        self.nLST = nLST
        self.ants = ants
        self.power = power
        self.deltaNu = deltaNu
        self.deltaT = deltaT
    
    def thermLevel(self):
        noiselev = self.power/np.sqrt(self.deltaNu*pow(10, 6)*self.deltaT*3600)
        return noiselev

    def noiseRealz(self):
        grvec = np.random.randn(*self.thermLevel().shape)
        return self.thermLevel()*grvec
    
    def covmat(self):
        covmat = np.zeros(shape=(len(self.nu)*self.nLST*self.ants,
                                 len(self.nu)*self.nLST*self.ants))
        for i in range(len(self.nu)*self.nLST*self.ants):
            covmat[i, i] = pow(self.thermLevel()[i], 2.0)
        return covmat

    def covmatInv(self):
        return self.invDiag(self.covmat())
    
    def wgtTs(self, modset, opt):
        if opt == '21':
            exp21 = np.identity(len(self.nu))
            exp21 = np.tile(exp21, (self.nLST*self.ants, 1))
            covmat21Inv = np.matmul(exp21.T, self.mulDiag(self.covmatInv(), exp21))
            wgtTs = self.mulDiag(np.sqrt(covmat21Inv), modset)
        if opt == 'FG':
            wgtTs = self.mulDiag(np.sqrt(self.covmatInv()), modset)
        return wgtTs
        
    @staticmethod
    def invDiag(A):
        B = np.zeros(A.shape)
        np.fill_diagonal(B, 1./np.diag(A))
        return B

    @staticmethod
    def mulDiag(A, B):
        return np.multiply(np.diag(A)[:, None], B)
