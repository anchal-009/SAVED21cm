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
        grvec = np.random.randn(*noiselev.shape)
        return noiselev, noiselev * grvec
    
    def covmat(self, noise):
        covmat = np.zeros(shape=(len(self.nu)*self.nLST*self.ants,
                                 len(self.nu)*self.nLST*self.ants))
        for i in range(len(self.nu)*self.nLST*self.ants):
            covmat[i, i] = pow(noise[i], 2.0)
        return covmat

    def covmatInv(self, noise):
        return self.invDiag(self.covmat(noise))
    
    def wgtTs(self, modset, opt, noise):
        if opt == '21':
            exp21 = np.identity(len(self.nu))
            exp21 = np.tile(exp21, (self.nLST*self.ants, 1))
            covmat21Inv = np.matmul(exp21.T, self.mulDiag(self.covmatInv(noise), exp21))
            wgtTs = self.mulDiag(np.sqrt(covmat21Inv), modset)
        if opt == 'FG':
            wgtTs = self.mulDiag(np.sqrt(self.covmatInv(noise)), modset)
        return wgtTs
        
    @staticmethod
    def invDiag(A):
        B = np.zeros(A.shape)
        np.fill_diagonal(B, 1./np.diag(A))
        return B

    @staticmethod
    def mulDiag(A, B):
        return np.multiply(np.diag(A)[:, None], B)
