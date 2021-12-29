import numpy as np

class Extractor:
    def __init__(self, nu, nLST, ants):
        self.nu = nu
        self.nLST = nLST
        self.ants = ants
    
    def extract(self, modesFg, modes21, wgtBasisFg, wgtBasis21, covmatInv, mockObs, y21):
        """This method extracts the 21cm and foreground signals from the mock observation.
        It also calculates some statistical measures of the extracted signals.

        Args:
            modesFg (int): number of foreground modes to fit (selected from min of IC)
            modes21 (int): number of 21cm modes to fit (selected from min of IC)
            wgtBasisFg (array): foreground modes
            wgtBasis21 (array): 21cm modes
            covmatInv (array): inverse of noise covariance matrix
            mockObs (array): mock observation
            y21 (array): input 21cm signal

        Returns:
            reconsFg: reconstructed foregrounds
            recons21: reconstructed 21cm signal
            sigmaFg: 1sigma interval on fitted foregrounds
            sigma21: 1sigma interval on fitted 21cm signal
            channelCovFg: foregrounds channel covariance
            channelCov21: 21cm signal channel covariance
            rmsFg: rms of the fitted foregrounds
            rms21: rms of the fitted 21cm signal
            dic: minimized dic value
            S: some covariance matrix
            epsilon: signal bias statistics (to ensure the confidence levels)
            D: normalized deviance (to ensure overfitting)
        """
        print('Extractor: Extracting the 21cm signal...', end='')    
        F_Fg = wgtBasisFg[:, 0:modesFg]
        F_21 = wgtBasis21[:, 0:modes21]
        
        exp21 = np.identity(len(self.nu))
        exp21 = np.tile(exp21, (self.nLST*self.ants, 1))   
        
        F_x21 = np.matmul(exp21, F_21)
        F = np.concatenate((F_Fg, F_x21), axis=-1)
        
        S = np.matmul(F.T, self.mulDiag(covmatInv, F))
        S = np.linalg.inv(S)
        
        Xi = np.matmul(S, np.matmul(F.T, self.mulDiag(covmatInv, mockObs)))
        XiFg = Xi[0:modesFg]
        Xi21 = Xi[modesFg:(modesFg + modes21)]
        
        reconsFg = np.matmul(F_Fg, XiFg)
        recons21 = np.matmul(F_x21, Xi21)
        reconsObs = np.matmul(F, Xi)
        
        delta = reconsObs - mockObs
        dic = np.matmul(delta.T, self.mulDiag(covmatInv, delta)) + 2*(modesFg + modes21)
        
        S_Fg_Fg = S[0:modesFg, 0:modesFg]
        S_21_21 = S[modesFg:(modesFg + modes21), modesFg:(modesFg + modes21)]
        
        channelCovFg = np.matmul(F_Fg, np.matmul(S_Fg_Fg, F_Fg.T))
        channelCov21 = np.matmul(F_21, np.matmul(S_21_21, F_21.T))

        sigma21 = np.sqrt(np.diag(channelCov21))
        sigmaFg = np.sqrt(np.diag(channelCovFg))

        trace21 = np.trace(channelCov21)
        tracefg = np.trace(channelCovFg)

        rms21 = np.sqrt(trace21/len(self.nu)) * 1000.0
        rmsFg = np.sqrt(tracefg/len(self.nu)) * 1000.0

        eps = []
        for i in range(len(self.nu)):
            eps.append(pow((recons21[i] - y21[i]), 2.0)/(np.diag(channelCov21))[i])

        epsilon = sum(eps)/len(self.nu)
        epsilon = np.sqrt(epsilon)
        
        reconsFg = reconsFg.reshape(reconsFg.shape[0],)
        recons21 = recons21.reshape(recons21.shape[0],)
        
        D = np.matmul(delta.T, self.mulDiag(covmatInv, delta))/\
            (len(self.nu) * self.nLST * self.ants - (modesFg + modes21))
        D = D[0][0]
        
        print('Done!' + u'\U0001f604')
        print('Extractor: RMS uncertainty = %.2f mK'%(rms21*epsilon[0]))
        print('Extractor: Signal Bias Statistic = %.2f'%epsilon[0])
        print('Extractor: Normalized Deviance = %.2f'%D)
        return reconsFg, recons21, sigmaFg, sigma21, channelCovFg, channelCov21,\
            rmsFg, rms21, dic, S, epsilon, D

    @staticmethod
    def mulDiag(A, B):
        return np.multiply(np.diag(A)[:, None], B)
