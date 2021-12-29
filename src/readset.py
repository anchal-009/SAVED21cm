import numpy as np
import h5py

class Modset:
    def __init__(self, nu):
        """This class contains methods for reading in the modelling sets 
        (foreground and 21cm signal) from our standard hdf5 files.

        Args:
            nu (array): Frequency range
        """
        self.nu = nu

    def get21modset(self, file, nuMin, nuMax):
        """This method returns the 21cm signal modelling set.

        Args:
            file (string): path to file
            nuMin (float): Minimum frequency accessible
            nuMax (float): Maximum frequency accessible

        Returns:
            array: 21cm modelling set of shape (n_curves, n_nu)
        """
        print('Modelling set: Reading 21 modelling set...', end='', flush=True)
        hf = h5py.File(file, 'r')
        nu = hf.attrs['frequencies']
        ind = np.where(np.logical_and(nu >= nuMin, nu <= nuMax))
        nu = nu[ind]
        nu = nu[0::5]
        mod21 = hf.get('signal_sample')
        mod21 = np.array(mod21)
        # Restricting the 21cm TS between nu_min and nu_max
        mod21 = ((mod21.T)[ind]).T
        mod21 = mod21.T[0::5].T/1000.
        print('Done!')
        return mod21

    def getFgmodset(self, file, nLST, nLST_tot):
        """This method returns the foreground modelling set of a specific antenna.

        Args:
            file (string): path to file
            nLST (int): Number of time bins to use in the analysis
            nLST_tot (int): Total number of time bins in the set

        Returns:
            array: Foreground modelling set of shape (n_curves, n_nu*n_t)
        """
        with h5py.File(file, 'r') as hdf:
            modFgRaw = hdf.get('training-set')
            modFgRaw = np.array(modFgRaw)
            n_samp = modFgRaw.shape[0]
            modFgRaw = modFgRaw[:, 0*len(self.nu):nLST_tot*len(self.nu)]
            modFgRaw = modFgRaw.flatten()
            modFgRaw = modFgRaw.reshape(n_samp*nLST*int(nLST_tot/nLST), len(self.nu))
            modFg = np.zeros(shape=(n_samp*nLST, len(self.nu)))
            j = 0
            for i in range(n_samp*nLST):
                modFg[i, :] = np.sum(modFgRaw[j:j+int(nLST_tot/nLST), :],
                                     axis=0)/int(nLST_tot/nLST)
                j = j + int(nLST_tot/nLST)
            modFg = modFg.reshape(n_samp, nLST*len(self.nu))
        return modFg
    
    def getcFgmodset(self, file, nLST, nLST_tot, ant):
        """This method concatenates the foreground modelling set for multiple antennas.

        Args:
            file (string): path to file
            nLST (int): Number of time bins to use in the analysis
            nLST_tot (int): Total number of time bins in the set
            ant (list): List of antenna designs

        Returns:
            array: Concatenated foreground modelling set of shape
                   (n_curves, n_nu*n_t*n_ant)
        """
        print('Modelling set: Reading FG modelling set...', end='', flush=True)
        mFg = np.concatenate(([Modset.getFgmodset(self, file='%s'%file+'%s'%ant[i]+'.h5',
                                                   nLST=nLST, nLST_tot=nLST_tot)
                               for i in range(len(ant))]), axis=-1)
        print('Done!')
        return mFg

class Inputs:
    def __init__(self, nu, nLST, ants):
        """This class contains methods for getting the inputs (foreground and 21cm)
        from their training set.

        Args:
            nu (array): Frequency range
            nLST (int): Number of time bins to use in the analysis
            ants (int): Number of antenna designs
        """
        self.nu = nu
        self.nLST = nLST
        self.ants = ants
    
    def getExp21(self, modset, ind):
        """This method returns one 21cm signal profile from the modelling set.

        Args:
            modset (array): 21cm modelling set
            ind (int): Index to randomly pick one sample

        Returns:
            array: Input 21cm signal profile, and the expanded version
        """
        y21 = modset[ind].reshape(len(self.nu), 1)
        exp21 = np.identity(len(self.nu))
        exp21 = np.tile(exp21, (self.nLST*self.ants, 1))
        y_x21 = np.matmul(exp21, y21)
        return y21, y_x21
    
    def getFg(self, modset, ind):
        """This method returns one foreground spectra from the modelling set.

        Args:
            modset (array): Foreground modelling set
            ind (int): Index to randomly pick one sample

        Returns:
            array: Input foregrounds
        """
        yFg = modset[ind].reshape(len(self.nu)*self.nLST*self.ants, 1)
        return yFg
