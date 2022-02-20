import os; import sys
from src.runpipe import Pipeline
from tqdm.auto import tqdm

class HiddenPrints:
    """To not print statements while calculating statistics over an ensemble.
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class StatQuants:
    def __init__(self, nu, nLST, ant, path21TS, pathFgTS, dT, modesFg, modes21, quantity, file):
        """Initialize for generating some statistical measures of the fitting.

        Args:
            nu (array): Frequency range
            nLST (int): Number of time bins to fit
            ant (lsit): List of antenna designs
            path21TS (string): Path to 21cm modelling set
            pathFgTS (string): Path to foregrounds modelling set
            dT (int): Integration time in hours
            modesFg (int): Total number of FG modes
            modes21 (int): Total number of 21 modes
            quantity (string): Quantity to minimize\
                               'DIC' for Deviance Information Criterion,\
                               'BIC' for Bayesian Information Criterion
            file (string): Filename to store the gridded IC
        """
        self.nu = nu
        self.nLST = nLST
        self.ant = ant
        self.path21TS = path21TS
        self.pathFgTS = pathFgTS
        self.dT = dT
        self.modesFg = modesFg
        self.modes21 = modes21
        self.quantity = quantity
        self.file = file
        
    def getStats(self, fname, iList21, iListFg):
        """Estimates the statistical measures over an ensemble of different foregrounds and 21cm
        signals and store that information in a file.

        Args:
            fname (string): Filename to store the statistical information
            iList21 (array): List of 21cm index to use as the input (from the modelling set)
            iListFg (array): List of foreground index to use as the input (from the modelling set)
        """
        print('--------------- Estimating statistical measures ---------------\n', flush=True)
        if not os.path.exists('StatsOutput/'):
            os.mkdir('StatsOutput/')
        with HiddenPrints():
            fp = open('StatsOutput/%s'%fname, 'w')
            ''' Write the settings for the analysis '''
            fp.write('# Frequency (min, max, step): (%s, %s, %s)\n'
                     %(self.nu[0], self.nu[-1], self.nu[1] - self.nu[0])); 
            fp.write('# Number of LST bins: %s\n'%self.nLST); 
            fp.write('# Antenna Design: %s\n'%self.ant); 
            fp.write('# 21 training set: %s\n'%self.path21TS); 
            fp.write('# Fg training set: %s\n'%self.pathFgTS); 
            fp.write('# Integration Time: %s\n'%self.dT); 
            fp.write('# Total Fg modes: %s\n'%self.modesFg); 
            fp.write('# Total 21 modes: %s\n'%self.modes21); 
            fp.write('\n# mFg\tm21\ticVal\tbias\tnormD\trms\n')
            
            for i21 in tqdm(iList21, desc='i21', colour='#36FAC5'):
                for iFg in tqdm(iListFg, desc='iFg', leave=False, colour='#FCB074'):
                    ind21 = int(i21); indFg = int(iFg)
                    pipe = Pipeline(nu=self.nu, nLST=self.nLST, ant=self.ant, path21TS=self.path21TS,
                                        pathFgTS=self.pathFgTS, dT=self.dT, modesFg=self.modesFg,
                                        modes21=self.modes21, quantity=self.quantity, file=self.file,
                                        indexFg=indFg, index21=ind21)
                    quants = pipe.runPipeline()
                    for i in range(len(quants)):
                        fp.write(str(quants[i]))
                        fp.write('\t')
                    fp.write('\n')
            fp.close()    
