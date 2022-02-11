import os; import sys
from src.readset import Modset, Inputs
from src.basis import Basis
from src.noise import Noise
from src.infocrit import Dic
from src.extractor import Extractor
from tqdm.auto import tqdm

class PipelineStat(Modset, Inputs, Basis, Noise, Dic, Extractor):
    def __init__(self, nu, nLST, ant, path21TS, pathFgTS,
                 dT=6, modesFg=50, modes21=50, file='test.txt',
                 indexFg=0, index21=0):
        self.nu = nu
        self.nLST = nLST
        self.ant = ant
        self.path21TS = path21TS
        self.pathFgTS = pathFgTS
        self.dT = dT
        self.modesFg = modesFg
        self.modes21 = modes21
        self.file = file
        self.indFg = indexFg
        self.ind21 = index21
    
    def runPipeline(self):
        ''' Reading in the modelling sets '''
        models = Modset(nu=self.nu, nLST=self.nLST, ant=self.ant)
        m21 = models.get21modset(file=self.path21TS, nuMin=self.nu[0], nuMax=self.nu[-1])
        mFg = models.getcFgmodset(file=self.pathFgTS, nLST_tot=144)
        
        ''' Generating inputs from the modelling sets '''
        inputs = Inputs(nu=self.nu, nLST=self.nLST, ant=self.ant)
        y21, y_x21 = inputs.getExp21(modset=m21, ind=self.ind21)
        yFg = inputs.getFg(modset=mFg, ind=self.indFg)
        
        ''' Generating the noise and getting its covariance '''
        noise = Noise(nu=self.nu, nLST=self.nLST, ant=self.ant, power=y_x21+yFg,
                      deltaNu=self.nu[1] - self.nu[0], deltaT=self.dT)
        thermRealz = noise.noiseRealz()        
        cmat = noise.covmat()
        cmatInv = noise.covmatInv()
        
        ''' Getting the noise covariance weighted modelling sets '''
        wgt_m21 = noise.wgtTs(modset=m21.T, opt='21')
        wgt_mFg = noise.wgtTs(modset=mFg.T, opt='FG')
        
        ''' Generating the mock observation '''
        y = y_x21 + yFg + thermRealz
        
        ''' Weighted SVD for getting the optimal modes '''
        basis = Basis(nu=self.nu, nLST=self.nLST, ant=self.ant)
        b21 = basis.wgtSVDbasis(modset=wgt_m21, covmat=cmat, opt='21')
        bFg = basis.wgtSVDbasis(modset=wgt_mFg, covmat=cmat, opt='FG')
        
        ''' Minimizing information criterion for selecting the number of modes '''
        ic = Dic(nu=self.nu, nLST=self.nLST, ant=self.ant)
        ic.gridinfo(modesFg=self.modesFg, modes21=self.modes21, wgtBasis21=b21, wgtBasisFg=bFg,
                    covmatInv=cmatInv, mockObs=y, file=self.file)
        icmodesFg, icmodes21, _ = ic.searchMinima(file=self.file)
        
        ''' Finally extracting the signal! '''
        ext = Extractor(nu=self.nu, nLST=self.nLST, ant=self.ant)
        quants = ext.extract(modesFg=icmodesFg, modes21=icmodes21,
                             wgtBasisFg=bFg, wgtBasis21=b21,
                             covmatInv=cmatInv, mockObs=y, y21=y21)
        qDic = quants[8]
        qBias = quants[10]
        qNormD = quants[11]
        qRms = quants[7] * qBias[0]

        os.system('rm %s'%self.file)
        return icmodesFg, icmodes21, qDic[0][0], qBias[0], qNormD, qRms

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class StatQuants:
    def __init__(self, nu, nLST, ant, path21TS, pathFgTS, dT, modesFg, modes21, file):
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
                    pipe = PipelineStat(nu=self.nu, nLST=self.nLST, ant=self.ant, path21TS=self.path21TS,
                                    pathFgTS=self.pathFgTS, dT=self.dT, modesFg=self.modesFg,
                                    modes21=self.modes21, file=self.file,
                                    indexFg=indFg, index21=ind21)
                    quants = pipe.runPipeline()
                    for i in range(len(quants)):
                        fp.write(str(quants[i]))
                        fp.write('\t')
                    fp.write('\n')
            fp.close()    
