import os
from src.readset import Modset, Inputs
from src.basis import Basis
from src.noise import Noise
from src.infocrit import Dic
from src.extractor import Extractor
from src.visuals import Visual

class Pipeline(Modset, Inputs, Basis, Noise, Dic, Extractor):
    def __init__(self, nu, nLST, ant, path21TS, pathFgTS,
                 dT=6, modesFg=50, modes21=50, file='test.txt',
                 indexFg=0, index21=0):
        self.nu = nu
        self.nLST = nLST
        self.ant = ant
        self.ants = len(self.ant)
        self.path21TS = path21TS
        self.pathFgTS = pathFgTS
        self.dT = dT
        self.modesFg = modesFg
        self.modes21 = modes21
        self.file = file
        self.indFg = indexFg
        self.ind21 = index21
    
    def runPipeline(self):
        print('-------------------- Running the pipeline ---------------------\n')
        ''' Reading in the modelling sets '''
        models = Modset(nu=self.nu)
        m21 = models.get21modset(file=self.path21TS, nuMin=self.nu[0], nuMax=self.nu[-1])
        mFg = models.getcFgmodset(file=self.pathFgTS, nLST=self.nLST, nLST_tot=144, ant=self.ant)
        
        ''' Generating inputs from the modelling sets '''
        inputs = Inputs(nu=self.nu, nLST=self.nLST, ants=self.ants)
        y21, y_x21 = inputs.getExp21(modset=m21, ind=self.ind21)
        yFg = inputs.getFg(modset=mFg, ind=self.indFg)
        
        ''' Generating the noise and getting its covariance '''
        noise = Noise(nu=self.nu, nLST=self.nLST, ants=self.ants, power=y_x21+yFg,
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
        basis = Basis(nu=self.nu)
        b21 = basis.wgtSVDbasis(modset=wgt_m21, covmat=cmat,nLST=self.nLST, ants=self.ants, opt='21')
        bFg = basis.wgtSVDbasis(modset=wgt_mFg, covmat=cmat, opt='FG')
        
        ''' Minimizing information criterion for selecting the number of modes '''
        ic = Dic(nu=self.nu, nLST=self.nLST, ants=self.ants)
        ic.gridinfo(modesFg=self.modesFg, modes21=self.modes21, wgtBasis21=b21, wgtBasisFg=bFg,
                    covmatInv=cmatInv, mockObs=y, file=self.file)
        icmodesFg, icmodes21, _ = ic.searchMinima(file=self.file)
        
        ''' Finally extracting the signal! '''
        ext = Extractor(nu=self.nu, nLST=self.nLST, ants=self.ants)
        _, e21, _, s21, *_ = ext.extract(modesFg=icmodesFg, modes21=icmodes21,
                                         wgtBasisFg=bFg, wgtBasis21=b21,
                                         covmatInv=cmatInv, mockObs=y, y21=y21)

        os.system('rm %s'%self.file)

        ''' Visuals '''
        vis = Visual(nu=self.nu, nLST=self.nLST, ant=self.ant)
        vis.plotExtSignal(y21=y21, recons21=e21, sigma21=s21)
        