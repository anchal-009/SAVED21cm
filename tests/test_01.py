import sys; sys.path.insert(1, './../')
from src.readset import Modset, Inputs
from src.noise import Noise
from src.basis import Basis
from src.visuals import Visual
from src.infocrit import Dic
from src.extractor import Extractor
import settings as set

''' Reading in the modelling sets '''
models = Modset(nu=set.nu)
m21 = models.get21modset(file=set.path21TS, nuMin=50, nuMax=200)
mFg = models.getcFgmodset(file=set.pathFgTS, nLST=set.LST,
                          nLST_tot=144, ant=set.ANT)

''' Generating inputs from the modelling sets '''
inputs = Inputs(nu=set.nu, nLST=set.LST, ants=set.ANTS)
y21, y_x21 = inputs.getExp21(modset=m21, ind=1000)
yFg = inputs.getFg(modset=mFg, ind=0)

''' Generating the noise and getting its covariance '''
noise = Noise(nu=set.nu, nLST=set.LST, ants=set.ANTS, power=y_x21 + yFg,
              deltaNu=set.dNU, deltaT=set.dT)
realz = noise.noiseRealz()
cmat = noise.covmat()
cmatInv = noise.covmatInv()

''' Getting the noise covariance weighted modelling sets '''
wgt_m21 = noise.wgtTs(modset=m21.T, opt='21')
wgt_mFg = noise.wgtTs(modset=mFg.T, opt='FG')

''' Generating the mock observation '''
y = y_x21 + yFg + realz

''' Weighted SVD for getting the optimal modes '''
basis = Basis(nu=set.nu)
b21 = basis.wgtSVDbasis(modset=wgt_m21, covmat=cmat,
                        nLST=set.LST, ants=set.ANTS, opt='21')
bFg = basis.wgtSVDbasis(modset=wgt_mFg, covmat=cmat, opt='FG')

''' Minimizing information criterion for selecting the number of modes '''
d = Dic(nu=set.nu, nLST=set.LST, ants=set.ANTS)
d.gridinfo(modesFg=set.nModesFg, modes21=set.nModes21,
           wgtBasisFg=bFg, wgtBasis21=b21,
           covmatInv=cmatInv, mockObs=y, file=set.FNAME)
modesFg, modes21, dic = d.searchMinima(file=set.FNAME)

''' Finally extracting the signal! '''
ext = Extractor(nu=set.nu, nLST=set.LST, ants=set.ANTS)
_, e21, _, s21, *_ = ext.extract(modesFg=modesFg, modes21=modes21,
                                 wgtBasisFg=bFg, wgtBasis21=b21,
                                 covmatInv=cmatInv, mockObs=y, y21=y21)

''' Visualizing everything! '''
if set.VISUALS:
    vis = Visual(nu=set.nu, nLST=set.LST, ant=set.ANT, save=set.SAVE)
    vis.plotModset(set=m21, opt='21', n_curves=1000)
    vis.plotModset(set=mFg, opt='FG', n_curves=100)
    vis.plotMockObs(y21=y21, yFg=yFg, noise=realz)
    vis.plotBasis(basis=b21, opt='21')
    vis.plotBasis(basis=bFg, opt='FG')
    vis.plotInfoGrid(file=set.FNAME, modesFg=set.nModesFg, modes21=set.nModes21)
    vis.plotExtSignal(y21=y21, recons21=e21, sigma21=s21)
