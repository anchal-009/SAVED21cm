import sys; sys.path.insert(1, './../')
from src.visuals import Visual
import tests.settings as set

visual = Visual(nu=set.nu, nLST=2, ant=set.ANT)

antNames = ['Log-Spiral', 'Dipole', 'Sinuous', 'Dipole + Log-Spiral']
fnames = ['./StatsOutput/Stats_lst-2_ant-logspiral.txt',
          './StatsOutput/Stats_lst-2_ant-dipole.txt',
          './StatsOutput/Stats_lst-2_ant-sinuous.txt',
          './StatsOutput/Stats_lst-2_ant-dipole-logspiral.txt']

visual.plotBiasCDF(antNames=antNames, fnames=fnames, save=True)
visual.plotNormD(antNames=antNames, fnames=fnames, save=True, bins=15)
visual.plotRmsCDF(antNames=antNames, fnames=fnames, save=True)
