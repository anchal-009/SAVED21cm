import numpy as np

nu = np.linspace(50, 200, 151)
path21TS = './../../../Projects/REACH-svd/Data/TS-21/\
lfcal_training_set_8-2020.hdf5'
pathFgTS = './../../../Projects/REACH-svd/Data/TS-FG/\
Foreground-training-set-haslam-beta-const-'

LST = 2
ANT = ['logspiral'] # , 'dipole', 'sinuous']
ANTS = len(ANT)

dNU = 1. 
dT = 4

nModesFg = 80
nModes21 = 80

VISUALS = True
SAVE = True
FNAME = './dic_search.txt'
