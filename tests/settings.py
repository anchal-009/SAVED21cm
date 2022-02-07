import numpy as np

nu = np.linspace(50, 200, 151)
path21TS = './../../../Projects/REACH-svd/Data/TS-21/lfcal_training_set_8-2020.hdf5'
pathFgTS = './../../../Projects/REACH-svd/Data/TS-FG/Foreground-training-set-haslam-beta-const-'

LST = 2
ANT = ['dipole', 'logspiral']
ANTS = len(ANT)
dNU = 1. 
dT = 6
nModesFg = 50
nModes21 = 80
VISUALS = True
SAVE = True
FNAME = './dic_search.txt'

print('------------------ Settings for the pipeline ------------------\n')
print('Frequencies: Between (%d - %d) MHz with step size of %d MHz.'
      %(nu[0], nu[-1], nu[1] - nu[0]))
print('Number of LST bins:', LST)
print('Antenna Design:', ANT)
print('Integration Time:', dT)
print('Total number of foreground modes:', nModesFg)
print('Total number of 21cm modes:', nModes21)
print('Filename to store the info criteria:', FNAME, '\n')
