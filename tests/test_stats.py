import sys; sys.path.insert(1, './../')
from src.getstats import StatQuants
import settings as set
import numpy as np

statquants = StatQuants(nu=set.nu, nLST=set.LST, ant=set.ANT,
                        path21TS=set.path21TS, pathFgTS=set.pathFgTS,
                        dT=set.dT, modesFg=set.nModesFg, modes21=set.nModes21,
                        file=set.FNAME)

statquants.getStats(fname='Stats_lst-%d_ant-%s.txt'%(set.LST, '-'.join(set.ANT)),
                    iList21=np.linspace(0, 20_000, 50),
                    iListFg=np.linspace(0, 100, 5))
