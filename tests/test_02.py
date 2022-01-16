import sys; sys.path.insert(1, './../')
from src.runpipe import Pipeline
import settings as set

pipe = Pipeline(nu=set.nu, nLST=set.LST, ant=set.ANT, path21TS=set.path21TS,
                pathFgTS=set.pathFgTS, indexFg=50, index21=1000)
pipe.runPipeline()
