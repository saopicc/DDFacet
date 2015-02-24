
import ClassMS
import ClassSM
import numpy as np
from progressbar import ProgressBar
ProgressBar.silent=1

Fail=False

freq=-1
ChanFreq=[[-1]]
nbl=-1
na=-1

try:
    MS=ClassMS.ClassMS(NameMS,Col=ColName,SelectSPW=[0],DoReadData=False)
    # MS.ReadData(t0=t0,t1=t1)
    freq=MS.Freq_Mean
    ChanFreq=MS.ChanFreq.tolist()
    nbl=MS.nbl
    na=MS.na
except:
    Fail=True
