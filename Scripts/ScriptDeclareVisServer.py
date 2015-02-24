import ClassMS
from progressbar import ProgressBar
import ClassVisServer
ProgressBar.silent=1


Fail=False

freq=-1
ChanFreq=[[-1]]
nbl=-1
na=-1


try:
    VS=ClassVisServer.ClassVisServer(GD,NameMS)
    MS=VS.MS
    # MS.ReadData(t0=t0,t1=t1)
    freq=MS.Freq_Mean
    ChanFreq=MS.ChanFreq.tolist()
    nbl=MS.nbl
    na=MS.na
except:
    Fail=True
