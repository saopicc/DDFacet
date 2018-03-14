import numpy as np

def findrms(mIn,maskSup=1e-7):
    m=mIn[np.abs(mIn)>maskSup]
    rmsold=np.std(m)
    diff=1e-1
    cut=3.
    bins=np.arange(np.min(m),np.max(m),(np.max(m)-np.min(m))/30.)
    med=np.median(m)
    for i in range(10):
        ind=np.where(np.abs(m-med)<rmsold*cut)[0]
        rms=np.std(m[ind])
        if np.abs((rms-rmsold)/rmsold)<diff: break
        rmsold=rms
    return rms
