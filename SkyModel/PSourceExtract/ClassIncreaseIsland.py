
import numpy as np

class ClassIncreaseIsland():
    def __init__(self):
        pass

    def IncreaseIsland(self,ListPix,dx=2):
        nx=dx*2+1
        xg,yg=np.mgrid[-dx:dx:1j*nx,-dx:dx:1j*nx]
        xg=np.int64(xg.flatten())
        yg=np.int64(yg.flatten())
        OutListPix=[]
        for Pix in ListPix:
            x0,y0=Pix
            x1=xg+x0
            y1=yg+y0
            for iPixAdd in range(xg.size):
                OutListPix.append((x1[iPixAdd],y1[iPixAdd]))
        OutListPix=list(set(OutListPix))
        OutListPix2=[[x,y] for x,y in OutListPix]
        return OutListPix2

