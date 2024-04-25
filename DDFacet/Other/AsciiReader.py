from astropy.io import ascii
from DDFacet.ToolsDir.rad2hmsdms import rad2hmsdms
import numpy as np
from DDFacet.Other import logger
log=logger.getLogger("ModAscii")

def readMultiFieldFile(FName):
    # Read the CSV file
    if FName.endswith(".txt"):
        return read_txt(FName)
    else:
        return read_csv(FName)

def read_txt(FName):
    
    with open(FName, 'r') as file:
        lines = file.readlines()[1:]
    data = ascii.read(lines, delimiter=' ', names=['ra', 'dec', "FOVarcsec"])
    
    return data

def read_csv(FName):
    data = ascii.read(FName, format="csv", fast_reader=False)
    ras=rad2hmsdms(data["RA"]*np.pi/180,Type="ra").replace(" ",":")
    decs=rad2hmsdms(data["DEC"]*np.pi/180,Type="dec").replace(" ",":")
    data["ra"]=ras
    data["dec"]=decs
    return data


def writeAscii(File=None):
    C=np.load(File)
    # ('Name','|S200'),
    # ('ra',np.float),('dec',np.float),('SumI',np.float),
    # ("Cluster",int),("SizeRad",np.float)

    OutFile="%s.MultiField.txt"%File
    # # ra	dec NPix
    log.print("Saving %s"%OutFile)
    with open(OutFile, 'w') as f:
        Header="# ra	dec  FOVarcsec"
        f.write(Header+'\n')
        
        
        for i in range(C.size):
            L=[]
            SRa=rad2hmsdms(C["ra"][i],Type="ra").replace(" ",":")
            L.append(SRa)
            SDec=rad2hmsdms(C["dec"][i]).replace(" ",":")
            L.append(SDec)
            FOV="%.2f"%(C["SizeRad"][i]*180/np.pi*3600)
            L.append(FOV)
            ss=" ".join(L)
            f.write(ss+'\n')        
