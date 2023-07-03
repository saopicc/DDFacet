from astropy.io import ascii
from DDFacet.ToolsDir.rad2hmsdms import rad2hmsdms

def readMultiFieldFile(FName):
    # Read the CSV file
    if FName.endswith(".txt"):
        return read_txt(FName)
    else:
        return read_csv(FName)

def read_txt(FName):
    with open(FName, 'r') as file:
        lines = file.readlines()[1:]
    data = ascii.read(lines, delimiter=' ', names=['ra', 'dec', "NPix"])
    
    return data

def read_csv(FName):
    data = ascii.read(FName, format="csv", fast_reader=False)
    ras=rad2hmsdms(data["RA"]*np.pi/180,Type="ra").replace(" ",":")
    decs=rad2hmsdms(data["DEC"]*np.pi/180,Type="dec").replace(" ",":")
    data["ra"]=ras
    data["dec"]=decs
    return data
