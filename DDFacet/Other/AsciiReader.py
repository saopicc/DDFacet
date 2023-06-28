from astropy.io import ascii

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
    return data,"txt"

def read_csv(FName):
    data = ascii.read(FName, format="csv", fast_reader=False)
    return data,"csv"
