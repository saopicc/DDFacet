from astropy.io import ascii

def readMultiFieldFile(FName):
    # Read the CSV file
    with open(FName, 'r') as file:
        lines = file.readlines()[1:]

    # Read the CSV file
    data = ascii.read(lines, delimiter=' ', names=['ra', 'dec', "NPix"])
    # # Get the columns for RA and Dec
    # ra_col = data['ra']
    # dec_col = data['dec']
    # NPix = data['NPix']  # Assuming 'dec' is the column name for Dec

    # Convert to SkyCoord
    return data
