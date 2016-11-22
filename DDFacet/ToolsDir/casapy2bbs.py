import numpy as np
import rad2hmsdms
from pyrap.images import image


def Print(ImageName="testImage.fits"):

    im = image(ImageName)
    d = im.getdata()
    _, _, lx, ly = np.where(d != 0)

    for x, y in zip(lx, ly):
        _, _, dec, ra = im.toworld([0, 0, x, y])
        print rad2hmsdms.rad2hmsdms(ra, Type="ra").replace(" ", ":"), ", ", rad2hmsdms.rad2hmsdms(dec, Type="dec").replace(" ", ".")
