import numpy as np
from scipy import ndimage
import sys
import os

# ---- User-defined DS9 colormap (blue → cyan → green → yellow → red) ----
DS9_CMAP = [
    (0.00, (0, 0, 255)),      # blue
    (0.25, (0, 255, 255)),    # cyan
    (0.50, (0, 255, 0)),      # green
    (0.75, (255, 255, 0)),    # yellow
    (1.00, (255, 0, 0)),      # red
]


def value_to_rgb(val, cmap=DS9_CMAP):
    """
    Map a scalar in [0,1] to RGB using piecewise linear interpolation.
    """
    val = float(np.clip(val, 0.0, 1.0))

    for i in range(len(cmap) - 1):
        v0, c0 = cmap[i]
        v1, c1 = cmap[i + 1]
        if v0 <= val <= v1:
            t = (val - v0) / (v1 - v0)
            r = int(c0[0] + t * (c1[0] - c0[0]))
            g = int(c0[1] + t * (c1[1] - c0[1]))
            b = int(c0[2] + t * (c1[2] - c0[2]))
            return r, g, b

    return cmap[-1][1]


def island_to_polygons(filename,Lpixels, nxny=None):
    if os.path.exists(filename):
        os.system("rm %s"%filename)
    nx,ny=nxny
    for ipixels,pixels in enumerate(Lpixels):
        polygons=island_to_polygons_single(pixels,nx,ny)#, contour_value)
        write_ds9_region(filename, polygons, one_indexed=True,
                         header=(ipixels==0),label="Isl %i"%ipixels)
    



import matplotlib.pyplot as plt


def island_to_polygons_single(pixels,nxin,nyin):
    """
    Convert a list of (x, y) island pixels into a single ordered polygon
    using matplotlib contour (no plotting, no private API).

    Returns:
        Nx2 numpy array of (x, y) vertices.
    """
    pixels = np.asarray(pixels)
    pixels[:,0]=nxin-pixels[:,0]
    
    # Bounding box
    min_xy = pixels.min(axis=0)
    max_xy = pixels.max(axis=0)

    ny = max_xy[1] - min_xy[1] + 3
    nx = max_xy[0] - min_xy[0] + 3

    mask = np.zeros((ny, nx), dtype=float)

    for x, y in pixels:
        mask[y - min_xy[1] + 1, x - min_xy[0] + 1] = 1.0

    # Create contour without displaying anything
    fig = plt.figure()
    cs = plt.contour(mask, levels=[0.5])
    plt.close(fig)

    if not cs.allsegs or not cs.allsegs[0]:
        return None

    polygon = cs.allsegs[0][0]

    # Shift back to original coordinates
    polygon[:, 0] += min_xy[0] - 1
    polygon[:, 1] += min_xy[1] - 1

    return polygon

def write_ds9_region(filename, polygons, one_indexed=True,header=True,label=None):
    """
    Write DS9 region file with per-polygon RGB colors.
    """
    with open(filename, "a") as f:
        if header:
            f.write("# Region file format: DS9 version 4.1\n")
            f.write("""global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n""")
            f.write("image\n")
        
        #coords = poly["vertices"].copy()
        #r, g, b = poly["color"]
        #if one_indexed:
        #    coords += 1
            
        flat = ",".join(f"{x:.3f},{y:.3f}" for x, y in polygons)
        #f.write(f"polygon({flat}) # color={r} {g} {b}\n")
        f.write(f"polygon({flat}) \n")

        if label is not None:
            xc,yc=np.mean(polygons,axis=0)
            f.write("point(%.1f,%.1f) # point=cross text={%s} \n"%(xc,yc,label))
        

# Example
def test():
    pixels = [
        (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2),
        (2, 3)
    ]
    
    S=np.load("IslandsMajor1.npz",allow_pickle=1)
    Lxy=S["ListIslandsNotIncreased"]
    nxny=S["nxny"]

    polygons = island_to_polygons(filename,Lxy,nxny=nxny)
    #write_ds9_region("island.reg", polygons)
