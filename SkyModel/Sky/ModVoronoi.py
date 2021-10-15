from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.spatial import Voronoi
import SkyModel.Tools.PolygonTools as PT


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue


        # reconstruct a non-finite region
        if p1 not in list(all_ridges.keys()):
            new_regions.append(vertices)
            continue
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            
            V=vor.vertices[v2]
            R=np.sqrt(np.sum(V**2))
            
            # while R>0.1:
            #     V*=0.9
            #     R=np.sqrt(np.sum(V**2))
            #     #print R
        
            vor.vertices[v2][:]=V[:]

            ThisRad=radius
            far_point = vor.vertices[v2] + direction * radius
            R=np.sqrt(np.sum(far_point**2))
            ThisRad=R

            # while R>1:
            #     ThisRad*=0.9
            #     far_point = vor.vertices[v2] + direction * ThisRad
            #     R=np.sqrt(np.sum(far_point**2))
            #     print "=============="
            #     print R,np.sqrt(np.sum(vor.vertices[v2]**2))
            #     print vor.vertices[v2]
            #     print direction
            #     print ThisRad
            #     #if R>1000:
            #     #    stop
            
            # RadiusTot=.3
            # Poly=np.array([[-RadiusTot,-RadiusTot],
            #                [+RadiusTot,-RadiusTot],
            #                [+RadiusTot,+RadiusTot],
            #                [-RadiusTot,+RadiusTot]])*1
            # stop
            # PT.CutLineInside(Poly,Line)


            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    regions, vertices=new_regions, np.asarray(new_vertices)

    return regions, vertices


def test():
    # make up data points


    np.random.seed(1234)
    
    R=3*np.pi/180
    points = np.random.rand(15, 2)*R

    x,y=np.mgrid[-R:R:11*1j,-R:R:11*1j]
    points=np.array([x.ravel(),y.ravel()]).T.reshape((x.size,2))

    
    # compute Voronoi tesselation
    vor = Voronoi(points)
    
    # plot
    regions, vertices = voronoi_finite_polygons_2d(vor)
    print("--")
    print(regions)
    print("--")
    print(vertices)
    Plot(points,regions, vertices)
    # colorize

def Plot(points,regions, vertices):
    import matplotlib.pyplot as pylab
    pylab.clf()
    for region in regions:
        polygon = vertices[region]
        pylab.fill(*list(zip(*polygon)), alpha=0.4)

    
    pylab.plot(points[:,0], points[:,1], 'ko')
    #plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    #plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
    
    pylab.draw()
    pylab.show(block=False)

if __name__=="__main__":
    test()
