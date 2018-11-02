import numpy as np
from math import (sqrt, pow, inf)
from numba import jit, jitclass
from numba import float32

spec_Point3D = [
    ('__x', float32), 
    ('__y', float32),
    ('__z', float32),
]

@jitclass(spec_Point3D)
class Point3D():
    """
    Represents a point in 3D space.
    """

    def __init__(self, x, y, z):
        self.__x = x
        self.__y = y
        self.__z = z

    @property
    def x(self): return self.__x

    @property
    def y(self): return self.__y

    @property
    def z(self): return self.__z

    @x.setter
    def x(self, value): self.__x = value

    @y.setter
    def y(self, value): self.__y = value

    @z.setter
    def z(self, value): self.__z = value

    def set_values(self, x, y ,z):
        self.__x = x
        self.__y = y
        self.__z = z

    def to_string(self):
        return "({0},{1},{2})".format(self.__x, self.__y, self.__z)
    
@jit(nopython=True)
def compute_euclidean_distance_3D(point1, point2):
    """
    Computes and returns the euclidean distance for two points in 3D space.

    Args:
        point1: The first point
        point2: The second point

    Returns:
        The Euclidean distance of the two points in 3D space
    """

    return sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2) + pow(point1.z - point2.z, 2))


@jit(nopython=True)
def compute_hausdorff_distance_3D(grid1, grid2):
    """
    Computes and returns the Hausdorff distance for two 3D grids of MRC format.

    Args:
        grid1: The first grid of points in MRC format
        grid2: The second grid of points in MRC format

    Returns:
        The Hausdorff distance of the two 3D grids
    """

    directedDistance1 = compute_directed_hausdorff_distance_3D(grid1, grid2)
    directedDistance2 = compute_directed_hausdorff_distance_3D(grid2, grid1)

    if directedDistance1[0] > directedDistance2[0]:
        return (directedDistance1[0], directedDistance1[1], directedDistance1[2])
    else:
        return (directedDistance2[0], directedDistance2[1], directedDistance2[2])

@jit(nopython=True, fastmath=True)
def compute_directed_hausdorff_distance_3D(grid1, grid2):
    """
    Computes and returns the directed Hausdorff distance for two 3D grids of MRC format.

    Args:
        grid1: The first grid of points in MRC format (origin)
        grid2: The second grid of points in MRC format (destination)

    Returns:
        The directed Hausdorff distance of the two 3D grids
    """

    max_distance = 0
    currrent_distance = 0

    # Create point objects
    point1 = Point3D(-1, -1, -1)
    point2 = Point3D(-1, -1, -1)

    temp_point_min = Point3D(-1, -1, -1)

    point_min = Point3D(-1, -1, -1)
    point_max = Point3D(-1, -1, -1)
        
    # Iterate over all points in grid 1
    for grid1_index, grid1_value in np.ndenumerate(grid1):

        if (grid1_value == 0): continue

        point1.set_values(grid1_index[0], grid1_index[1], grid1_index[2])

        min_distance = inf
            
        # Iterate over all points in grid 2
        for grid2_index, grid2_value in np.ndenumerate(grid2):

            if (grid2_value == 0): continue

            point2.set_values(grid2_index[0], grid2_index[1], grid2_index[2])

            currrent_distance = compute_euclidean_distance_3D(point1, point2)

            if currrent_distance <= max_distance:
                min_distance = 0
                break

            if (min_distance > currrent_distance):
                min_distance = currrent_distance
                temp_point_min.set_values(point2.x, point2.y, point2.z)

        if max_distance < min_distance:
            max_distance = min_distance
            point_max.set_values(point1.x, point1.y, point1.z)
            point_min.set_values(temp_point_min.x, temp_point_min.y, temp_point_min.z)

    return (max_distance, point_min, point_max)