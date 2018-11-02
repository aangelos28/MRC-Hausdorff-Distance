import numpy as np
from math import (sqrt, pow, inf)
from numba import jit, jitclass
from numba import float32

spec_Point2D = [
    ('__x', float32), 
    ('__y', float32),
]

@jitclass(spec_Point2D)
class Point2D():
    """
    Represents a point in 2D space.
    """

    def __init__(self, x, y, z):
        self.__x = x
        self.__y = y

    @property
    def x(self): return self.__x

    @property
    def y(self): return self.__y

    def set_values(self, x, y):
        self.__x = x
        self.__y = y

    def to_string(self):
        return "({1},{2})".format(self.__x, self.__y)

def compute_euclidean_distance_2D(point1, point2):
    """
    Computes and returns the euclidean distance for two points in 2D space.

    Args:
        point1: The first point
        point2: The second point

    Returns:
        The Euclidean distance of the two points in 2D space
    """

    return sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2))

@jit(nopython=True)
def compute_hausdorff_distance_2D(plane1, plane2):
    """
    Computes and returns the Hausdorff distance for two 2D planes of MRC format.

    Args:
        plane1: The first plane of points in MRC format
        plane2: The second plane of points in MRC format

    Returns:
        The Hausdorff distance of the two 2D planes
    """

    directedDistance1 = compute_directed_hausdorff_distance_2D(plane1, plane2)
    directedDistance2 = compute_directed_hausdorff_distance_2D(plane2, plane1)

    if directedDistance1[0] > directedDistance2[0]:
        return (directedDistance1[0], directedDistance1[1], directedDistance1[2])
    else:
        return (directedDistance2[0], directedDistance2[1], directedDistance2[2])

@jit(nopython=True, fastmath=True)
def compute_directed_hausdorff_distance_2D(plane1, plane2):
    """
    Computes and returns the directed Hausdorff distance for two 2D planes of MRC format.

    Args:
        plane1: The first plane of points in MRC format (origin)
        plane2: The second plane of points in MRC format (destination)

    Returns:
        The directed Hausdorff distance of the two 2D planes
    """

    max_distance = 0
    currrent_distance = 0

    # Create point objects
    point1 = Point2D(-1, -1, -1)
    point2 = Point2D(-1, -1, -1)

    temp_point_min = Point2D(-1, -1, -1)

    point_min = Point2D(-1, -1, -1)
    point_max = Point2D(-1, -1, -1)
        
    # Iterate over all points in grid 1
    for plane1_index, plane1_value in np.ndenumerate(plane1):

        if (plane1_value == 0): continue

        point1.set_values(plane1_index[0], plane1_index[1])

        min_distance = inf
            
        # Iterate over all points in grid 2
        for plane2_index, plane2_value in np.ndenumerate(plane2):

            if (plane2_value == 0): continue

            point2.set_values(plane2_index[0], plane2_index[1])

            currrent_distance = compute_euclidean_distance_2D(point1, point2)

            if currrent_distance <= max_distance:
                min_distance = 0
                break

            if (min_distance > currrent_distance):
                min_distance = currrent_distance
                temp_point_min.set_values(point2.x, point2.y)

        if max_distance < min_distance:
            max_distance = min_distance
            point_max.set_values(point1.x, point1.y)
            point_min.set_values(temp_point_min.x, temp_point_min.y)

    return (max_distance, point_min, point_max)