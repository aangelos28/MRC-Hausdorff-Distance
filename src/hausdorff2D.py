import numpy as np
import mrcfile
from math import (sqrt, pow, inf)
from concurrent.futures import ThreadPoolExecutor
from numba import jit

def point2D_to_mrc_file(output_mrc_filename, origin_plane, point):
    """
    Outputs the passed point as an MRC file based on its original MRC file.
    The point will have double the density of the maximum density of the original MRC.

    Args:
        output_mrc_filename: Path of the output MRC file
        origin_plane: The plane the point originated from
        point: The point to output

    Returns:
        Handle to the new MRC file.
    """

    # Create a zero array with the same shape as the original
    new_plane = np.copy(origin_plane)

    # Determine largest element
    max_element = np.amax(new_plane)

    # Set the element identified by the point to twice the value of the max_element so it can be distinguished
    new_plane[int(point[0]), int(point[1])] = max_element * 2

    return mrcfile.new(name=output_mrc_filename, data=new_plane, overwrite=True)
    
#################################################
# EARLYBREAK        
#################################################

@jit(nopython=True, cache=True)
def euclidean_distance_2D(point1, point2):
    """
    Computes and returns the euclidean distance for two points in the 2D plane.

    Args:
        point1: The first point
        point2: The second point

    Returns:
        The Euclidean distance of the two points in the 2D plane
    """

    return sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2))

def hausdorff_distance_2D(plane1, plane2):
    """
    Computes and returns the Hausdorff distance for two 2D planes of MRC format
    using the Earlybreak algorithm.

    Args:
        plane1: The first plane of points in MRC format
        plane2: The second plane of points in MRC format

    Returns:
        The Hausdorff distance of the two 2D planes
    """

    executor = ThreadPoolExecutor(max_workers=2)

    future1 = executor.submit(directed_hausdorff_distance_2D, plane1, plane2)
    future2 = executor.submit(directed_hausdorff_distance_2D, plane1, plane2)

    directedDistance1 = future1.result()
    directedDistance2 = future2.result()

    executor.shutdown()

    if directedDistance1[0] > directedDistance2[0]:
        return directedDistance1
    else:
        return directedDistance2

@jit(nopython=True, nogil=True, cache=True)
def directed_hausdorff_distance_2D(plane1, plane2):
    """
    Computes and returns the directed Hausdorff distance for two 2D planes of MRC format
    using the Earlybreak algorithm.

    Args:
        plane1: The first grid of points in MRC format (origin)
        plane2: The second grid of points in MRC format (destination)

    Returns:
        The directed Hausdorff distance of the two 2D planes
    """

    max_distance = 0
    currrent_distance = 0

    # Create point objects
    point1 = np.empty(2, np.intc)
    point2 = np.empty(2, np.intc)

    temp_point_min = np.empty(2, np.intc)

    point_min = np.zeros(2, np.intc)
    point_max = np.zeros(2, np.intc)
        
    # Iterate over all points in grid 1
    for plane1_index, plane1_value in np.ndenumerate(plane1):

        if (plane1_value == 0): continue

        point1[0] = plane1_index[0]
        point1[1] = plane1_index[1]

        min_distance = inf
            
        # Iterate over all points in grid 2
        for plane2_index, plane2_value in np.ndenumerate(plane2):

            if (plane2_value == 0): continue

            point2[0] = plane2_index[0]
            point2[1] = plane2_index[1]

            currrent_distance = euclidean_distance_2D(point1, point2)

            if currrent_distance <= max_distance:
                min_distance = 0
                break

            if (min_distance > currrent_distance):
                min_distance = currrent_distance

                temp_point_min[0] = point2[0]
                temp_point_min[1] = point2[1]

        if max_distance < min_distance:
            max_distance = min_distance

            point_max[0] = point1[0]
            point_max[1] = point1[1]

            point_min[0] = temp_point_min[0]
            point_min[1] = temp_point_min[1]

    return (max_distance, (point_min, plane2), (point_max, plane1))