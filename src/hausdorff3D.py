import numpy as np
import mrcfile
from math import (sqrt, pow, inf)
from numba import jit, jitclass
from numba import int32
from morton import (morton3D, extract_morton_coords_int)

spec_Point3D = [
    ('__x', int32), 
    ('__y', int32),
    ('__z', int32),
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
        return "({0}, {1}, {2})".format(self.__x, self.__y, self.__z)

def point3D_to_mrc_file(output_mrc_filename, origin_grid, point):
    """
    Outputs the passed point as an MRC file based on its original MRC file.
    The point will have double the density of the maximum density of the original MRC.

    Args:
        output_mrc_filename: Path of the output MRC file
        origin_grid: The grid the point originated from
        point: The point to output

    Returns:
        Handle to new MRC file.
    """

    # Create a zero array with the same shape as the original
    new_grid = np.copy(origin_grid)

    # Determine largest element
    max_element = np.amax(new_grid)

    # Set the element identified by the point to twice the value of the max_element so it can be distinguished
    new_grid[int(point.x), int(point.y), int(point.z)] = max_element * 2

    return mrcfile.new(name=output_mrc_filename, data=new_grid, overwrite=True)
    
#################################################
# EARLYBREAK        
#################################################

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
        return directedDistance1
    else:
        return directedDistance2

@jit(nopython=True)
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

    return (max_distance, (point_min, grid2), (point_max, grid1))

#################################################
# ZHD        
#################################################

def compute_euclidean_distance_3D_no_jit(point1, point2):
    """
    Computes and returns the euclidean distance for two points in 3D space.

    Args:
        point1: The first point
       point2: The second point

    Returns:
        The Euclidean distance of the two points in 3D space
    """

    return sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2) + pow(point1.z - point2.z, 2))

def mrc_z_order(k, grid):
    """
    Creates a 1-D z-order array representation of the passed 3D MRC file

    Args:
        grid: The 3D MRC file

    Returns:
        1-D z-order array representation of the passed 3D MRC file
    """

    # Create Z order array
    z_order_array = np.zeros(grid.size, dtype=np.intc)

    # Iterate over all points in the grid
    index = 0
    for grid_index, grid_value in np.ndenumerate(grid):

        if grid_value == 0: continue

        # Convert index to morton code and add to array
        z_order_array[index] = morton3D(k=k, x=grid_index[0], y=grid_index[1], z=grid_index[2])

        index += 1

    # Trim zero density elements
    z_order_array = np.trim_zeros(z_order_array)

    return np.sort(z_order_array, axis=None)

def compute_hausdorff_distance_3D_ZHD(morton_bits, grid1, grid2):
    """
    Computes and returns the Hausdorff distance for two 3D grids of MRC format.

    Args:
        grid1: The first grid of points in MRC format
        grid2: The second grid of points in MRC format

    Returns:
        The Hausdorff distance of the two 3D grids
    """

    directedDistance1 = compute_directed_hausdorff_distance_3D_ZHD(morton_bits, grid1, grid2)
    directedDistance2 = compute_directed_hausdorff_distance_3D_ZHD(morton_bits, grid2, grid1)

    if directedDistance1[0] > directedDistance2[0]:
        return directedDistance1
    else:
        return directedDistance2

def compute_directed_hausdorff_distance_3D_ZHD(morton_bits, grid1, grid2):

    # Get Z-order curves of the two grids
    z_order_1 = mrc_z_order(morton_bits, grid1)
    z_order_2 = mrc_z_order(morton_bits, grid2)

    max_distance = 0
    diffusion_center = int(z_order_2.size / 2)

    temp_point_1 = Point3D(-1, -1, -1)
    temp_point_2 = Point3D(-1, -1, -1)

    global_point_max_index = -1
    global_point_min_index = -1
    temp_point_max_index = -1
    temp_point_min_index = -1

    for i in range(1, z_order_1.size):

        min_distance = inf

        j = diffusion_center
        k = diffusion_center
        while j >= 0 or k < z_order_2.size:
            
            if j >= 0:
                extracted_coords_1 = extract_morton_coords_int(dim=3, k=morton_bits, morton_code=z_order_1[i])
                extracted_coords_2 = extract_morton_coords_int(dim=3, k=morton_bits, morton_code=z_order_2[j])

                temp_point_1.set_values(extracted_coords_1[0], extracted_coords_1[1], extracted_coords_1[2])
                temp_point_2.set_values(extracted_coords_2[0], extracted_coords_2[1], extracted_coords_2[2])

                left_distance = compute_euclidean_distance_3D_no_jit(temp_point_1, temp_point_2)

            if k < z_order_2.size:
                extracted_coords_1 = extract_morton_coords_int(dim=3, k=morton_bits, morton_code=z_order_1[i])
                extracted_coords_2 = extract_morton_coords_int(dim=3, k=morton_bits, morton_code=z_order_2[k])

                temp_point_1.set_values(extracted_coords_1[0], extracted_coords_1[1], extracted_coords_1[2])
                temp_point_2.set_values(extracted_coords_2[0], extracted_coords_2[1], extracted_coords_2[2])

                right_distance = compute_euclidean_distance_3D_no_jit(temp_point_1, temp_point_2)

            if left_distance < right_distance and left_distance < min_distance:
                min_distance = left_distance
                temp_point_max_index = i
                temp_point_min_index = j

            elif right_distance < left_distance and right_distance < min_distance:
                min_distance = right_distance
                temp_point_max_index = i
                temp_point_min_index = k

            if min_distance <= max_distance:
                diffusion_center = i if (left_distance < right_distance) else j
                break

            if j >= 0: j -= 1
            if k < z_order_2.size: k += 1

        if max_distance < min_distance:
            max_distance = min_distance
            global_point_max_index = temp_point_max_index
            global_point_min_index = temp_point_min_index

    # Get x0 and y0 as Point3D
    extracted_coords_1 = extract_morton_coords_int(dim=3, k=morton_bits, morton_code=z_order_1[global_point_max_index])
    extracted_coords_2 = extract_morton_coords_int(dim=3, k=morton_bits, morton_code=z_order_2[global_point_min_index])

    x0 = Point3D(extracted_coords_1[0], extracted_coords_1[1], extracted_coords_1[2])
    y0 = Point3D(extracted_coords_2[0], extracted_coords_2[1], extracted_coords_2[2])

    return (max_distance, (y0, grid2), (x0, grid1))
