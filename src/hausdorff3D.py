import numpy as np
import mrcfile
from math import (sqrt, pow, inf)
from concurrent.futures import ThreadPoolExecutor
from numba import jit
from morton import (morton3D, extract_morton_coords_int_3D)

def point3D_to_mrc_file(output_mrc_filename, origin_grid, point):
    """
    Outputs the passed point as an MRC file based on its original MRC file.
    The point will have double the density of the maximum density of the original MRC.

    Args:
        output_mrc_filename: Path of the output MRC file
        origin_grid: The grid the point originated from
        point: The point to output

    Returns:
        Handle to the new MRC file.
    """

    # Create a zero array with the same shape as the original
    new_grid = np.copy(origin_grid)

    # Determine largest element
    max_element = np.amax(new_grid)

    # Set the element identified by the point to twice the value of the max_element so it can be distinguished
    new_grid[int(point[0]), int(point[1]), int(point[2])] = max_element * 2

    return mrcfile.new(name=output_mrc_filename, data=new_grid, overwrite=True)
    
#################################################
# EARLYBREAK        
#################################################

@jit(nopython=True, cache=True)
def euclidean_distance_3D(point1, point2):
    """
    Computes and returns the euclidean distance for two points in 3D space.

    Args:
        point1: The first point
        point2: The second point

    Returns:
        The Euclidean distance of the two points in 3D space
    """

    return sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2) + pow(point1[2] - point2[2], 2))

def hausdorff_distance_3D(grid1, grid2):
    """
    Computes and returns the Hausdorff distance for two 3D grids of MRC format
    using the Earlybreak algorithm.

    Args:
        grid1: The first grid of points in MRC format
        grid2: The second grid of points in MRC format

    Returns:
        The Hausdorff distance of the two 3D grids
    """

    executor = ThreadPoolExecutor(max_workers=2)

    future1 = executor.submit(directed_hausdorff_distance_3D, grid1, grid2)
    future2 = executor.submit(directed_hausdorff_distance_3D, grid2, grid1)

    directedDistance1 = future1.result()
    directedDistance2 = future2.result()

    executor.shutdown()

    if directedDistance1[0] > directedDistance2[0]:
        return directedDistance1
    else:
        return directedDistance2

@jit(nopython=True, nogil=True, cache=True)
def directed_hausdorff_distance_3D(grid1, grid2):
    """
    Computes and returns the directed Hausdorff distance for two 3D grids of MRC format
    using the Earlybreak algorithm.

    Args:
        grid1: The first grid of points in MRC format (origin)
        grid2: The second grid of points in MRC format (destination)

    Returns:
        The directed Hausdorff distance of the two 3D grids
    """

    max_distance = 0
    currrent_distance = 0

    # Create point objects
    point1 = np.empty(3, np.intc)
    point2 = np.empty(3, np.intc)

    temp_point_min = np.empty(3, np.intc)

    point_min = np.zeros(3, np.intc)
    point_max = np.zeros(3, np.intc)
        
    # Iterate over all points in grid 1
    for grid1_index, grid1_value in np.ndenumerate(grid1):

        if (grid1_value == 0): continue

        point1[0] = grid1_index[0]
        point1[1] = grid1_index[1]
        point1[2] = grid1_index[2]

        min_distance = inf
            
        # Iterate over all points in grid 2
        for grid2_index, grid2_value in np.ndenumerate(grid2):

            if (grid2_value == 0): continue

            point2[0] = grid2_index[0]
            point2[1] = grid2_index[1]
            point2[2] = grid2_index[2]

            currrent_distance = euclidean_distance_3D(point1, point2)

            if currrent_distance <= max_distance:
                min_distance = 0
                break

            if (min_distance > currrent_distance):
                min_distance = currrent_distance

                temp_point_min[0] = point2[0]
                temp_point_min[1] = point2[1]
                temp_point_min[2] = point2[2]

        if max_distance < min_distance:
            max_distance = min_distance

            point_max[0] = point1[0]
            point_max[1] = point1[1]
            point_max[2] = point1[2]

            point_min[0] = temp_point_min[0]
            point_min[1] = temp_point_min[1]
            point_min[2] = temp_point_min[2]

    return (max_distance, (point_min, grid2), (point_max, grid1))

#################################################
# ZHD        
#################################################

@jit(nopython=True, cache=True)
def mrc_z_order(k, grid):
    """
    Creates a 1-D z-order array representation of the passed 3D MRC file

    Args:
        grid: The 3D MRC file

    Returns:
        1-D z-order array representation of the passed 3D MRC file
    """

    # Create Z order array
    z_order_array = np.full(grid.size, -1, dtype=np.intc)

    # Iterate over all points in the grid
    index = 0
    for grid_index, grid_value in np.ndenumerate(grid):

        if grid_value == 0: continue

        # Convert index to morton code and add to array
        z_order_array[index] = morton3D(k=k, x=grid_index[0], y=grid_index[1], z=grid_index[2])

        index += 1

    z_order_array = z_order_array[z_order_array != -1]

    return np.sort(z_order_array)

def hausdorff_distance_3D_ZHD(morton_bits, grid1, grid2):
    """
    Computes and returns the Hausdorff distance for two 3D grids of MRC format
    using the ZHD algorithm.

    Args:
        morton_bits: Bits to represent each coordinate with in morton code
        grid1: The first grid of points in MRC format
        grid2: The second grid of points in MRC format

    Returns:
        The Hausdorff distance of the two 3D grids
    """

    executor = ThreadPoolExecutor(max_workers=2)

    future1 = executor.submit(directed_hausdorff_distance_3D_ZHD, morton_bits, grid1, grid2)
    future2 = executor.submit(directed_hausdorff_distance_3D_ZHD, morton_bits, grid2, grid1)

    directedDistance1 = future1.result()
    directedDistance2 = future2.result()

    executor.shutdown()

    if directedDistance1[0] > directedDistance2[0]:
        return directedDistance1
    else:
        return directedDistance2

@jit(nopython=True, fastmath=True, cache=True)
def directed_hausdorff_distance_3D_ZHD(morton_bits, grid1, grid2):
    """
    Computes and returns the directed Hausdorff distance for two 3D grids of MRC format
    using the ZHD algorithm.

    Args:
        morton_bits: Bits to represent each coordinate with in morton code
        grid1: The first grid of points in MRC format (origin)
        grid2: The second grid of points in MRC format (destination)

    Returns:
        The directed Hausdorff distance of the two 3D grids
    """

    # Get Z-order curves of the two grids
    z_order_1 = mrc_z_order(morton_bits, grid1)
    z_order_2 = mrc_z_order(morton_bits, grid2)

    max_distance = 0
    diffusion_center = z_order_2.size // 2

    temp_point_1 = np.empty(3, dtype=np.intc)
    temp_point_2 = np.empty(3, dtype=np.intc)

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
                extracted_coords_1 = extract_morton_coords_int_3D(k=morton_bits, morton_code=z_order_1[i])
                extracted_coords_2 = extract_morton_coords_int_3D(k=morton_bits, morton_code=z_order_2[j])

                temp_point_1[0] = extracted_coords_1[0]
                temp_point_1[1] = extracted_coords_1[1]
                temp_point_1[2] = extracted_coords_1[2]

                temp_point_2[0] = extracted_coords_2[0]
                temp_point_2[1] = extracted_coords_2[1]
                temp_point_2[2] = extracted_coords_2[2]

                left_distance = euclidean_distance_3D(temp_point_1, temp_point_2)

            if k < z_order_2.size:
                extracted_coords_1 = extract_morton_coords_int_3D(k=morton_bits, morton_code=z_order_1[i])
                extracted_coords_2 = extract_morton_coords_int_3D(k=morton_bits, morton_code=z_order_2[k])

                temp_point_1[0] = extracted_coords_1[0]
                temp_point_1[1] = extracted_coords_1[1]
                temp_point_1[2] = extracted_coords_1[2]
                
                temp_point_2[0] = extracted_coords_2[0]
                temp_point_2[1] = extracted_coords_2[1]
                temp_point_2[2] = extracted_coords_2[2]

                right_distance = euclidean_distance_3D(temp_point_1, temp_point_2)

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
    extracted_coords_1 = extract_morton_coords_int_3D(k=morton_bits, morton_code=z_order_1[global_point_max_index])
    extracted_coords_2 = extract_morton_coords_int_3D(k=morton_bits, morton_code=z_order_2[global_point_min_index])

    x0 = np.array([extracted_coords_1[0], extracted_coords_1[1], extracted_coords_1[2]], dtype=np.intc)
    y0 = np.array([extracted_coords_2[0], extracted_coords_2[1], extracted_coords_2[2]], dtype=np.intc)

    return (max_distance, (y0, grid2), (x0, grid1))
