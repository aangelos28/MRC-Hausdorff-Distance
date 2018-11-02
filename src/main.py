import mrcfile
import argparse
from termcolor import colored

# TEMP
import time

import hausdorff3D
import hausdorff2D

def main():
    """
    Script entry point
    """

    args = parse_arguments()

    #np.set_printoptions(threshold=np.inf)

    mrc_file_1 = readMRCFile(args.mrc1)
    mrc_file_2 = readMRCFile(args.mrc2)

    # 2D data
    if is_2D(mrc_file_1.data) and is_2D(mrc_file_2.data):
        print(colored("Computing Hausdorff distance for 2D data...\n", "green"))

        start = time.time()

        result = hausdorff2D.compute_hausdorff_distance_2D(mrc_file_1.data, mrc_file_2.data)
        print("Hausdorff Distance: {}".format(result[0]))
        print("Point x0: ({},{})".format(result[1].x, result[1].y))
        print("Point y0: ({},{})".format(result[2].x, result[2].y))
        end = time.time()

        print("\nExecution Time: {}s".format(end-start))

        print(colored("Done.", "green"))

    # 3D data
    elif not is_2D(mrc_file_1.data) and not is_2D(mrc_file_2.data):
        print(colored("Computing Hausdorff distance for 3D data...\n", "green"))

        start = time.time()

        result = hausdorff3D.compute_hausdorff_distance_3D(mrc_file_1.data, mrc_file_2.data)
        print("Hausdorff Distance: {}".format(result[0]))
        print("Point x0: ({},{},{})".format(result[1].x, result[1].y, result[1].z))
        print("Point y0: ({},{},{})".format(result[2].x, result[2].y, result[2].z))
        end = time.time()

        print("\nExecution Time: {}s".format(end-start))

        print(colored("Done.", "green"))

    # Error: Both mrc files must contain either 2D or 3D data
    else:
        print(colored("Error: ", "red") + "Both mrc files must contain either 2D or 3D data.")

    # Close file handles
    mrc_file_1.close()
    mrc_file_2.close()

def is_2D(mrc_data):
    """
    Checks if the given numpy array is 2D

    Args:
        mrc_data: Numpy array to check

    Returns:
        Whether the array is 2D
    """

    return (mrc_data.shape[0] == 1 or mrc_data.shape[1] == 1 or mrc_data.shape[2] == 1)

def parse_arguments():
    """
    Parse CLI arguments and return an object containing the values.
    """

    parser = argparse.ArgumentParser(description="Hausdorff Distance Solver for 2 Sets.")

    parser.add_argument("--mrc1", required=True, help="The first set of points in MRC format", dest="mrc1")
    parser.add_argument("--mrc2", required=True, help="The second set of points in MRC format", dest="mrc2")
    parser.add_argument("--outputPDB", "-o", required=True, help="The output x0 and y0 points in PDB format.", dest="output_pdb_path")

    return parser.parse_args()

def readMRCFile(path):
    """
    Open a handle to an MRC file. Must close it when done.

    Args:
        path: The path of the mrc file

    Returns:
        The mrc file handle
    """

    return mrcfile.open(path)


if __name__ == "__main__": main()