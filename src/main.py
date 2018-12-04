import numpy as np
import argparse
import mrcfile
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

        # Ensure that we are not using ZHD
        if args.algorithm == "zhd":
            print(colored("ZHD only works for 3D data.\n", "red"))
            quit()

        print(colored("Computing Hausdorff distance for 2D data...\n", "green"))

        start = time.time()

        # Compute the Hausdorff distance
        result = hausdorff2D.compute_hausdorff_distance_2D(mrc_file_1.data, mrc_file_2.data)
        result = hausdorff_result_to_dict(result)

        # Output point x0 to mrc file
        x0_mrc_file = hausdorff2D.point2D_to_mrc_file(args.output_mrc1, result["x0_origin"], result["x0"])

        # Output point y0 to mrc file
        y0_mrc_file = hausdorff2D.point2D_to_mrc_file(args.output_mrc2, result["y0_origin"], result["y0"])

        # Close output mrc files
        x0_mrc_file.close()
        y0_mrc_file.close()

        print("Hausdorff Distance: {}".format(result["max_distance"]))
        print("Point x0: ({},{})".format(result["x0"][0], result["x0"][1]))
        print("Point y0: ({},{})".format(result["y0"][0], result["y0"][1]))
        end = time.time()

        print("\nExecution Time: {}s".format(end-start))

        print(colored("Done.", "green"))

    # 3D data
    elif not is_2D(mrc_file_1.data) and not is_2D(mrc_file_2.data):
        print(colored("Computing Hausdorff distance for 3D data...\n", "green"))

        start = time.time()

        # Compute the Hausdorff distance
        result = None
        if args.algorithm == "earlybreak":
            result = hausdorff3D.compute_hausdorff_distance_3D(mrc_file_1.data, mrc_file_2.data)
        else:
            result = hausdorff3D.compute_hausdorff_distance_3D_ZHD(12, mrc_file_1.data, mrc_file_2.data)

        result = hausdorff_result_to_dict(result)

        # Output point x0 to mrc file
        x0_mrc_file = hausdorff3D.point3D_to_mrc_file(args.output_mrc1, result["x0_origin"], result["x0"])

        # Output point y0 to mrc file
        y0_mrc_file = hausdorff3D.point3D_to_mrc_file(args.output_mrc2, result["y0_origin"], result["y0"])

        # Close output mrc files
        x0_mrc_file.close()
        y0_mrc_file.close()

        print("Hausdorff Distance: {}".format(result["max_distance"]))
        print("Point x0: ({},{},{})".format(result["x0"][0], result["x0"][1], result["x0"][2]))
        print("Point y0: ({},{},{})".format(result["y0"][0], result["y0"][1], result["y0"][2]))
        end = time.time()

        print("\nExecution Time: {}s".format(end-start))

        print(colored("Done.", "green"))

    # Error: Both mrc files must contain either 2D or 3D data
    else:
        print(colored("Error: ", "red") + "Both mrc files must contain either 2D or 3D data.")

    # Close file handles
    mrc_file_1.close()
    mrc_file_2.close()

def hausdorff_result_to_dict(haudorff_result):
    """
    Converts the tuple returned by the Hausdorff distance functions to a dictionary
    for easier use.

    Args:
        haudorff_result: Tuple returned from Hausdorff distance function

    Returns:
        Dictionary version of the tuple.
    """

    return {
        "max_distance": haudorff_result[0],
        "x0": haudorff_result[1][0],
        "x0_origin": haudorff_result[1][1],
        "y0": haudorff_result[2][0],
        "y0_origin": haudorff_result[2][1]
    }

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
    parser.add_argument("--outputMrc1", required=True, help="The output x0 point in MRC format.", dest="output_mrc1")
    parser.add_argument("--outputMrc2", required=True, help="The output y0 point in MRC format.", dest="output_mrc2")
    parser.add_argument("--algorithm", "-a", required=False, choices=["earlybreak", "zhd"], default="earlybreak", help="The algorithm to use.", dest="algorithm")

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