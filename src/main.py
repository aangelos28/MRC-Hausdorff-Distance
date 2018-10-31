import mrcfile
import argparse

# TEMP
import time

import hausdorff

def main():
    """
    Script entry point
    """

    args = parse_arguments()

    #np.set_printoptions(threshold=np.inf)

    mrc_file_1 = readMRCFile(args.mrc1)
    mrc_file_2 = readMRCFile(args.mrc2)

    start = time.time()
    result = hausdorff.compute_hausdorff_distance_3D(mrc_file_1.data, mrc_file_2.data)
    print("Hausdorff Distance: {}".format(result[0]))
    end = time.time()
    print("Execution Time: {}s".format(end-start))

    # Close file handles
    mrc_file_1.close()
    mrc_file_2.close()

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
    """

    return mrcfile.open(path)


if __name__ == "__main__": main()