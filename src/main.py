import mrcfile
import numpy as np
import argparse

def main():
    """
    Script entry point
    """

    args = parse_arguments()

    mrc_file_1 = readMRCFile(args.set_1)

    #np.set_printoptions(threshold=np.inf)
    print(mrc_file_1.data.shape)
    print(mrc_file_1.data)

    # Close file handles
    mrc_file_1.close()

def parse_arguments():
    """
    Parse CLI arguments and return an object containing the values.
    """

    parser = argparse.ArgumentParser(description="Hausdorff Distance Solver for 2 Sets.")

    parser.add_argument("--set1", required=True, help="The first set of points in MRC format", dest="set_1")
    parser.add_argument("--set2", required=True, help="The second set of points in MRC format", dest="set_2")
    parser.add_argument("--outputPDB", "-o", required=True, help="The output x0 and y0 points in PDB format.", dest="output_pdb_path")

    return parser.parse_args()

def readMRCFile(path):
    """
    Open a handle to an MRC file. Must close it when done.
    """

    return mrcfile.open(path)


if __name__ == "__main__": main()