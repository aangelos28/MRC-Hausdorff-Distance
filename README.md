# MRC Hausdorff Distance
A tool for computing the Hausdorff distance of two MRC files.

Implements the Earlybreak [1] and ZHD [2] algorithms.

## Installation
Install the required dependencies from `requirements.txt`

## Usage
`./main.py --mrc1 MRC_1_PATH --mrc2 MRC_2_PATH --outputMrc1 OUTPUT_MRC_1_W_x0 --outputMrc2 OUTPUT_MRC_2_W_y0 --algorithm earlybreak,zhd`

## Citations
[1] A. A. Taha and A. Hanbury, "An Efficient Algorithm for Calculating the Exact Hausdorff Distance," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 37, no. 11, pp. 2153-2163, 1 Nov. 2015, doi: 10.1109/TPAMI.2015.2408351.

[2] D. Zhang, L. Zou, Y. Chen and F. He, "Efficient and Accurate Hausdorff Distance Computation Based on Diffusion Search," in IEEE Access, vol. 6, pp. 1350-1361, 2018, doi: 10.1109/ACCESS.2017.2778745.