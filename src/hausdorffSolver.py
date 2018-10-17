from math import sqrt, pow

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
    
    def euclidean_norm(self):
        """
        Computes and returns the euclidean norm for this point.
        """

        return sqrt(pow(self.__x, 2) + pow(self.__y, 2) + pow(self.__z, 2))

class HausdorffSolver:

    @staticmethod
    def compute_hausdorff_distance_3D(grid1, grid2):
        """
        Computes and returns the hausdorff distance for two 3D grids of MRC format.
        """

        print("STUB: Compute Hausdorff distance")

    @staticmethod
    def compute_euclidean_distance_3D(point1, point2):
        """
        Computes and returns the euclidean distance for two points in 3D space.
        """

        return sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2) + pow(point1.z - point2.z, 2))