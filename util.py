#  ____  ____      _    __  __  ____ ___
# |  _ \|  _ \    / \  |  \/  |/ ___/ _ \
# | | | | |_) |  / _ \ | |\/| | |  | | | |
# | |_| |  _ <  / ___ \| |  | | |__| |_| |
# |____/|_| \_\/_/   \_\_|  |_|\____\___/
#                           research group
#                             dramco.be/
#
#  KU Leuven - Technology Campus Gent,
#  Gebroeders De Smetstraat 1,
#  B-9000 Gent, Belgium
#
#         File: util.py
#      Created: 2018-01-31
#       Author: Geoffrey Ottoy
#      Version: 0.1
#
#  Description: General utility functions for localization purposes
#      some more info
#
#

import numpy as np


# Compute Euclidian distances between a point P and other cluster in two dimensions
# inputs:
#   point: an 1-by-2 array containing the coordinates [x, y] of the point P
#   cluster: an n-by-2 array containing the coordinates [x0, y0; x1, 2; ... xn-1, yn-1] of the cluster to which to
#           compute the distance to point P to
# output:
#   an n-by-1 array containing the distances to P (or NaN)
def distance_between_2d(point, cluster):
    distance = np.nan

    # Type check for 'point'
    if type(point) is np.ndarray:
        # get dimensions
        pshape = point.shape
    else:
        print("Error: Argument 'point' should be an array (np.ndarray).")
        return distance         # [NaN]

    # Type check for 'cluster'
    if type(cluster) is np.ndarray:
        # get dimensions
        cshape = cluster.shape
    else:
        print("Error: Argument 'cluster' should be an array (np.ndarray).")
        return distance         # [NaN]

    # Check 'point' dimensions
    if len(pshape) != 1:
        print("Error: Argument 'point' should be a 2-element ndarray. a")
        return distance  # [NaN]
    if pshape[0] != 2:
        print("Error: Argument 'point' should be a 2-element ndarray. b")
        return distance         # [NaN]

    # Check 'cluster' dimensions
    if len(cshape) != 2:
        print("Error: Argument 'cluster' should be an n-by-2 ndarray.")
        return distance  # [NaN]
    if cshape[1] != 2:
        print("Error: Argument 'cluster' should be an n-by-2 ndarray.")
        return distance  # [NaN]

    x_d = cluster[:, 0] - point[0]
    y_d = cluster[:, 1] - point[1]
    distances = np.sqrt(np.square(x_d) + np.square(y_d))

    return distances


# Compute Euclidian distances between a point P and other cluster in two dimensions
# inputs:
#   point: an 1-by-3 array containing the coordinates [x, y] of the point P
#   cluster: an n-by-3 array containing the coordinates [x0, y0; x1, 2; ... xn-1, yn-1] of the cluster to which to
#           compute the distance to point P to
# output:
#   an n-by-1 array containing the distances to P (or NaN)
def distance_between_3d(point, cluster):
    distance = np.nan

    # Type check for 'point'
    if type(point) is np.ndarray:
        # get dimensions
        pshape = point.shape
    else:
        print("Error: Argument 'point' should be an array (np.ndarray).")
        return distance         # [NaN]

    # Type check for 'cluster'
    if type(cluster) is np.ndarray:
        # get dimensions
        cshape = cluster.shape
    else:
        print("Error: Argument 'cluster' should be an array (np.ndarray).")
        return distance         # [NaN]

    # Check 'point' dimensions
    if len(pshape) != 1:
        print("Error: Argument 'point' should be a 3-element ndarray. a")
        return distance  # [NaN]
    if pshape[0] != 3:
        print("Error: Argument 'point' should be a 3-element ndarray. b")
        return distance         # [NaN]

    # Check 'cluster' dimensions
    if len(cshape) != 2:
        print("Error: Argument 'cluster' should be an n-by-3 ndarray.")
        return distance  # [NaN]
    if cshape[1] != 3:
        print("Error: Argument 'cluster' should be an n-by-3 ndarray.")
        return distance  # [NaN]

    x_d = cluster[:, 0] - point[0]
    y_d = cluster[:, 1] - point[1]
    z_d = cluster[:, 2] - point[2]
    distances = np.sqrt(np.square(x_d) + np.square(y_d) + np.square(z_d))

    return distances


def colinear(cluster):
    # Type check for 'cluster'
    if type(cluster) is np.ndarray:
        # get dimensions
        cshape = cluster.shape
    else:
        print("Error: Argument 'cluster' should be an array (np.ndarray).")
        return None

    # Check 'cluster' dimensions
    if len(cshape) != 2:
        print("Error: Argument 'cluster' should be an n-by-2 ndarray.")
        return None
    if cshape[1] != 2:
        print("Error: Argument 'cluster' should be an n-by-2 ndarray.")
        return None

    if np.linalg.matrix_rank(cluster) == 1:
        return True
    else:   # first 2 points are on a vertical line
        return False


def coplanar(cluster):
    # Type check for 'cluster'
    if type(cluster) is np.ndarray:
        # get dimensions
        cshape = cluster.shape
    else:
        print("Error: Argument 'cluster' should be an array (np.ndarray).")
        return None

    # Check 'cluster' dimensions
    if len(cshape) != 2:
        print("Error: Argument 'cluster' should be an n-by-3 ndarray.")
        return None
    if cshape[1] != 3:
        print("Error: Argument 'cluster' should be an n-by-3 ndarray.")
        return None

    if np.linalg.matrix_rank(cluster) == 2:
        return True
    else:   # first 2 points are on a vertical line
        return False
