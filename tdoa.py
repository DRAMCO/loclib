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
#         File: tdoa.py
#      Created: 2018-01-31
#       Author: Geoffrey Ottoy
#      Version: 0.1
#
#  Description: Functions for TDOA localization
#      some more info
#
#

import numpy as np
import loclib.util as loctools


# compute a 2 dimensional position
def compute_xy(ref_nodes, d_differences, min_r_error=0.001, max_nr_steps=10):
    position = np.array([np.nan, np.nan], dtype=float)

    # Type check for 'ref_nodes'
    if type(ref_nodes) is np.ndarray:
        # get dimensions
        rshape = ref_nodes.shape
    else:
        print("Error: Argument 'ref_nodes' should be a matrix (np.matrix).")
        return position  # [NaN, NaN]

    # Type check for 'd_differences'
    if type(d_differences) is np.ndarray:
        # get dimensions
        dshape = d_differences.shape
    else:
        print("Error: Argument 'd_differences' should be a matrix (np.matrix).")
        return position  # [NaN, NaN]

    # Verify that first value in d_differences is 0
    if d_differences[0] != 0:
        print("Error: Differences need to be referenced to first the node,")
        print("       i.e., the first distance difference needs to be 0.")
        return position  # [NaN, NaN]

    # Check d_differences dimensions
    if len(dshape) != 1:
        print("Error: Argument 'd_differences' should be an n-element array.")
        return position  # [NaN, NaN]
    else:
        drows = dshape[0]

    # Check ref_nodes dimensions
    if len(rshape) != 2:
        print("Error: Argument 'ref_nodes' needs to be an n-by-2 matrix.")
        return position  # [NaN, NaN]
    else:
        rrows = rshape[0]
        rcols = rshape[1]
    if rcols != 2:
        print("Error: Argument 'ref_nodes' needs to be an n-by-2 matrix.")
        return position  # [NaN, NaN]

    if rrows != drows:
        print("Error: Arguments 'ref_nodes' and 'd_differences' should have an equal number of rows.")
        return position  # [NaN, NaN]

    # TODO: add check that min(x) != max(x) and min(y) != max(y)

    # Make sure enough data are provided to run the TDOA algorithm
    if drows < 3:
        print("Error: At least 3 reference nodes are required to compute a position.")
        return position  # [NaN, NaN]

    # Now we can do the magic (i.e., run the algorithm)
    # Create necessary matrices
    x1 = ref_nodes[0, 0]
    y1 = ref_nodes[0, 1]

    # -------------------------------------------------
    # Initial estimation
    # -------------------------------------------------
    # we start 'in the middle'
    p_est = np.array([0, 0], dtype=float)
    p_est[0] = ref_nodes[:, 0].mean()
    p_est[1] = ref_nodes[:, 1].mean()

    # -------------------------------------------------
    # Iterative phase
    # -------------------------------------------------
    step = 0
    r_error = 100
    while (r_error > min_r_error) and (step < max_nr_steps):
        step = step + 1
        r_est = loctools.distance_between_2d(p_est, ref_nodes)

        a1 = (-x1 + p_est[0]) / r_est[0]
        b1 = (-y1 + p_est[1]) / r_est[0]
        ai = (-ref_nodes[:, 0] + p_est[0]) / r_est  # division is element-by-element
        bi = (-ref_nodes[:, 1] + p_est[1]) / r_est  # division is element-by-element

        a_mat = np.column_stack(((a1 - ai), (b1 - bi)))
        b_mat = d_differences - r_est + r_est[0]

        # least-squares estimation of the positioning error
        step1 = np.matmul(a_mat.T,  a_mat)
        try:
            step2 = np.linalg.inv(step1)
        except np.linalg.LinAlgError as err:
            print(err)
            return position  # [NaN, NaN]

        esti = np.matmul(np.matmul(step2, a_mat.T), b_mat)
        if np.isnan(esti).any():
            return position  # [NaN, NaN]

        # see how far we are still off
        r_error = abs(esti.mean())
        # update location guess based on error
        p_est[0] -= esti[0]
        p_est[1] -= esti[1]

    return p_est


# compute a 3 dimensional position
def compute_xyz(ref_nodes, d_differences, min_r_error=0.001, max_nr_steps=10):
    position = np.array([np.nan, np.nan, np.nan], dtype=float)

    # Type check for 'ref_nodes'
    if type(ref_nodes) is np.ndarray:
        # get dimensions
        rshape = ref_nodes.shape
    else:
        print("Error: Argument 'ref_nodes' should be a matrix (np.matrix).")
        return position  # [NaN, NaN]

    # Type check for 'd_differences'
    if type(d_differences) is np.ndarray:
        # get dimensions
        dshape = d_differences.shape
    else:
        print("Error: Argument 'd_differences' should be a matrix (np.matrix).")
        return position  # [NaN, NaN]

    # Verify that first value in d_differences is 0
    if d_differences[0] != 0:
        print("Error: Differences need to be referenced to first the node,")
        print("       i.e., the first distance difference needs to be 0.")
        return position  # [NaN, NaN]

    # Check d_differences dimensions
    if len(dshape) != 1:
        print("Error: Argument 'd_differences' should be an n-element array.")
        return position  # [NaN, NaN]
    else:
        drows = dshape[0]

    # Check ref_nodes dimensions
    if len(rshape) != 2:
        print("Error: Argument 'ref_nodes' needs to be an n-by-3 matrix.")
        return position  # [NaN, NaN]
    else:
        rrows = rshape[0]
        rcols = rshape[1]
    if rcols != 3:
        print("Error: Argument 'ref_nodes' needs to be an n-by-3 matrix.")
        return position  # [NaN, NaN]

    if rrows != drows:
        print("Error: Arguments 'ref_nodes' and 'd_differences' should have an equal number of rows.")
        return position  # [NaN, NaN]

    # TODO: add check that min(x) != max(x) and min(y) != max(y) and min(z) != max(z)

    # Make sure enough data are provided to run the TDOA algorithm
    if drows < 4:
        print("Error: At least 4 reference nodes are required to compute a position.")
        return position  # [NaN, NaN]

    # Now we can do the magic (i.e., run the algorithm)
    # Create necessary matrices
    x1 = ref_nodes[0, 0]
    y1 = ref_nodes[0, 1]
    z1 = ref_nodes[0, 2]

    # -------------------------------------------------
    # Initial estimation
    # -------------------------------------------------
    # we start 'in the middle'
    p_est = np.array([ref_nodes[:, 0].mean(), ref_nodes[:, 1].mean(), ref_nodes[:, 2].mean()], dtype=float)

    # -------------------------------------------------
    # Iterative phase
    # -------------------------------------------------
    step = 0
    r_error = 100
    while (r_error > min_r_error) and (step < max_nr_steps):
        step = step + 1
        r_est = loctools.distance_between_3d(p_est, ref_nodes)

        a1 = (-x1 + p_est[0]) / r_est[0]
        b1 = (-y1 + p_est[1]) / r_est[0]
        c1 = (-z1 + p_est[2]) / r_est[0]
        ai = (-ref_nodes[:, 0] + p_est[0]) / r_est  # division is element-by-element
        bi = (-ref_nodes[:, 1] + p_est[1]) / r_est  # division is element-by-element
        ci = (-ref_nodes[:, 2] + p_est[2]) / r_est  # division is element-by-element

        a_mat = np.column_stack(((a1 - ai), (b1 - bi), (c1 - ci)))
        b_mat = d_differences - r_est + r_est[0]

        # least-squares estimation of the positioning error
        step1 = np.matmul(a_mat.T,  a_mat)
        try:
            step2 = np.linalg.inv(step1)
        except np.linalg.LinAlgError as err:
            print(err)
            return position  # [NaN, NaN]

        esti = np.matmul(np.matmul(step2, a_mat.T), b_mat)
        if np.isnan(esti).any():
            return position  # [NaN, NaN]

        # see how far we are still off
        r_error = abs(esti.mean())
        # update location guess based on error
        p_est[0] -= esti[0]
        p_est[1] -= esti[1]
        p_est[2] -= esti[2]

    return p_est
