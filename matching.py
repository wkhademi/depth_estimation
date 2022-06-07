import cv2
import scipy
import numpy as np

from scipy.optimize import linear_sum_assignment


def construct_cost_matrix(F_q, F_r):
    def similarity():
        F_q_norm = np.linalg.norm(F_q, axis=-1)
        F_r_norm = np.linalg.norm(F_r, axis=-1)
        cos_sim_matrix = (F_q @ F_r.T) / np.outer(F_q_norm, F_r_norm)
        cos_sim_matrix = np.clip(cos_sim_matrix, -1., 1.)

        S = 0.5 * (1 + cos_sim_matrix)

        return S

    C = 1 - similarity()

    return C


def one2one_matching(C):
    matches = linear_sum_assignment(C)
    X = np.zeros_like(C)
    X[matches[0], matches[1]] = 1

    return X


def find_keypoint_matches(F_q, F_r):
    C = construct_cost_matrix(F_q, F_r)
    X = one2one_matching(C)  # compute one to one matchings

    return C, X


def select_n_best_matches(img1_keypoints, img2_keypoints, C, X, n=50):
    match_costs = C * X
    threshold = np.sort(np.sum(match_costs, axis=-1))[n-1]
    criteria = np.logical_and(match_costs <= threshold, X != 0)
    best_n_ids = np.argwhere(criteria)

    p_ids = best_n_ids[:,0]
    p = img1_keypoints[p_ids]

    q_ids = best_n_ids[:,1]
    q = img2_keypoints[q_ids]

    return p, q


def stereo_matching(img1, img2, method, window_size, num_disp, min_disp):
    if method == 'SGBM':
        left_matcher = cv2.StereoSGBM_create(minDisparity = min_disp,
                                             numDisparities = num_disp,
                                             blockSize = window_size,
                                             P1 = 8*3*(window_size**2),
                                             P2 = 32*3*(window_size**2),
                                             disp12MaxDiff = 1,
                                             uniquenessRatio = 5,
                                             speckleWindowSize = 200,
                                             speckleRange = 2
                                             )
    elif method == 'BM':
        left_matcher = cv2.StereoBM_create(numDisparities=num_disp,
                                           blockSize=window_size)
        left_matcher.setMinDisparity(min_disp)
    else:
        raise ValueError

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # compute disparities
    left_disp = left_matcher.compute(img1, img2).astype(np.float32) / 16.0
    right_disp = right_matcher.compute(img2, img1).astype(np.float32) / 16.0
    right_disp = np.where(-1*right_disp < min_disp + num_disp, -1*right_disp, -1.)

    return left_disp, right_disp
