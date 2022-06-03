import scipy
import numpy as np

from scipy.optimize import linear_sum_assignment


def construct_cost_matrix(F_q, F_r):
    def similarity():
        F_q_norm = np.linalg.norm(F_q, axis=-1)
        F_r_norm = np.linalg.norm(F_r, axis=-1)
        cos_sim_matrix = (F_q @ F_r.T) / np.outer(F_q_norm, F_r_norm)

        S = 0.5 * (1 + cos_sim_matrix)

        return S

    C = 1 - similarity()

    return C


def one2one_matching(C):
    matches = linear_sum_assignment(C)
    X = np.zeros_like(C)
    X[matches[0], matches[1]] = 1

    return X


def find_matches(F_q, F_r):
    C = construct_cost_matrix(F_q, F_r)
    X = one2one_matching(C)  # compute one to one matchings

    return C, X


def select_n_best_matches(img1_keypoints, img2_keypoints, C, X, n=50):
    match_costs = C * X
    threshold = np.sort(np.sum(match_costs, axis=-1))[n]
    match_costs[match_costs == 0] = np.inf
    best_n_ids = np.argwhere(match_costs < threshold)

    print(threshold)
    print(best_n_ids.shape)

    #points1_ids = best_50_ids[:,:,:2].reshape(-1, 2)
    #points1 = imgs1_keypoints[points1_ids[:,0], points1_ids[:,1]].reshape(7, 50, 2)

    #points2_ids = best_50_ids[:,:,::2].reshape(-1, 2)
    #points2 = imgs2_keypoints[points2_ids[:,0], points2_ids[:,1]].reshape(7, 50, 2)

    #print(points1.shape)
    #print(points2.shape)

    points1 = None
    points2 = None

    return points1, points2
