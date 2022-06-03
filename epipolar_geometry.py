import cv2
import numpy as np


def estimate_fundamental_matrix(p, q, method='RANSAC'):
    p = p.astype(np.float32)
    q = q.astype(np.float32)

    if method == '8Point':
        F, inlier_mask = cv2.findFundamentalMat(p, q, method=cv2.FM_8Point)
    elif method == 'RANSAC':
        F, inlier_mask = cv2.findFundamentalMat(p, q, method=cv2.FM_RANSAC)
    else:
        raise ValueError

    return F, inlier_mask


def compute_epipolar_lines():
    pass


def compute_epipoles():
    pass
