import cv2
import numpy as np


def estimate_fundamental_matrix(p, q, method='RANSAC'):
    if method == '8Point':
        F, inlier_mask = cv2.findFundamentalMat(p.astype(np.float32), q.astype(np.float32), method=cv2.FM_8POINT)
    elif method == 'RANSAC':
        F, inlier_mask = cv2.findFundamentalMat(p.astype(np.float32), q.astype(np.float32), method=cv2.FM_RANSAC)
    else:
        raise ValueError

    return F, inlier_mask


def compute_epipolar_lines(p, q, F, inlier_mask):
    p = p[inlier_mask.ravel() == 1]
    q = q[inlier_mask.ravel() == 1]

    # compute epipolar lines for image 1
    l1 = cv2.computeCorrespondEpilines(q.reshape(-1, 1, 2), 2, F)
    l1 = l1.reshape(-1, 3)

    # compute epipolar lines for image 2
    l2 = cv2.computeCorrespondEpilines(p.reshape(-1, 1, 2), 1, F)
    l2 = l2.reshape(-1, 3)

    return l1, l2


def compute_epipoles(F):
    # compute epipole in image 1
    U, S, Vh = np.linalg.svd(F)
    e1 = np.rint(Vh[2,:] / Vh[2,-1]).astype(np.int32)

    # compute epipole in image 2
    U, S, Vh = np.linalg.svd(F.T)
    e2 = np.rint(Vh[2,:] / Vh[2,-1]).astype(np.int32)

    return e1, e2


def rectify(img1, p, img2, q, F, inlier_mask, h, w):
    p = p[inlier_mask.ravel() == 1]
    q = q[inlier_mask.ravel() == 1]

    # compute rectification transformation
    _, H1, H2 = cv2.stereoRectifyUncalibrated(p, q, F, (w, h))

    # warp images to same plane parallel to the baseline
    img1_rectified = cv2.warpPerspective(img1, H1, (w, h))
    img2_rectified = cv2.warpPerspective(img2, H2, (w, h))

    return img1_rectified, img2_rectified, H1, H2
