import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from CNN import ShallowNet, DeepNet


def SIFT_detector(img, num_keypoints):
    def select_n_best(keypoints, n=20):
        '''
        Select n best (highest response) SIFT keypoints that are not duplicates.
        '''
        keypoint_response = np.array([keypoint.response for keypoint in keypoints])
        keypoints, unique_keypoints_id = np.unique([keypoint.pt for keypoint in keypoints], return_index=True, axis=0)
        keypoint_response = keypoint_response[unique_keypoints_id]
        response_order = np.argsort(keypoint_response)[::-1]

        keypoints = keypoints[response_order]
        keypoint_response = keypoint_response[response_order]

        best_keypoints = keypoints[:n].astype(np.int32)

        return best_keypoints

    # create SIFT detector
    sift = cv2.SIFT_create(nfeatures=1000)

    # detect SIFT keypoints in images
    sift_keypoints = sift.detect(img, None)

    # select highest response unique keypoints
    sift_keypoints = select_n_best(sift_keypoints, n=num_keypoints)

    return sift_keypoints


def Harris_detector(img, num_keypoints):
    # detect harris corners in image
    harris_keypoints = cv2.cornerHarris(img, 2, 3, 0.04)

    # adaptive thresholding
    threshold = np.sort(harris_keypoints, axis=None)[-num_keypoints]
    harris_keypoints = np.argwhere(harris_keypoints >= threshold)
    harris_keypoints = np.flip(harris_keypoints, axis=-1).astype(np.int32)

    return harris_keypoints


def detect(img, n=100, detector='SIFT'):
    if detector == 'SIFT':
        return SIFT_detector(img, n)
    elif detector == 'Harris':
        return Harris_detector(img, n)
    else:
        raise ValueError


def extract_patches(img, keypoints, size=32):
    def get_patches(kps, img, size, num):
        res = torch.zeros(num, 1, size, size)
        if type(img) is np.ndarray:
            img = torch.from_numpy(img)
        h, w = img.shape      # note: for image, the x direction is the verticle, y-direction is the horizontal...
        for i in range(num):
            cx, cy = kps[i]
            cx, cy = int(cx), int(cy)
            dd = int(size/2)
            xmin, xmax = max(0, cx - dd), min(w, cx + dd )
            ymin, ymax = max(0, cy - dd), min(h, cy + dd )

            xmin_res, xmax_res = dd - min(dd,cx), dd + min(dd, w - cx)
            ymin_res, ymax_res = dd - min(dd,cy), dd + min(dd, h - cy)

            cropped_img = img[ymin: ymax, xmin: xmax]
            ch, cw = cropped_img.shape
            res[i, 0, ymin_res: ymin_res+ch, xmin_res: xmin_res+cw] = cropped_img

        return res

    n, _ = keypoints.shape
    patches = get_patches(keypoints, img, size, n)

    return patches


def extract_features():
    pass
