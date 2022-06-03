import cv2
import argparse
import numpy as np

import metrics
import keypoints
import image_utils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Middleburry2021')
parser.add_argument('--detector', type=str, default='SIFT')
parser.add_argument('--CNN', type=str, default='Deep')
args = parser.parse_args()

def main():
    # get dataloader
    dataloader = image_utils.image_dataloader(dataset=args.dataset)

    # predict disparity for each stereo pair
    for data in dataloader:
        img1, img2, left_gt_disp, right_gt_disp, scene_name = data.values()

        # detect keypoints
        img1_keypoints = keypoints.detect(img1, n=100, detector=args.detector)
        img2_keypoints = keypoints.detect(img2, n=100, detector=args.detector)

        # extract 32x32 patches around keypoints
        img1_patches = keypoints.extract_patches(img1, img1_keypoints, size=32)
        img2_patches = keypoints.extract_patches(img2, img2_keypoints, size=32)

        # extract feature from 32x32 image patch for each keypoint

        break

if __name__ == '__main__':
    main()
