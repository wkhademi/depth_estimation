import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

import metrics
import matching
import keypoints
import image_utils
import epipolar_geometry

from CNN import ShallowNet, DeepNet


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Middleburry2014')
parser.add_argument('--detector', type=str, default='SIFT')
parser.add_argument('--CNN', type=str, default='Deep')
parser.add_argument('--F_estimator', type=str, default='RANSAC')
parser.add_argument('--full_pipeline', action='store_true')
args = parser.parse_args()

def main():
    # get dataloader
    dataloader = image_utils.image_dataloader(dataset=args.dataset)

    if args.full_pipeline:
        # build CNN model
        if args.CNN == 'Shallow':
            trained_weight_path = 'CNN1.pth'
            CNN_model = ShallowNet().cuda()
        elif args.CNN == 'Deep':
            trained_weight_path = 'checkpoint.pth'
            CNN_model = DeepNet().cuda()
        else:
            raise ValueError

        # load pretrained weights
        trained_weight = torch.load(trained_weight_path)['state_dict']
        CNN_model.load_state_dict(trained_weight)
        CNN_model.eval()

        # predict disparity for each stereo pair
        for data in dataloader:
            img1, img2, left_gt_disp, right_gt_disp, ndisp, vmin, vmax, scene_name = data.values()

            # detect keypoints
            img1_keypoints = keypoints.detect(img1, n=200, detector=args.detector)
            img2_keypoints = keypoints.detect(img2, n=200, detector=args.detector)

            # extract 32x32 patches around keypoints
            img1_patches = keypoints.extract_patches(img1, img1_keypoints, size=32)
            img2_patches = keypoints.extract_patches(img2, img2_keypoints, size=32)

            # extract feature from 32x32 image patch for each keypoint
            img1_keypoint_features = keypoints.extract_features(CNN_model, img1_patches)
            img2_keypoint_features = keypoints.extract_features(CNN_model, img2_patches)

            # match keypoints
            C, X = matching.find_keypoint_matches(img1_keypoint_features, img2_keypoint_features)
            p, q = matching.select_n_best_matches(img1_keypoints, img2_keypoints, C, X, n=100)

            # estimate fundamental matrix from matched keypoints
            F, inlier_mask = epipolar_geometry.estimate_fundamental_matrix(p, q, method=args.F_estimator)

            # images have not been rectified yet
            if F is not None:
                # compute epipoles and epipolar lines
                l1, l2 = epipolar_geometry.compute_epipolar_lines(p, q, F, inlier_mask)
                e1, e2 = epipolar_geometry.compute_epipoles(F)

                # rectify images
                img1, img2, H1, H2 = epipolar_geometry.rectify(img1, p, img2, q, F, inlier_mask)

                # perform stereo matching
                window_size = 3
                min_disp = 0
                num_disp = ndisp
                stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                                               numDisparities = num_disp,
                                               blockSize = 9,
                                               P1 = 100,
                                               P2 = 1000,
                                               disp12MaxDiff = 1,
                                               uniquenessRatio = 10,
                                               speckleWindowSize = 100,
                                               speckleRange = 2
                                               )
                disp = stereo.compute(img1, img2).astype(np.float32)
                print(left_gt_disp)
                print(disp)

                h, w = img1.shape
                disp = cv2.warpPerspective(disp, H1, (w, h), cv2.WARP_INVERSE_MAP)

                plt.imshow(disp, vmin=vmin, vmax=vmax)
                plt.show()

    else:  # just do stereo matching
        for data in dataloader:
            img1, img2, left_gt_disp, right_gt_disp, ndisp, vmin, vmax, scene_name = data.values()

            window_size = 3
            min_disp = 16
            num_disp = ndisp
            stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                numDisparities = num_disp,
                blockSize = 9,
                P1 = 100,
                P2 = 1000,
                disp12MaxDiff = 1,
                uniquenessRatio = 10,
                speckleWindowSize = 100,
                speckleRange = 2
                )
            disp = stereo.compute(img1, img2).astype(np.float32) / 16.0
            print(disp)
            print(left_gt_disp)

            plt.imshow((disp - min_disp) / ndisp)
            plt.show()

if __name__ == '__main__':
    main()
