import cv2
import time
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
parser.add_argument('--stereo_matching', type=str, default='SGBM')
parser.add_argument('--window_size', type=int, default=9)
parser.add_argument('--num_disp', type=int, default=256)
parser.add_argument('--min_disp', type=int, default=0)
args = parser.parse_args()

def main():
    # get dataloader
    dataloader = image_utils.image_dataloader(dataset=args.dataset)

    if args.full_pipeline:
        avg_aae = []
        avg_rmse = []
        avg_bad2 = []
        times = []

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
            start_time = time.time()

            img1, img2, left_gt_disp, right_gt_disp, ndisp, vmin, vmax, scene_name = data.values()

            h, w = img1.shape

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

            # rectify images
            img1, img2, H1, H2 = epipolar_geometry.rectify(img1, p, img2, q, F, inlier_mask, h, w)

            # perform stereo matching
            left_disp, right_disp = matching.stereo_matching(img1, img2, method=args.stereo_matching,
                                                             window_size=args.window_size, num_disp=ndisp,
                                                             min_disp=args.min_disp)

            times.append(time.time() - start_time)

            # warp disparity maps back to align with original image
            left_disp = cv2.warpPerspective(left_disp, np.linalg.inv(H1), (w, h))
            right_disp = cv2.warpPerspective(right_disp, np.linalg.inv(H2), (w, h))

            plt.imshow(left_disp, vmin=vmin, vmax=vmax)
            plt.savefig('data/%s/%s/%s_%s%s_left_disp.png'%(args.dataset, scene_name, args.detector, args.stereo_matching, args.window_size), bbox_inches='tight')

            plt.imshow(right_disp, vmin=vmin, vmax=vmax)
            plt.savefig('data/%s/%s/%s_%s%s_right_disp.png'%(args.dataset, scene_name, args.detector, args.stereo_matching, args.window_size), bbox_inches='tight')

            # compute evaluation metrics
            aae = 0.5 * (metrics.AAE(left_gt_disp, left_disp) + metrics.AAE(right_gt_disp, right_disp))
            rmse = 0.5 * (metrics.RMSE(left_gt_disp, left_disp) + metrics.RMSE(right_gt_disp, right_disp))
            bad2 = 0.5 * (metrics.percent_bad_pixels(left_gt_disp, left_disp) + metrics.percent_bad_pixels(right_gt_disp, right_disp))

            avg_aae.append(aae)
            avg_rmse.append(rmse)
            avg_bad2.append(bad2)

        print('Average Runtime: %.2f'%np.mean(times))
        print('Absolute Average Error: %f'%np.mean(avg_aae))
        print('Root-Mean-Square Error: %f'%np.mean(avg_rmse))
        print('Bad2.0: %f'%np.mean(avg_bad2))
    else:  # just do stereo matching
        avg_aae = []
        avg_rmse = []
        avg_bad2 = []
        times = []

        for data in dataloader:
            start_time = time.time()

            img1, img2, left_gt_disp, right_gt_disp, ndisp, vmin, vmax, scene_name = data.values()

            # stereo matching
            left_disp, right_disp = matching.stereo_matching(img1, img2, method=args.stereo_matching,
                                                             window_size=args.window_size, num_disp=ndisp,
                                                             min_disp=args.min_disp)

            times.append(time.time() - start_time)

            plt.imshow(img1)
            plt.savefig('data/%s/%s/left.png'%(args.dataset, scene_name), bbox_inches='tight')

            plt.imshow(img2)
            plt.savefig('data/%s/%s/right.png'%(args.dataset, scene_name), bbox_inches='tight')

            aae = 0.5 * (metrics.AAE(left_gt_disp, left_disp) + metrics.AAE(right_gt_disp, right_disp))
            rmse = 0.5 * (metrics.RMSE(left_gt_disp, left_disp) + metrics.RMSE(right_gt_disp, right_disp))
            bad2 = 0.5 * (metrics.percent_bad_pixels(left_gt_disp, left_disp) + metrics.percent_bad_pixels(right_gt_disp, right_disp))

            avg_aae.append(aae)
            avg_rmse.append(rmse)
            avg_bad2.append(bad2)

        print('Average Runtime: %.2f'%np.mean(times))
        print('Absolute Average Error: %f'%np.mean(avg_aae))
        print('Root-Mean-Square Error: %f'%np.mean(avg_rmse))
        print('Bad2.0: %f'%np.mean(avg_bad2))

if __name__ == '__main__':
    main()
