import numpy as np


def AAE(gt_disp, pred_disp):
    average_abs_error = np.mean(np.abs(gt_disp - pred_disp))

    return average_abs_error


def RMSE(gt_disp, pred_disp):
    rmse = np.sqrt(np.mean((gt_disp - pred_disp)**2))

    return RMSE


def percent_bad_pixels(gt_disp, pred_disp, threshold=2):
    disp_error = np.abs(gt_disp - pred_disp)
    percent_bad = np.mean(np.where(disp_error > threshold, 1, 0)) * 100

    return percent_bad
