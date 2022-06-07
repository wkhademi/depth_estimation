import numpy as np


def AAE(gt_disp, pred_disp, invalid_disp=-1.):
    cond = np.logical_or(np.isinf(gt_disp), pred_disp == invalid_disp)
    gt_disp = np.where(cond, 0., gt_disp)
    pred_disp = np.where(cond, 0., pred_disp)

    abs_error = np.abs(gt_disp - pred_disp)
    average_abs_error = np.mean(abs_error)

    return average_abs_error


def RMSE(gt_disp, pred_disp, invalid_disp=-1.):
    cond = np.logical_or(np.isinf(gt_disp), pred_disp == invalid_disp)
    gt_disp = np.where(cond, 0., gt_disp)
    pred_disp = np.where(cond, 0., pred_disp)

    square_error = (gt_disp - pred_disp)**2
    rmse = np.sqrt(np.mean(square_error))

    return rmse


def percent_bad_pixels(gt_disp, pred_disp, invalid_disp=-1., threshold=4):
    cond = np.logical_or(np.isinf(gt_disp), pred_disp == invalid_disp)
    invalid_pixels = np.where(cond, 1., 0.)
    num_invalid_pixels = np.sum(invalid_pixels)
    H, W = gt_disp.shape
    num_pixels = H*W
    num_valid_pixels = num_pixels - num_invalid_pixels

    gt_disp = np.where(cond, 0., gt_disp)
    pred_disp = np.where(cond, 0., pred_disp)

    disp_error = np.abs(gt_disp - pred_disp)
    percent_bad = (np.sum(np.where(disp_error > threshold, 1, 0)) / num_valid_pixels) * 100

    return percent_bad
