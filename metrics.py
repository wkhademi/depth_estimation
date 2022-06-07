import numpy as np


def AAE(gt_disp, pred_disp, invalid_disp=-1.):
    gt_disp = np.where(np.isinf(gt_disp), 0., gt_disp)
    pred_disp = np.where(np.isinf(gt_disp), 0., pred_disp)
    gt_disp = np.where(pred_disp == invalid_disp, 0., gt_disp)
    pred_disp = np.where(pred_disp == invalid_disp, 0., pred_disp)

    abs_error = np.abs(gt_disp - pred_disp)
    average_abs_error = np.mean(abs_error)

    return average_abs_error


def RMSE(gt_disp, pred_disp, invalid_disp=-1.):
    gt_disp = np.where(np.isinf(gt_disp), 0., gt_disp)
    pred_disp = np.where(np.isinf(gt_disp), 0., pred_disp)
    gt_disp = np.where(pred_disp == invalid_disp, 0., gt_disp)
    pred_disp = np.where(pred_disp == invalid_disp, 0., pred_disp)

    square_error = (gt_disp - pred_disp)**2
    rmse = np.sqrt(np.mean(square_error))

    return rmse


def percent_bad_pixels(gt_disp, pred_disp, invalid_disp=-1., threshold=4):
    gt_disp = np.where(np.isinf(gt_disp), 0., gt_disp)
    pred_disp = np.where(np.isinf(gt_disp), 0., pred_disp)
    gt_disp = np.where(pred_disp == invalid_disp, 0., gt_disp)
    pred_disp = np.where(pred_disp == invalid_disp, 0., pred_disp)

    disp_error = np.abs(gt_disp - pred_disp)
    percent_bad = np.mean(np.where(disp_error > threshold, 1, 0)) * 100

    return percent_bad
