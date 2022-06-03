import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pypfm import PFMLoader


class DatasetIterator:
    def __init__(self, dataset):
        self._dataset = dataset
        self._index = 0

    def __next__(self):
        if self._index < len(self._dataset):
            scene = self._dataset.scenes[self._index]
            scene_path = os.path.join(self._dataset.datadir, scene)

            # load left image
            left_img_path = os.path.join(scene_path, 'im0.png')
            left_img = cv2.imread(left_img_path)
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)

            # load right image
            right_img_path = os.path.join(scene_path, 'im0.png')
            right_img = cv2.imread(right_img_path)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            # load calibration file
            calib_path = os.path.join(scene_path, 'calib.txt')

            calib_dict = {}
            with open(calib_path) as calib_file:
                for line in calib_file:
                    name, val = line.partition('=')[::2]
                    calib_dict[name] = val

            width = int(calib_dict['width'])
            height = int(calib_dict['height'])
            loader = PFMLoader((width, height), color=False, compress=False)

            left_disp_path = os.path.join(scene_path, 'disp0.pfm')
            left_disp = np.flip(loader.load_pfm(left_disp_path), axis=0)

            right_disp_path = os.path.join(scene_path, 'disp1.pfm')
            right_disp = np.flip(loader.load_pfm(right_disp_path), axis=0)

            data = {'left_img': left_img, 'right_img': right_img,
                    'left_disp': left_disp, 'right_disp': right_disp,
                    'scene_name': scene}

            self._index += 1

            return data

        raise StopIteration


class Middleburry2021Dataset:
    def __init__(self):
        self.datadir = 'data/Middleburry2021/'
        self.scenes = sorted(os.listdir(self.datadir))

    def __len__(self):
        return len(self.scenes)

    def __iter__(self):
        return DatasetIterator(self)


def image_dataloader(dataset='Middleburry2021'):
    if dataset == 'Middleburry2021':
        dataloader = Middleburry2021Dataset()

    return dataloader


def plot_image():
    pass


def save_image():
    pass


def draw_epipolar_lines():
    pass
