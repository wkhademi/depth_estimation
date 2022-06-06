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
            right_img_path = os.path.join(scene_path, 'im1.png')
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

            ndisp = int(calib_dict['ndisp'])
            vmin = int(calib_dict['vmin'])
            vmax = int(calib_dict['vmax'])

            data = {'left_img': left_img, 'right_img': right_img,
                    'left_disp': left_disp, 'right_disp': right_disp,
                    'ndisp': ndisp, 'vmin': vmin, 'vmax': vmax,
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


class Middleburry2014Dataset:
    def __init__(self):
        self.datadir = 'data/Middleburry2014/'
        self.scenes = sorted(os.listdir(self.datadir))

    def __len__(self):
        return len(self.scenes)

    def __iter__(self):
        return DatasetIterator(self)


class HW3Dataset:
    def __init__(self):
        self.datadir = 'data/HW3/'
        self.scenes = sorted(os.listdir(self.datadir))

    def __len__(self):
        return len(self.scenes)

    def __iter__(self):
        return DatasetIterator(self)


def image_dataloader(dataset='Middleburry2021'):
    if dataset == 'Middleburry2021':
        dataloader = Middleburry2021Dataset()
    elif dataset == 'Middleburry2014':
        dataloader = Middleburry2014Dataset()
    elif dataset == 'HW3':
        dataloader = HW3Dataset()
    else:
        raise ValueError

    return dataloader


def save_image():
    pass


def draw_epipolar_lines(img, lines, p):
    p = p.astype(np.int32)
    r,c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for r, pt in zip(lines, p):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img = cv2.line(img, (x0,y0), (x1,y1), color, 2)
        img = cv2.circle(img, tuple(pt), 3, color, -1)

    return img
