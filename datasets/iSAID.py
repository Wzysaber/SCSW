import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import imageio
from . import transforms
import torchvision
from PIL import Image
from torchvision import transforms as T
import random

color_table = {
    (255, 127, 0): 14,  # plane
    (127, 0, 0): 9,  # small_vehicle
    (127, 127, 0): 8,  # large_vehicle
    (63, 0, 0): 1,  # ship
    (155, 100, 0): 15,  # harbor
    (127, 63, 0): 4,  # tennis_court
    (255, 0, 0): 11,  # swimming_pool
    (255, 63, 0): 3,  # baseball_diamond
    (63, 63, 0): 2,  # stroage_tank
    (63, 127, 0): 7,  # bridge
    (0, 63, 0): 6,  # Ground_Track_Field
    (191, 0, 0): 10,  # helicopter
    (191, 63, 0): 5,  # basketball_court
    (191, 127, 0): 13,  # Soccer_ball_field
    (127, 191, 0): 12  # Roundabout
}

class_list = ["_background_(BG)", 'ship(SH)', 'stroage_tank(ST)', 'baseball_diamond(BD)', 'tennis_court(TC)',
              'basketball_court(BC)',
              'Ground_Track_Field(GTF)', 'bridge(BR)', 'large_vehicle(LV)', 'small_vehicle(SV)', 'helicopter(HC)',
              'swimming_pool(SP)', 'Roundabout(RA)', 'Soccer_ball_field(SBF)', 'plane(PL)', 'harbor(HA)']


def convert_label_to_category(label, color_table):
    category_label = np.zeros(label.shape[:2], dtype=np.uint8)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            color = tuple(label[i, j])
            if color in color_table:
                category_label[i, j] = color_table[color]
    return category_label


def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str)
    return img_name_list


def load_cls_label_list(name_list_dir):
    label_dict = np.load(os.path.join(name_list_dir, 'cls_labels_onehot.npy'), allow_pickle=True).item()
    modified_dict = {}
    for key, value in label_dict.items():
        modified_dict[key] = value[1:]  # 提取除第一个元素外的所有元素
    return modified_dict


class iSAIDDataset(Dataset):
    def __init__(
            self,
            root_dir=None,
            name_list_dir=None,
            split='train',
            stage='train',
    ):
        super().__init__()

        self.root_dir = root_dir
        self.stage = stage

        self.img_dir = os.path.join(root_dir, stage, 'images')
        self.label_dir = os.path.join(root_dir, stage, 'labels')
        self.name_list_dir = os.path.join(name_list_dir, split + '.txt')

        self.name_list = load_img_name_list(self.name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):

        _img_name = self.name_list[idx]
        img_name = os.path.join(self.img_dir, _img_name + '.png')
        image = np.asarray(imageio.imread(img_name))

        if self.stage == "train":

            label_dir = os.path.join(self.label_dir, _img_name + '_instance_color_RGB.png')
            label = np.asarray(imageio.imread(label_dir))

            label = convert_label_to_category(label, color_table)

        elif self.stage == "val":

            label_dir = os.path.join(self.label_dir, _img_name + '_instance_color_RGB.png')
            label = np.asarray(imageio.imread(label_dir))
            label = convert_label_to_category(label, color_table)

        elif self.stage == "test":

            label_dir = os.path.join(self.label_dir, _img_name + '_instance_color_RGB.png')
            label = np.asarray(imageio.imread(label_dir))
            label = convert_label_to_category(label, color_table)

        elif self.stage == "test.txt":
            label = image[:, :, 0]

        return _img_name, image, label


class iSAIDClsDataset(iSAIDDataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',

                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,

                 img_fliplr=True,
                 ignore_index=255,
                 num_classes=16,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.local_crop_size = 96
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        # self.color_jittor = transforms.PhotoMetricDistortion()

        self.gaussian_blur = transforms.GaussianBlur
        self.solarization = transforms.Solarization(p=0.2)  # 进行太阳值化

        # 将标签进行转化
        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

        # 归一化
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
        ])

        self.global_view1 = T.Compose([
            # T.RandomResizedCrop(224, scale=[0.4, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            self.gaussian_blur(p=1.0),
            # self.normalize,
        ])

        self.global_view2 = T.Compose([
            T.RandomResizedCrop(self.crop_size, scale=[0.8, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            self.gaussian_blur(p=0.05),
            self.solarization,
            self.normalize,
        ])

        self.local_view = T.Compose([
            # T.RandomResizedCrop(self.local_crop_size, scale=[0.4, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            self.gaussian_blur(p=0.5),
            self.solarization,
            self.normalize,
        ])

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image):
        img_box = None
        local_image = None

        if self.aug:

            if self.rescale_range:
                image = transforms.random_scaling(image, scale_range=self.rescale_range)
            if self.img_fliplr:
                image = transforms.random_fliplr(image)
            if self.crop_size:
                image, img_box = transforms.random_crop(image, crop_size=self.crop_size, mean_rgb=[0, 0, 0],
                                                        ignore_index=self.ignore_index)

            local_image = self.local_view(Image.fromarray(image))
            image = self.global_view1(Image.fromarray(image))

        image = self.normalize(image)

        return image, local_image, img_box

    @staticmethod
    def _to_onehot(label_mask, num_classes, ignore_index):
        # label_onehot = F.one_hot(label, num_classes)

        _label = np.unique(label_mask).astype(np.int16)
        # exclude ignore index
        _label = _label[_label != ignore_index]
        # exclude background
        _label = _label[_label != 0]

        label_onehot = np.zeros(shape=(num_classes), dtype=np.uint8)
        label_onehot[_label] = 1
        return label_onehot

    def __getitem__(self, idx):

        img_name, image, _ = super().__getitem__(idx)

        pil_image = Image.fromarray(image)

        image, local_image, img_box = self.__transforms(image=image)

        cls_label = self.label_list[img_name + '_instance_color_RGB']

        if self.aug:

            crops = []
            crops.append(image)
            crops.append(self.global_view2(pil_image))
            crops.append(local_image)

            return img_name, image, cls_label, img_box, crops
        else:
            return img_name, image, cls_label


class iSAIDSegDataset(iSAIDDataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.color_jittor = transforms.PhotoMetricDistortion()

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label):
        if self.aug:
            if self.img_fliplr:
                image, label = transforms.random_fliplr(image, label)
            image = self.color_jittor(image)
            if self.crop_size:
                image, label = transforms.random_crop(image, label, crop_size=self.crop_size,
                                                      mean_rgb=[123.675, 116.28, 103.53],
                                                      ignore_index=self.ignore_index)
        '''
        if self.stage != "train":
            image = transforms.img_resize_short(image, min_size=min(self.resize_range))
        '''
        image = transforms.normalize_img(image)
        ## to chw
        image = np.transpose(image, (2, 0, 1))

        return image, label

    def __getitem__(self, idx):
        img_name, image, label = super().__getitem__(idx)

        image, label = self.__transforms(image=image, label=label)

        cls_label = self.label_list[img_name + '_instance_color_RGB']

        return img_name, image, label, cls_label
