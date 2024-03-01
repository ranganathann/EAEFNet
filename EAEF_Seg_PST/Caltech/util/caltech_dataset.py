import os
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ColorJitter, Normalize
import cv2
import numpy as np
import PIL


class caltech_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=608, input_w=960, transform=[]):
        super(caltech_dataset, self).__init__()

        assert split in ['train', 'val', 'test'], \
            'split must be "train"|"val"|"test"'  # test_day, test_night

        with open(os.path.join(os.path.join(data_dir, "splits"), "rgb_"+split+'.txt'), 'r') as f:
            self.rgb_names = [name.strip() for name in f.readlines()]
        with open(os.path.join(os.path.join(data_dir, "splits"), "thermal8_"+split+'.txt'), 'r') as f:
            self.thermal_names = [name.strip() for name in f.readlines()]

        assert len(self.rgb_names) == len(
            self.thermal_names), "Should have the same number of RBG and thermal images."
        self.data_dir = data_dir
        self.split = split
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform
        self.n_data = len(self.rgb_names)

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s' % name)
        image = np.asarray(PIL.Image.open(file_path))
        return image

    def __getitem__(self, index):
        thermal_name_list = self.thermal_names[index].split(",")
        rgb_name_list = self.rgb_names[index].split(",")

        # thermal_name_list = ["thermal8/big_bear_ONR_2022-05-08-11-23-59_thermal-02160.jpg",
        #                      "annotations/big_bear_ONR_2022-05-08-11-23-59_mask-02160.png"]
        # rgb_name_list = ["color/big_bear_ONR_2022-05-08-11-23-59_eo-01260.jpg",
        #                  "annotations/big_bear_ONR_2022-05-08-11-23-59_mask-02160.png"]

        assert thermal_name_list[1] == rgb_name_list[1], "Annotation should be the same for both RGB and thermal."
        image = self.read_image(rgb_name_list[0], 'color')
        thermal = self.read_image(thermal_name_list[0], 'thermal8')[
            :, :, 0]

        # Setup normalizations
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        normalization_rgb = Normalize(imagenet_mean, imagenet_std)
        normalization_thermal = Normalize(imagenet_mean[0], imagenet_std[0])

        # image = self.read_image(thermal_name_list[0], 'thermal8')  # TEMPORARY
        label = self.read_image(
            rgb_name_list[1], 'annotations')

        # for func in self.transform:
        #     image, thermal, label = func(image, thermal, label)
        # cj = ColorJitter()
        # image = cj(image)
        # thermal = cj(thermal))

        image = np.asarray(PIL.Image.fromarray(image).resize(
            (self.input_w, self.input_h)), dtype=np.float32).transpose((2, 0, 1))/255
        thermal = np.asarray(PIL.Image.fromarray(thermal).resize(
            (self.input_w, self.input_h)), dtype=np.float32)[np.newaxis, :]/255
        label = np.asarray(PIL.Image.fromarray(label).resize(
            (self.input_w, self.input_h), resample=PIL.Image.NEAREST), dtype=np.int64)

        # "Remove" classes 0 and 1
        label = np.clip(label-2, -1, None)
        label[label == -1] = 10

        image = normalization_rgb(torch.tensor(image)).numpy()
        thermal = normalization_thermal(
            torch.tensor(thermal[np.newaxis, :])).numpy()[0, :, :]

        return torch.tensor(image).float(), torch.tensor(thermal).float(), torch.tensor(label), rgb_name_list[1]

    def __len__(self):
        return self.n_data
