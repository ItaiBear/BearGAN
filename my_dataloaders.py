from OASIS.dataloaders.dataloaders import get_dataset_name
from create_panoptic import _create_panoptic_label
from create_panoptic import _read_segments
import torch
from constants import root_project_directory
import os
from PIL import Image
from torchvision import transforms as TR
import random
import numpy as np


def get_dataloaders(opt):
    #only city scapes is now supported for panoptic
    dataset_name   = get_dataset_name(opt.dataset_mode)

    dataset_train = CityscapesDataset(opt, for_metrics=False)
    dataset_val   = CityscapesDataset(opt, for_metrics=True)

    print("Created %s, size train: %d, size val: %d" % (dataset_name, len(dataset_train), len(dataset_val)))

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = opt.batch_size, shuffle = True, drop_last=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = opt.batch_size, shuffle = False, drop_last=False)

    return dataloader_train, dataloader_val


class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        opt.load_size = 512
        opt.crop_size = 512
        opt.label_nc = 34
        opt.contain_dontcare_label = True
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 2.0

        #if opt.segmentation == "panoptic":
          #opt.label_nc = 19
          #self.segments_dict = _read_segments(opt.dataroot, "val" if opt.phase == "test" or for_metrics else "train")

        opt.semantic_nc = opt.label_nc + 1    # label_nc + unknown

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels, self.paths = self.list_images()
        

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
        #label_data = np.load(os.path.join(self.paths[1], self.labels[idx]), allow_pickle=True)
        #label_data, label_format = _create_panoptic_label(os.path.join(self.paths[0], self.images[idx]), self.segments_dict)
        #label = Image.fromarray(label_data)
        if self.opt.segmentation=="panoptic":
            label_arr = np.asarray(label, dtype=np.uint32)
            mask = label_arr < 1000 
            mask = (mask * 999) + 1
            label_arr = mask * label_arr
            label = Image.fromarray(label_arr.astype(np.uint32))

        image, label = self.transforms(image, label)
        if (not self.opt.segmentation=="panoptic"):
          label = label * 255
        return {"image": image, "label": label, "name": self.images[idx]}

    def list_images(self):
        mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"
        images = []
        path_img = os.path.join(self.opt.dataroot, "leftImg8bit", mode)
        for city_folder in sorted(os.listdir(path_img)):
            cur_folder = os.path.join(path_img, city_folder)
            for item in sorted(os.listdir(cur_folder)):
                images.append(os.path.join(city_folder, item))
        labels = []
        path_lab = os.path.join(self.opt.dataroot, "gtFine", mode)
        file_extension = "_gtFine.png"
        for city_folder in sorted(os.listdir(path_lab)):
            cur_folder = os.path.join(path_lab, city_folder)
            for item in sorted(os.listdir(cur_folder)):
                if item.find("labelIds") != -1 and self.opt.segmentation=="semantic":
                    labels.append(os.path.join(city_folder, item))
                if item.find("instanceIds") != -1 and self.opt.segmentation=="panoptic":
                    labels.append(os.path.join(city_folder, item))
        
        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        '''
        for i in range(len(images)):
            assert images[i].replace("_leftImg8bit.png", "") == labels[i].replace(file_extension, ""),\
                '%s and %s are not matching' % (images[i], labels[i])
                '''
        return images, labels, (path_img, path_lab)

    def transforms(self, image, label):
        assert image.size == label.size
        # resize
        new_width, new_height = (int(self.opt.load_size / self.opt.aspect_ratio), self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label