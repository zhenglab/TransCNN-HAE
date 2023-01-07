import os 
import glob
import scipy
import torch
import random
import numpy as np
import cv2
import torchvision.transforms.functional as F

import torch.nn.functional as FF

from torch.utils.data import DataLoader
from PIL import Image
from imageio import imread
from .utils import * 
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .masks import Masks


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, input_flist, noise_flist, noise_aux=None, mask_flist=None, batchsize=None, augment=True, training=True):
        super(Dataset, self).__init__()
        getFileList = lambda f: open(f, 'rt').read().splitlines() \
            if os.path.isfile(f) else glob.glob(os.path.join(f, '*.JPG')) + glob.glob(os.path.join(f, '*.jpg'))
        self.input_size = config.INPUT_SIZE
        self.center = config.CENTER
        self.model = config.MODEL
        self.augment = augment
        self.training = training
        self.data_file = getFileList(input_flist)
        self.noise_file = getFileList(noise_flist)

        if len(self.noise_file) > len(self.data_file):
            self.noise_file = self.noise_file[:len(self.data_file)]
        else:
            t = len(self.data_file) // len(self.noise_file) + 1
            self.noise_file = self.noise_file * t
            self.noise_file = self.noise_file[:len(self.data_file)]
        if noise_aux is not None:
            self.file_aux = getFileList(noise_aux)
            np.random.shuffle(self.file_aux)
            if len(self.file_aux) > len(self.data_file) // 2:
                self.noise_file = self.noise_file[:len(self.data_file) // 2]
            else:
                self.noise_file = self.noise_file[:len(self.data_file)-len(self.file_aux)]

            self.noise_file += self.file_aux[:len(self.data_file) - len(self.noise_file)]
            np.random.shuffle(self.noise_file)

        self.mask_type = config.MASK_TYPE
        if self.training == False:
            self.mask_file = getFileList(mask_flist)
        self.side = config.SIDE
        self.mean = config.MEAN
        self.std = config.STD
        self.count = 0
        self.pos = None
        self.batchsize = batchsize
        self.catmask = config.CATMASK
        self.datatype = config.DATATYPE
        if self.datatype == 2:
            self.scence_width = 512
            self.scence_height = 256
        self.known_mask = not training
 
    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, index):
        if self.training == True:
            item = self.load_train_item(index)
        else:
            if self.mask_type == 'pollute':
                item = self.pollute_load_item(index)
            else:
                item = self.load_test_item(index)
        return item

    def resize(self, img, width, height):
        img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
        return img

    def load_name(self, index):
        name = self.data_file[index]
        return os.path.basename(name)
    
    def load_train_item(self, index):
        data = imread(self.data_file[index])
        if self.count == 0:
            self.noise = imread(self.noise_file[index])
            if self.mask_type == 'freeform':
                self.mask = free_form_mask(h=self.input_size, w=self.input_size)
            if self.mask_type == 'rect':
                self.mask = generate_rectangle(h=self.input_size, w=self.input_size)
            if self.mask_type == 'irregular':
                self.mask = Masks(h=self.input_size, w=self.input_size)
            self.coin = torch.rand(1)[0]
        
        self.seq = index
        
        if len(data.shape) == 2:
            data = data[:, :, np.newaxis]
            data = data.repeat(3, axis=2)

        data = self.resize(data, self.input_size, self.input_size)
        # mask = self.resize(mask, self.input_size, self.input_size)
        
        self.count += 1
        if self.count == self.batchsize:
            self.count = 0

        mask = self.mask
        noise = self.noise

        if len(noise.shape) == 2:
            noise = noise[:, :, np.newaxis]
            noise = noise.repeat(3, axis=2) 

        noise = self.resize(noise, self.input_size, self.input_size)
        mask_tensor = self.mask_tensor(mask)
        mask_soft = self.priority_mask(1 - mask_tensor) + mask_tensor
        mask_soft = torch.squeeze(mask_soft, dim=0)

        coin = self.coin

        mask_used = mask_soft

        if self.augment and np.random.binomial(1, 0.5) > 0:
            data = data[:, ::-1, ...]

        data = self.to_tensor(data)
        noise = self.to_tensor(noise)

        input_data = data * (1 - mask_used) + noise * mask_used

        return data, input_data, mask_tensor

    def load_test_item(self, index): 
        data = imread(self.data_file[index])
        noise = imread(self.noise_file[index])
        if len(data.shape) == 2:
            data = data[:, :, np.newaxis]
            data = data.repeat(3, axis=2)
        if len(noise.shape) == 2:
            noise = noise[:, :, np.newaxis]
            noise = noise.repeat(3, axis=2)   
        data = cv2.resize(data, (self.input_size, self.input_size))
        noise = cv2.resize(noise, (self.input_size, self.input_size))
        mask = imread(self.mask_file[index])

        h, w = data.shape[:2]
        grid = 4
        data = data[:h // grid * grid, :w // grid * grid, :]
        noise = noise[:h // grid * grid, :w // grid * grid, :] 
        mask = mask[:h // grid * grid, :w // grid * grid]

        mask_tensor = self.mask_tensor(mask)
        # mask_used = mask_tensor
        mask_used = self.priority_mask(1 - mask_tensor) + mask_tensor
        mask_used = torch.squeeze(mask_used, dim=0)

        data = self.to_tensor(data)
        noise = self.to_tensor(noise)

        input_data = data * (1 - mask_used) + noise * mask_used        
        return data, input_data, mask_tensor 

    def pollute_load_item(self, index):
        data = imread(self.data_file[index])
        noise = imread(self.noise_file[index])
        if len(data.shape) == 2:
            data = data[:, :, np.newaxis]
            data = data.repeat(3, axis=2)
        if len(noise.shape) == 2:
            noise = noise[:, :, np.newaxis]
            noise = noise.repeat(3, axis=2)

        data = cv2.resize(data, (self.input_size, self.input_size))
        mask = generate_graffiti(self.input_size, self.input_size, noise)

        mask_tensor = self.mask_tensor(mask)
        mask_used = self.priority_mask(1 - mask_tensor) + mask_tensor
        mask_used = torch.squeeze(mask_used, dim=0)

        data = data / 127.5 - 1
        noise = noise /127.5 - 1
        data = self.to_tensor(data)
        noise = self.to_tensor(noise)

        input_data = data * (1 - mask_used) + noise * mask_used
        
        return data, input_data, mask_tensor
        
    def img_resize(self, img, width, height, centerCrop=False):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)

        return img

    
    def priority_mask(self, mask, hsize=15, sigma=1/40, iters=4):
        mask = mask[:, :, :, np.newaxis].permute(0, 3, 1, 2)
        kernel = gauss_kernel(hsize, sigma)
        kernel = torch.from_numpy(kernel).permute(2,3,0,1) 
        init = 1 - mask
        mask_priority = torch.ones(mask.shape)
        for i in range(iters):
            init = same_padding(init, [hsize, hsize], [1, 1], [1, 1])
            mask_priority = FF.conv2d(init, kernel, stride=1)
            mask_priority = mask_priority * mask
            init = mask_priority + (1-mask)
        return mask_priority 

    

    def cpimage(self, data):
        if self.known_mask:
            # print(" seq: ",self.seq," mask file: ",self.mask_file[self.seq])
            mask = imread(self.test_mask[self.seq])
            rc, pos = self.dealimage(data, mask)
            self.pos = pos
        rc, pos, mask = random_crop(data, int(data.shape[1]/2), self.datatype, self.count, self.pos, self.known_mask)
        # rc, pos, mask = center_crop(data, int(data.shape[1]/2))
        self.pos = pos
        return rc, pos, mask
    

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        if isinstance(flist, str):
            # print(flist)
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png')) + list(glob.glob(flist + '/*.JPEG'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                # except:
                    # print(11, flist)
                #    return [flist]
        
        return []

    def to_tensor(self, img):
        img = torch.from_numpy(img)
        img = (img/127.5 - 1.0).to(memory_format=torch.contiguous_format).permute(2,0,1).float()
        return img

    def mask_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def map_to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).long()
        return img_t


    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item