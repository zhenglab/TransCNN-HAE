import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .dataset import Dataset
from .model.model import Network
from .utils import Progbar, create_dir, stitch_images, imsave, PositionEmbeddingSine, PatchPositionEmbeddingSine
from PIL import Image
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from .metrics import PSNR

class TransCNN_HAE():
    def __init__(self, config):
        self.config = config
        self.model_name = 'Network'
        self.Model = Network(config).to(config.DEVICE)
  
        self.psnr = PSNR(255.0).to(config.DEVICE)

        self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.NOISE_TRAIN_FLIST,noise_aux=config.NOISE_TRAIN_AUX, mask_flist=config.MASK_FLIST, batchsize=config.BATCH_SIZE, augment=False, training=True)
        self.val_dataset = Dataset(config, config.VAL_FLIST, config.NOISE_VAL_FLIST, mask_flist=config.MASK_FLIST, batchsize=1, augment=False, training=False)
        self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.val_iterator = config.VAL_ITERS
        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results'+ str(self.val_iterator))
        
        self.log_file = os.path.join(config.PATH, 'log-' + self.model_name + '.txt')
        self.testlog_file = os.path.join(config.PATH, 'log-' + 'test' + '.txt')
       
        self.writer = SummaryWriter(os.path.join(config.PATH, 'runs'))
                
        
    def load(self):
        self.Model.load()
        
    def save(self, ite=None):
        self.Model.save(ite=ite)

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=False,
            shuffle=True,
            pin_memory=True
        )

        epoch = 0
        keep_training = True

        max_iter = int(self.config.MAX_ITERS)
        total = len(self.train_dataset)

        while(keep_training):
            epoch += 1
            
            probar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter', 'mean_gate', 'max_gate', 'min_gate'])
            
            ite = self.Model.iteration


            for it in train_loader:

                self.Model.train()

                data, pdata, mask = self.cuda(*it)

                output, g_loss, logs = self.Model.process(data, pdata, mask, ite)
                
                psnr = self.psnr(self.postprocess(data), self.postprocess(output))
                
                ite = self.Model.iteration
                
                # ------------------------------------------------------------------------------------
                # end training
                
                if ite > max_iter:
                    keep_training = False
                    break

                # ------------------------------------------------------------------------------------
                # save log & sample & eval & save model
                logs.append(('psnr', psnr.item()))
                    
                logs = [("epoch", epoch), ("iter", ite)] + logs
                self.writer.add_scalars('Generator', {'domaink': g_loss}, epoch)
                self.writer.add_scalars('Detail', self.log2dict(logs), epoch)
                
                # progbar
                probar.add(len(data), values=[x for x in logs])

                if self.config.INTERVAL and ite % self.config.INTERVAL == 0:
                    self.log(logs)
                    self.sample()
                    self.save()

                if self.config.SAVE_INTERAL and ite % self.config.SAVE_INTERAL == 0:
                    self.save(ite=ite)

        print('\nEnd trainging...')
        self.writer.close()
    
    def log2dict(self, logs):
        dict = {}
        for i in range(2, len(logs)):
            dict[logs[i][0]] = logs[i][1]
        return dict
    
    def test(self):
        test_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            drop_last=True
        )

        create_dir(self.results_path)
        
        input_data = os.path.join(self.results_path, 'input')
        masks = os.path.join(self.results_path, 'mask_gt')
        state_results = os.path.join(self.results_path, 'output')
        mask_results = os.path.join(self.results_path, 'outputmask')
        gt = os.path.join(self.results_path, 'gt')
        
        create_dir(input_data)
        create_dir(masks)
        create_dir(state_results)
        create_dir(mask_results)
        create_dir(gt)

        total = len(self.val_dataset)

        index = 0

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        
        for it in test_loader:
            
            # file name
            name = self.val_dataset.load_name(index)
            index += 1

            data, pdata, mask = self.cuda(*it)

            output = self.Model(pdata)
            
            data = self.postprocess_re(data)[0]
            pdata = self.postprocess_re(pdata)[0]
            mask = self.postprocess(mask)[0]
            output = self.postprocess_re(output)[0]
            
            imsave(data, os.path.join(gt, name))
            imsave(pdata, os.path.join(input_data, name))
            imsave(mask, os.path.join(masks, name))
            imsave(output, os.path.join(state_results, name))

            print(index, name)
            
        print('\nEnd test....')
        
    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\r\n' % ' '.join([str(item[1]) for item in logs]))

    def log_sample(self, logs):
        with open(self.testlog_file, 'a') as f:
            f.write('%s\r\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess_re(self, img):
        img = (img + 1.0) / 2.0
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()  

    def postprocess(self, img):
        # img = (img + 1.0) / 2.0
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()  

    def sample(self):

        ite = self.Model.iteration
        its = next(self.sample_iterator)
        
        data, pdata, mask = self.cuda(*its)
        
        output  = self.Model(pdata)
        psnr = self.psnr(self.postprocess(data), self.postprocess(output))

        logs = [
            ("iter", ite),
            ('psnr', psnr.item())
        ]

        self.log_sample(logs)


        # draw sample image
        image_per_row = 1
        images = stitch_images(
            self.postprocess_re(pdata),
            self.postprocess_re(output),
            self.postprocess_re(data),
            img_per_row = image_per_row
        )

        path = os.path.join(self.samples_path)
        name = os.path.join(path, str(ite).zfill(5) + '.png')
        create_dir(path)

        print('\nSaving sample images...' + name)
        images.save(name)