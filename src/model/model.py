import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as tdist
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from .networks import *
from .loss import *


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0
        self.device = config.DEVICE
        self.val_iter = config.VAL_ITERS
        self.imagine_g_weights_path = os.path.join(config.PATH, 'g.pth')

    def load(self):
        if os.path.exists(self.imagine_g_weights_path):
            print('Loading %s Model ...' % self.name)

            g_data = torch.load(self.imagine_g_weights_path)
            self.g.load_state_dict(g_data['params'])
            self.iteration = g_data['iteration']

    def save(self, ite=None):
        print('\nSaving %s...\n' % self.name)
        if ite is not None:
            torch.save({
                'iteration': self.iteration,
                'params': self.g.state_dict()}, self.imagine_g_weights_path + '_' + str(ite))
        else:
            torch.save({
                'iteration': self.iteration,
                'params': self.g.state_dict()}, self.imagine_g_weights_path)

class Network(BaseModel):
    def __init__(self, config):
        super(Network, self).__init__('Network', config)
        
        g = TransCNN(config)

        l1_loss = nn.L1Loss()
        content_loss = PerceptualLoss()

        self.add_module('g', g)
        self.add_module('l1_loss', l1_loss)
        self.add_module('content_loss', content_loss)

        self.g_optimizer = torch.optim.Adam(params=g.parameters(), lr=config.G_LR, betas=(config.BETA1, config.BETA2))

    def process(self, data, pdata, mask, ite):
        self.iteration += 1
        self.ite = ite
        # zero optimizers
        self.g_optimizer.zero_grad()
          
        g_loss = 0
        c_loss = 0
        f_loss = 0

        output = self.g(pdata)

        # g l1 loss ##     
        g_l1_loss = self.l1_loss(output, data) * self.config.G2_L1_LOSS_WEIGHT
        c_loss = c_loss + g_l1_loss

        # g content loss #
        g_content_loss, g_mrf_loss = self.content_loss(output, data)
        g_content_loss = g_content_loss * self.config.G1_CONTENT_LOSS_WEIGHT
        g_mrf_loss = g_mrf_loss * self.config.G2_STYLE_LOSS_WEIGHT
        c_loss = c_loss + g_content_loss
        f_loss = f_loss + g_mrf_loss

        g_loss = c_loss + f_loss

        g_loss.backward()
        self.g_optimizer.step()

        logs = [
            ("l_g", g_loss.item()),
            ("l_l1", g_l1_loss.item())     
            ]
            
        return output, g_loss, logs
    

    def forward(self, input):
        output = self.g(input)
        return output
