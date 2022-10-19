from model.networks import HTGenerator
from thop import profile
# from thop import profile
import os
from config import Config
# from options.train_options import TrainOptions
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config_path = os.path.join('/data2/zhaohaoru/code/blind_trans/checkpoints/ffhq_0209_kernal35', 'config.yml')

# load config file
config = Config(config_path)

model = HTGenerator(config)

input = torch.randn(1, 3, 256, 256)
flops, params = profile(model, inputs=(input,))

print("%.2fM" % (flops/1e6), "%.5fM" % (params/1e6))