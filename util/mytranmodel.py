#!/usr/bin/env python
# encoding: utf-8
'''
transport the state_dict of source model to specific model
'''

import torch, os, sys
sys.path.append("..")
from model.single_DR_GAN_model import Generator as G_dest
from model.single_DR_GAN_model_old import Generator as G_sour

model_sour_dir = '/data5/lixinpeng/fromhjy1312/single/netG_epoch_39.pth'
model_dest_dir = '/home/lixinpeng/myDRGANv3/snapshot/bestmodel/best_single_G.pth'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class args:
	images_perID = 6
	channel_num = 3
	Nf = 320
	Nz = 50
	Np = 9

netG_sour = G_sour(args)
netG_dest = G_dest(args)

try:
	netG_sour.load_state_dict(torch.load(model_sour_dir))
	print('success!')
except:
	print('error!')

# print(torch.load(model_sour_dir).state_dict().keys())
# print(netG_sour.state_dict().keys())
# print(netG_dest.state_dict().keys())

# transport parameters
netG_dest.G_enc_convLayers = netG_sour.Genc
netG_dest.G_dec_convLayers = netG_sour.Gdec
netG_dest.G_dec_fc = netG_sour.dec_fc

# print(netG_dest.state_dict())
torch.save(netG_dest.state_dict(), model_dest_dir)