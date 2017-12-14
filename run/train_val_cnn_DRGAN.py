#!/usr/bin/env python
# encoding: utf-8

import os, datetime, sys, random
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

sys.path.append("..")
from util.myacc import cal_acc_id
from util.mylog import plot_loss
import data.mydataset as mydataset

class CNNenc(nn.Module):
    """Compare supervised CNN with semi-supervised encoder"""
    def __init__(self, args):
        super(CNNenc, self).__init__()
        CNN_enc_convLayers = [
            nn.Conv2d(args.channel_num, 32, 3, 1, 1, bias=False), # Bxchx96x96 -> Bx32x96x96
            nn.BatchNorm2d(32),
            nn.ELU(), 
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), # Bx32x96x96 -> Bx64x96x96
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx64x96x96 -> Bx64x97x97
            nn.Conv2d(64, 64, 3, 2, 0, bias=False), # Bx64x97x97 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx128x48x48
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx128x48x48 -> Bx128x49x49
            nn.Conv2d(128, 128, 3, 2, 0, bias=False), #  Bx128x49x49 -> Bx128x24x24
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 96, 3, 1, 1, bias=False), #  Bx128x24x24 -> Bx96x24x24
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.Conv2d(96, 192, 3, 1, 1, bias=False), #  Bx96x24x24 -> Bx192x24x24
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx192x24x24 -> Bx192x25x25
            nn.Conv2d(192, 192, 3, 2, 0, bias=False), # Bx192x25x25 -> Bx192x12x12
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.Conv2d(192, 128, 3, 1, 1, bias=False), # Bx192x12x12 -> Bx128x12x12
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), # Bx128x12x12 -> Bx256x12x12
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx256x12x12 -> Bx256x13x13
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x13x13 -> Bx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False), # Bx256x6x6 -> Bx160x6x6
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.Conv2d(160, args.Nf, 3, 1, 1, bias=False), # Bx160x6x6 -> Bx320x6x6
            nn.BatchNorm2d(args.Nf),
            nn.ELU(),
            nn.AvgPool2d(6, stride=1), #  Bx320x6x6 -> Bx320x1x1
        ]

        self.CNN_enc_convLayers = nn.Sequential(*CNN_enc_convLayers)
        
        self.fc = nn.Linear(args.Nf, args.Nd)

    def forward(self, input, extract=False):
        x = self.CNN_enc_convLayers(input)

        x = x.squeeze(2)
        x = x.squeeze(2)
        if extract: return x
        # 全結合
        x = self.fc(x) # Bx320 -> B x Nd

        return x

def train_cnn(train_loader, model, optimizer, args):
    """Train cnn and calculate accuracy to identity"""
    CE_loss = nn.CrossEntropyLoss()
    end = time.time()
    for i, [batch_image, batch_id_label, _] in enumerate(train_loader):

        if args.cuda: batch_image, batch_id_label=batch_image.cuda(), batch_id_label.cuda()
        batch_image, batch_id_label = Variable(batch_image), Variable(batch_id_label)

        # compute output
        output_id = model(batch_image)
        loss = CE_loss(output_id, batch_id_label)
        acc = cal_acc_id(output_id.data, batch_id_label)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

       