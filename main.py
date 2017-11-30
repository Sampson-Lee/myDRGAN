#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
#import numpy as np
import torch
import torch.nn as nn
from model import single_DR_GAN_model as single_model
from model import multiple_DR_GAN_model as multi_model
import data.mydataset as mydataset
from train.train_single_DRGAN import train_single_DRGAN
from train.train_multiple_DRGAN import train_multiple_DRGAN
from inference.generate_image import generate_image
from inference.representation_learning import representation_learning


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='DR_GAN')
    # learning parameters
    parser.add_argument('-lr', type=float, default=0.0002, help='initial learning rate [default: 0.0002]')
    parser.add_argument('-beta1', type=float, default=0.5, help='adam optimizer parameter [default: 0.5]')
    parser.add_argument('-beta2', type=float, default=0.999, help='adam optimizer parameter [default: 0.999]')
    parser.add_argument('-epochs', type=int, default=20, help='number of epochs for train [default: 1000]')
    parser.add_argument('-batch_size', type=int, default=8, help='batch size for training [default: 8]')
    parser.add_argument('-cuda', action='store_true', default=True, help='enable the gpu')
    parser.add_argument('-device', type=str, default="0", help='the number of gpu to use')
    
    # model
    parser.add_argument('-mode', default='trainmulti', help=' trainrandom | trainsingle | trainmulti | gensingle | genmulti | idensingle | idenmulti')
    parser.add_argument('-images_perID', type=int, default=8, help='number of images per person to input to multi image DR_GAN')
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot(snapshot/{Single or Multiple}/{date}/{epoch}) [default: None]')
    
    # saving parameterss
    parser.add_argument('-save_dir', type=str, default='./snapshot', help='where to save the snapshot')
    parser.add_argument('-save_freq', type=int, default=1, help='save learned model for every "-save-freq" epoch')
    
    # data source
    parser.add_argument('-data_path', type=str, default=None, help='path to filelist')
    
    # model args
    args = parser.parse_args()
    print("Parameters:")
    for attr, value in sorted(args.__dict__.items()):
        text ="\t{}={}\n".format(attr.upper(), value)
        print(text)
        with open('{}/Parameters.txt'.format(args.save_dir),'a') as f:
            f.write(text)
    
    # model data
    if args.mode == 'trainrandom':
        print('mode is {}'.format(args.mode))
        dataset = mydataset.randomData()
        Nd, Np, Nz, channel_num = [dataset.Nd, dataset.Np, dataset.Nz, dataset.channel_num]
        print([Np, Nd])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    if args.mode == 'trainsingle':
        print('mode is {}'.format(args.mode)+'\n'+'file list is {}'.format(args.data_path))
        dataset = mydataset.multiPIE(args.data_path)
        Nd, Np, Nz, channel_num = [dataset.Nd, dataset.Np, dataset.Nz, dataset.channel_num]
        print([Np, Nd])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    if args.mode == 'trainmulti':
        print('mode is {}'.format(args.mode)+'\n'+'file list is {}'.format(args.data_path))
        if args.batch_size % args.images_perID != 0:  
            print("Please give valid combination of batch_size, images_perID");exit();
                 
        dataset = mydataset.multiPIE(args.data_path)
        Nd, Np, Nz, channel_num = [dataset.Nd, dataset.Np, dataset.Nz, dataset.channel_num]
        print([Np, Nd])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if args.mode in ['gensingle', 'genmulti', 'idensingle', 'idenmulti']:
        print('mode is {}'.format(args.mode)+'\n'+'file list is {}'.format(args.data_path))
        if args.mode in ['genmulti', 'idenmulti'] and args.batch_size % args.images_perID != 0:  
            print("Please give valid combination of batch_size, images_perID");exit();
                 
        dataset = mydataset.multiPIE(args.data_path)
        Nd, Np, Nz, channel_num = [200, 9, 50, 3] # be consistent with training
        print([Np, Nd])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # model construct
    netD = single_model.Discriminator(Nd, Np, channel_num)
    netG = single_model.Generator(Np, Nz, channel_num)
    if args.mode in ['trainmulti', 'genmulti', 'idenmulti']:
        netD = multi_model.Discriminator(Nd, Np, channel_num)
        netG = multi_model.Generator(Np, Nz, channel_num, args.images_perID)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if len(args.device.split(','))>1:
        netD = nn.DataParallel(netD)
        netG = nn.DataParallel(netG)
        
    # model resume
    if args.snapshot != None:
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            netD.load_state_dict(torch.load('{}_D.pth'.format(args.snapshot)))
            netG.load_state_dict(torch.load('{}_G.pth'.format(args.snapshot)))
            print("\nLoading successfully.")
        except:
            print("Sorry, This snapshot doesn't exist.")
            exit()
    
    # model train or inference
    if args.mode in ['trainsingle', 'trainrandom']:
        train_single_DRGAN(dataloader, Nd, Np, Nz, netD, netG, args)
    if args.mode=='trainmulti':
        train_multiple_DRGAN(dataloader, Nd, Np, Nz, netD, netG, args)
    if args.mode in ['gensingle', 'genmulti']:
        status = generate_image(dataloader, Np, Nz, netG, args)
    if args.mode in ['idensingle', 'idenmulti']:
        iden_rate = representation_learning(dataloader, netG, args)
