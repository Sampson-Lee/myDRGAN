#!/usr/bin/env python
# encoding: utf-8

import os, datetime
import numpy as np
import torchvision.utils as vutils
import matplotlib as mpl
mpl.use('Agg')
import torch
from torch.autograd import Variable

def generate_image(dataloader, G_model, args):
    G_model.eval()
    image_number = 0
    # generate images
    if args.mode=='gensingle':
        save_dir = os.path.join(args.save_dir, 'Gensingle', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        if not os.path.isdir(save_dir): os.makedirs(save_dir)
        
        G_model.load_state_dict(torch.load(args.modeldir+'goodgen_single_G.pth'))
        for i, [batch_image, _, batch_pose_label] in enumerate(dataloader):
            batch_size = batch_image.size(0)
            noise = torch.FloatTensor(np.random.uniform(-1,1, (batch_size, args.Nz)))
            pose_code = np.zeros((batch_size, args.Np))
            pose_code[range(batch_size), batch_pose_label.tolist()] = 1
            pose_code = torch.FloatTensor(pose_code.tolist())

            if args.cuda:
                batch_image, noise, pose_code = \
                    batch_image.cuda(), noise.cuda(), pose_code.cuda()

            batch_image, noise, pose_code = \
                Variable(batch_image), Variable(noise), Variable(pose_code)

            # Generator generates images
            generated = G_model(batch_image, pose_code, noise)

            for j in range(batch_size):
                image_number += 1

                single_image_dir = os.path.join(save_dir, 'genbySimage_num{}.jpg'.format(str(image_number)))
                print('saving {}'.format(single_image_dir))
                # normalize (bool, optional): If True, shift the image to the range (0, 1)
                # here normalize is not working, and we 
                vutils.save_image(generated[j].cpu().data.mul(0.5).add(0.5), single_image_dir, normalize=True)

                single_image_dir = os.path.join(save_dir, 'realimage_num{}.jpg'.format(str(image_number)))
                print('saving {}'.format(single_image_dir))
                vutils.save_image(batch_image[j].cpu().data.mul(0.5).add(0.5), single_image_dir, normalize=True)

            batch_image_dir =  os.path.join(save_dir, 'genbySimage_batch{}to{}.jpg'.format(image_number-batch_size+1, image_number))
            vutils.save_image(generated.cpu().data, batch_image_dir, normalize=True)
            batch_image_dir =  os.path.join(save_dir, 'realimage_batch{}to{}.jpg'.format(image_number-batch_size+1, image_number))
            vutils.save_image(batch_image.cpu().data, batch_image_dir, normalize=True)
            if (image_number>100) : break;

    if args.mode=='genmulti':
        save_dir = os.path.join(args.save_dir, 'Genmulti',datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        if not os.path.isdir(save_dir): os.makedirs(save_dir)
        
        G_model.load_state_dict(torch.load(args.modeldir+'goodgen_multi_G.pth'))
        for i, [batch_image, _, batch_pose_label] in enumerate(dataloader):
            batch_size = batch_image.size(0)
            batch_size_unique = batch_size // args.images_perID
            batch_pose_label_unique = batch_pose_label[::args.images_perID]
            batch_image_unique = batch_image[::args.images_perID]
            pose_code_unique = np.zeros((batch_size_unique, args.Np))
            pose_code_unique[range(batch_size_unique), batch_pose_label_unique.tolist()] = 1
            pose_code_unique = torch.FloatTensor(pose_code_unique.tolist())         
            noise = torch.FloatTensor(np.random.uniform(-1,1, (batch_size_unique, args.Nz)))

            if args.cuda:
               batch_image, noise, pose_code_unique, batch_image_unique = \
                   batch_image.cuda(), noise.cuda(), pose_code_unique.cuda(), batch_image_unique.cuda()

            batch_image, noise, pose_code_unique, batch_image_unique = \
               Variable(batch_image), Variable(noise), Variable(pose_code_unique), Variable(batch_image_unique)

            # Generator generates images
            generated = G_model(batch_image, pose_code_unique, noise)

            for j in range(batch_size_unique):
                image_number += 1

                multi_image_dir = os.path.join(save_dir, 'genbyMimage_num{}.jpg'.format(str(image_number)))
                print('saving {}'.format(multi_image_dir))
                vutils.save_image(generated[j].cpu().data.mul(0.5).add(0.5), multi_image_dir, normalize=True)

                multi_image_dir = os.path.join(save_dir, 'realimage_num{}.jpg'.format(str(image_number)))
                print('saving {}'.format(multi_image_dir))
                vutils.save_image(batch_image_unique[j].cpu().data.mul(0.5).add(0.5), multi_image_dir, normalize=True)

            batch_image_dir =  os.path.join(save_dir, 'genbyMimage_batch{}to{}.jpg'.format(image_number-batch_size, image_number))
            vutils.save_image(generated.cpu().data, batch_image_dir, normalize=True)
            batch_image_dir =  os.path.join(save_dir, 'realimage_batch{}to{}.jpg'.format(image_number-batch_size, image_number))
            vutils.save_image(batch_image_unique.cpu().data, batch_image_dir, normalize=True)
            if (image_number>100) : break;

    return 'generate_image successfully'