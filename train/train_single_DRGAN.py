#!/usr/bin/env python
# encoding: utf-8

import os, datetime
import numpy as np
import torchvision.utils as vutils
import matplotlib as mpl
mpl.use('Agg')
import torch
from torch import nn, optim
from torch.autograd import Variable
from util.mylog import log_learning, plot_loss, imgtogif

def train_single_DRGAN(dataloader, Nd, Np, Nz, D_model, G_model, args):
    if args.cuda:
        D_model.cuda()
        G_model.cuda()

    D_model.train()
    G_model.train()

    lr_Adam    = args.lr
    beta1_Adam = args.beta1
    beta2_Adam = args.beta2
    eps = 10**-300

    optimizer_D = optim.Adam(D_model.parameters(), lr = lr_Adam, betas=(beta1_Adam, beta2_Adam))
    optimizer_G = optim.Adam(G_model.parameters(), lr = lr_Adam, betas=(beta1_Adam, beta2_Adam))
    CE_loss = nn.CrossEntropyLoss()
    BCE_Loss = nn.BCELoss()

    loss_log = []; g_loss_log = []; d_loss_log = [];
    ratio = 2
    # D_acc = 0
    save_dir = os.path.join(args.save_dir, 'Single',datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    
    epo_list = range(1,args.epochs+1)
    for epoch in range(1,args.epochs+1):
        steps = 0
        for i, [batch_image, batch_id_label, batch_pose_label] in enumerate(dataloader):
            D_model.zero_grad()
            G_model.zero_grad()
            batch_size = batch_image.size(0)
            batch_real_label = torch.ones(batch_size)
            batch_sys_label = torch.zeros(batch_size)

            # generate noise code and pose code, label: LongTensor, input: FloatTensor
            noise = torch.FloatTensor(np.random.uniform(-1,1, (batch_size, Nz)))
            pose_code_label  = np.random.randint(Np, size=batch_size) # get a list of int in range(Np)
            pose_code = np.zeros((batch_size, Np))
            pose_code[range(batch_size), pose_code_label] = 1
            pose_code_label = torch.LongTensor(pose_code_label.tolist())
            pose_code = torch.FloatTensor(pose_code.tolist())
            # use cuda for label and input
            if args.cuda:
                batch_image, batch_id_label, batch_pose_label, batch_real_label, batch_sys_label = \
                    batch_image.cuda(), batch_id_label.cuda(), batch_pose_label.cuda(), batch_real_label.cuda(), batch_sys_label.cuda()

                noise, pose_code, pose_code_label = \
                    noise.cuda(), pose_code.cuda(), pose_code_label.cuda()
            # use Variable for label and input
            batch_image, batch_id_label, batch_pose_label, batch_real_label, batch_sys_label = \
                Variable(batch_image), Variable(batch_id_label), Variable(batch_pose_label), Variable(batch_real_label), Variable(batch_sys_label)

            noise, pose_code, pose_code_label = \
                Variable(noise), Variable(pose_code), Variable(pose_code_label)
            
            # generator forward
            generated = G_model(batch_image, pose_code, noise) #forward
            steps += 1

            # D and G learn alternately
            # D accuracy is related with ratio, the D_acc is higher, the ratio is larger
            # if D_acc<(1-0.5**1): ratio=2**1+1 # 1:2
            # elif D_acc<(1-0.5**2): ratio=2**2+1 # 1:4
            # elif D_acc<(1-0.5**3): ratio=2**3+1 # 1:8
            # elif D_acc<(1-0.5**4): ratio=2**4+1 # 1:16
            # if epoch<2**1+1: ratio=2**0+1 # 1:1
            # elif epoch<2**2+1: ratio=2**1+1 # 1:2
            # elif epoch<2**3+1: ratio=2**2+1 # 1:4
            # elif epoch<2**4+1: ratio=2**3+1 # 1:8
            if epoch<2**1+3: ratio=1+1 # 1:1
            elif epoch<2**2+3: ratio=2+1 # 1:2
            elif epoch<2**3+3: ratio=3+1 # 1:3
            elif epoch<2**4+3: ratio=4+1 # 1:4
            elif epoch<2**5+3: ratio=5+1 # 1:5
            elif epoch<2**6+3: ratio=6+1 # 1:6
         
            # Discriminator learning as followed
            if i % ratio == 0:
                real_output = D_model(batch_image)
                syn_output = D_model(generated.detach()) # for D, we do not update the parameters to Generator and detach gradient
                                    
                # three loss: L_d_id, L_d_gan, L_d_pose
                # we do not use sysimg classification loss to update D
                L_d_id    = CE_loss(real_output[:, :Nd], batch_id_label)
                L_d_gan   = BCE_Loss(real_output[:, Nd].sigmoid(), batch_real_label) + BCE_Loss(syn_output[:, Nd].sigmoid(), batch_sys_label)
                L_d_pose  = CE_loss(real_output[:, Nd+1:], batch_pose_label)
                d_loss = L_d_gan + L_d_id + L_d_pose

                d_loss.backward()
                optimizer_D.step()
                loss_dict = {'d_loss':d_loss.data[0], 'L_d_id':L_d_id.data[0], 'L_d_gan':L_d_gan.data[0], 'L_d_pose':L_d_pose.data[0]}
                log_learning(epoch, steps, 'SingleDRGAN-D', save_dir, d_loss.data[0], loss_dict)
                # judge whether discriminator is strong or not after D update
                # D_acc = cal_acc(real_output, syn_output, batch_id_label, batch_pose_label, Nd)
                # print(D_acc)
                
            # Generator learning as followed
            else:
                syn_output=D_model(generated)   # no detach here

                L_g_id    = CE_loss(syn_output[:, :Nd], batch_id_label)
                L_g_gan   = BCE_Loss(syn_output[:, Nd].sigmoid().clamp(min=eps), batch_real_label) # for G, we use real_label in loss
                L_g_pose  = CE_loss(syn_output[:, Nd+1:], pose_code_label)

                g_loss = L_g_gan + L_g_id + L_g_pose

                g_loss.backward()
                optimizer_G.step()                
                loss_dict = {'g_loss':g_loss.data[0], 'L_g_id':L_g_id.data[0], 'L_g_gan':L_g_gan.data[0], 'L_g_pose':L_g_pose.data[0]}
                log_learning(epoch, steps, 'SingleDRGAN-G', save_dir, g_loss.data[0], loss_dict)
                
        # save loss in each epoch
        loss_log.append([d_loss.data[0], g_loss.data[0]])
        g_loss_log.append([L_g_gan.data[0], L_g_id.data[0], L_g_pose.data[0]])
        d_loss_log.append([L_d_gan.data[0], L_d_id.data[0], L_d_pose.data[0]])
        # save model in some epoches
        if epoch % args.save_freq == 0:
            save_path_D = os.path.join(save_dir,'epoch{}_D.pth'.format(epoch))
            torch.save(D_model.state_dict(), save_path_D)
            save_path_G = os.path.join(save_dir,'epoch{}_G.pth'.format(epoch))
            torch.save(G_model.state_dict(), save_path_G)
        #  save generated images in some epoches
        if epoch % args.save_freq == 0:
            save_gen_path = os.path.join(save_dir, 'epoch{}_genbySimage.jpg'.format(epoch))
            vutils.save_image(generated.cpu().data, save_gen_path, normalize=True)
            save_real_path = os.path.join(save_dir, 'epoch{}_realimage.jpg'.format(epoch))
            vutils.save_image(batch_image.cpu().data, save_real_path, normalize=True)

    # plot loss of G and D after training
    loss_log = np.array(loss_log); g_loss_log = np.array(g_loss_log); d_loss_log = np.array(d_loss_log);
    plot_loss(save_dir+'/loss.png', epo_list, {'d_loss':loss_log[:,0], 'g_loss':loss_log[:,1]})
    plot_loss(save_dir+'/loss_g.png', epo_list, {'L_g_gan':g_loss_log[:,0], 'L_g_id':g_loss_log[:,1], 'L_g_pose':g_loss_log[:,2]})
    plot_loss(save_dir+'/loss_d.png', epo_list, {'L_d_gan':d_loss_log[:,0], 'L_d_id':d_loss_log[:,1], 'L_d_pose':d_loss_log[:,2]})
    imgtogif(save_dir, (args.epochs/args.save_freq))

def cal_acc(real_output, syn_output, id_label_tensor, pose_label_tensor, Nd):
    """
    we should improve here
    """
    _, id_real_ans = torch.max(real_output[:, :Nd], 1) # return (max, max_indices)
    _, pose_real_ans = torch.max(real_output[:, Nd+1:], 1)
    _, id_syn_ans = torch.max(syn_output[:, :Nd], 1)

    id_real_precision = (id_real_ans==id_label_tensor).type(torch.FloatTensor).sum() / real_output.size()[0]
    pose_real_precision = (pose_real_ans==pose_label_tensor).type(torch.FloatTensor).sum() / real_output.size()[0]
    gan_real_precision = (real_output[:,Nd].sigmoid()>=0.5).type(torch.FloatTensor).sum() / real_output.size()[0]
    gan_syn_precision = (syn_output[:,Nd].sigmoid()<0.5).type(torch.FloatTensor).sum() / syn_output.size()[0]

    total_precision = (id_real_precision+pose_real_precision+gan_real_precision+gan_syn_precision)/4

    # Variable(FloatTensor) -> Float
    total_precision = total_precision.data[0]
    return total_precision
