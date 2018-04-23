#!/usr/bin/env python
# encoding: utf-8

import os, datetime, sys, random
import numpy as np
import torchvision.utils as vutils
import matplotlib as mpl
mpl.use('Agg')
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

sys.path.append("..")
from util.mylog import log_learning, plot_loss, imgtogif
#from util.myacc import cal_acc
from util.mybuffer import ImageHistoryBuffer
from model.weights import init_weights

epoch_set = [0, 6, 12, 18, 24, 30, 36]
ratio_set = [2**power+1 for power in range(7)]

def train_single_DRGAN(dataloader, D_model, G_model, args):
    batch_size, Np, Nd, Nz = args.batch_size, args.Np, args.Nd, args.Nz
    buffer_img = ImageHistoryBuffer((0, args.channel_num, args.img_size, args.img_size),
                                              args.batch_size * 10, args.batch_size)
    D_model.train()
    G_model.train()

    lr_Adam    = args.lr
    beta1_Adam = args.beta1
    beta2_Adam = args.beta2
    eps = 10**-300

    optimizer_D = optim.Adam(D_model.parameters(), lr = lr_Adam, betas=(beta1_Adam, beta2_Adam))
    optimizer_G = optim.Adam(G_model.parameters(), lr = lr_Adam, betas=(beta1_Adam, beta2_Adam))
    scheduler_D = lr_scheduler.MultiStepLR(optimizer_D, milestones=[30, 35], gamma=0.1)
    CE_loss = nn.CrossEntropyLoss()
    BCE_Loss = nn.BCELoss()
    
    loss_log = []; g_loss_log = []; d_loss_log = [];
    ratio = 2
    totalsteps = len(dataloader); totalepochs = args.epochs;
    # D_acc = 0
    save_dir = os.path.join(args.save_dir, 'Single',datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.isdir(save_dir): os.makedirs(save_dir)

    print("Parameters:")
    with open('{}/Parameters.txt'.format(save_dir),'a') as f:
        for attr, value in sorted(args.__dict__.items()):
            text ="{}={}\n".format(attr.upper(), value)
            print(text)
            f.write(text)
        print(text)
        f.write(text)
        for epo, rat in zip(epoch_set, ratio_set):
            text = 'epoch '+str(epo)+' - '+'ratio 1:'+str(rat-1)+'\n'
            print(text)
            f.write(text)

    epo_list = range(args.epochs)
    for epoch in epo_list:
        steps = 0
        sample_index = random.randint(0,len(dataloader))
        if args.use_allhistory_epoch: scheduler_D.step()
        print('learning rate is {}'.format(optimizer_D.param_groups[0]['lr']))
        if epoch in epoch_set:
            ratio = ratio_set[epoch_set.index(epoch)]
            print('ratio is 1:{} now! \n'.format(ratio-1))
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
            batch_pose_code = np.zeros((batch_size, Np))
            batch_pose_code[range(batch_size), batch_pose_label] = 1
            batch_pose_code = torch.FloatTensor(pose_code.tolist())
            # use cuda for label and input
            if args.cuda:
                batch_image, batch_id_label, batch_pose_label, batch_real_label, batch_sys_label = \
                    batch_image.cuda(), batch_id_label.cuda(), batch_pose_label.cuda(), batch_real_label.cuda(), batch_sys_label.cuda()

                noise, pose_code, pose_code_label, batch_pose_code = \
                    noise.cuda(), pose_code.cuda(), pose_code_label.cuda(), batch_pose_code.cuda()
            # use Variable for label and input
            batch_image, batch_id_label, batch_pose_label, batch_real_label, batch_sys_label = \
                Variable(batch_image), Variable(batch_id_label), Variable(batch_pose_label), Variable(batch_real_label), Variable(batch_sys_label)

            noise, pose_code, pose_code_label, batch_pose_code = \
                Variable(noise), Variable(pose_code), Variable(pose_code_label), Variable(batch_pose_code)
            
            # generator forward
            generated = G_model(batch_image, pose_code, noise) #forward
            steps += 1
            # if i ==3: break;       

            # Discriminator learning as followed
            if i % ratio == 0:
                # we can use some tricks to training D
                if args.use_allhistory_epoch and epoch>=25:
                    buffer_img.add_to_image_history_buffer(generated.cpu().data.numpy())
                    print('using all history buffer')
                    for iter_ in range(len(buffer_img.image_history_buffer)//buffer_img.batch_size):
                        D_model.zero_grad()
                        img_history = buffer_img.image_history_buffer[(iter_*buffer_img.batch_size):(iter_+1)*buffer_img.batch_size]
                        img_history = Variable(torch.from_numpy(img_history).float())
                        bufferbatch_sys_label = Variable(torch.zeros(buffer_img.batch_size))
                        if args.cuda:
                            img_history=img_history.cuda()
                            bufferbatch_sys_label=bufferbatch_sys_label.cuda()
                        real_output = D_model(batch_image)
                        syn_output = D_model(img_history.detach()) # for D, we do not update the parameters to Generator and detach gradient
                        L_d_id    = CE_loss(real_output[:, :Nd], batch_id_label)
                        L_d_gan   = BCE_Loss(real_output[:, Nd].sigmoid(), batch_real_label) + BCE_Loss(syn_output[:, Nd].sigmoid(), bufferbatch_sys_label)
                        L_d_pose  = CE_loss(real_output[:, Nd+1:], batch_pose_label)
                        d_loss = L_d_gan + L_d_id + L_d_pose
                        d_loss.backward()
                        optimizer_D.step()

                if args.use_history_epoch and epoch>=10:
                    buffer_img.add_to_image_history_buffer(generated.cpu().data.numpy())
                    print('using history buffer')
                    img_history = buffer_img.get_from_image_history_buffer()
                    img_history = Variable(torch.from_numpy(img_history).float())
                    if args.cuda:
                        img_history = img_history.cuda()
                    generated[:args.batch_size//2] = img_history

                if args.use_updatingmore_epoch and epoch>=25:#15:
                    for updatenum in range(ratio):
                        pose_code = np.zeros((batch_size, Np))
                        pose_code[range(batch_size), np.random.randint(Np, size=batch_size)] = 1
                        pose_code = Variable(torch.FloatTensor(pose_code.tolist()))
                        print('updating D more')
                        if args.cuda:
                            pose_code=pose_code.cuda()
                        generated = G_model(batch_image, pose_code, noise)
                        real_output = D_model(batch_image)
                        syn_output = D_model(generated.detach()) # for D, we do not update the parameters to Generator and detach gradient
                        L_d_id    = CE_loss(real_output[:, :Nd], batch_id_label)
                        L_d_gan   = BCE_Loss(real_output[:, Nd].sigmoid(), batch_real_label) + BCE_Loss(syn_output[:, Nd].sigmoid(), batch_sys_label)
                        L_d_pose  = CE_loss(real_output[:, Nd+1:], batch_pose_label)
                        d_loss = L_d_gan + L_d_id + L_d_pose
                        d_loss.backward()
                        optimizer_D.step()

                if args.use_softlabel:
                    print('using soft label')
                    batch_real_label = torch.ones(batch_size)*np.random.uniform(0.7,1.2)
                    batch_sys_label = torch.ones(batch_size)*np.random.uniform(0.0,0.3)
                    if args.cuda: batch_real_label, batch_sys_label=batch_real_label.cuda(), batch_sys_label.cuda();
                    batch_real_label, batch_sys_label = Variable(batch_real_label), Variable(batch_sys_label)

                if args.use_noiselabel and np.random.randint(50)==0:
                    print('using noise label')
                    batch_real_label = torch.zeros(batch_size)
                    batch_sys_label = torch.ones(batch_size)
                    if args.cuda: batch_real_label, batch_sys_label=batch_real_label.cuda(), batch_sys_label.cuda();
                    batch_real_label, batch_sys_label = Variable(batch_real_label), Variable(batch_sys_label)
                 
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
                log_learning(epoch, totalepochs, steps, totalsteps, 'SingleDRGAN-D', d_loss.data[0], loss_dict, save_dir)
                # judge whether discriminator is strong or not after D update
                # D_acc = cal_acc(real_output, syn_output, batch_id_label, batch_pose_label, Nd)
                # print(D_acc)
                
            # Generator learning as followed
            else:
                syn_output=D_model(generated)   # no detach here
                reconstructed = G_model(generated, batch_pose_code, noise)
                
                L_g_id    = CE_loss(syn_output[:, :Nd], batch_id_label)
                L_g_gan   = BCE_Loss(syn_output[:, Nd].sigmoid().clamp(min=eps), batch_real_label) # for G, we use real_label in loss
                L_g_pose  = CE_loss(syn_output[:, Nd+1:], pose_code_label)
                L_g_rec = torch.mean(torch.abs(batch_image - reconstructed))
                
                g_loss = L_g_gan + L_g_id + L_g_pose
                if args.use_rec:
                    g_loss += 10*L_g_rec

                g_loss.backward()
                optimizer_G.step()                
                loss_dict = {'g_loss':g_loss.data[0], 'L_g_id':L_g_id.data[0], 'L_g_gan':L_g_gan.data[0], 'L_g_pose':L_g_pose.data[0], 'L_g_rec':L_g_rec.data[0]}
                log_learning(epoch, totalepochs, steps, totalsteps, 'SingleDRGAN-G', g_loss.data[0], loss_dict, save_dir)

            #  save generated images in some epoches
            if epoch % args.save_freq == 0 and i == sample_index:
                save_gen_path = os.path.join(save_dir, 'epoch{}_genbySimage.jpg'.format(epoch))
                vutils.save_image(generated.cpu().data, save_gen_path, normalize=True)
                save_real_path = os.path.join(save_dir, 'epoch{}_realimage.jpg'.format(epoch))
                vutils.save_image(batch_image.cpu().data, save_real_path, normalize=True)    
            
        # save loss in each epoch
        loss_log.append([d_loss.data[0], g_loss.data[0]])
        g_loss_log.append([L_g_gan.data[0], L_g_id.data[0], L_g_pose.data[0], L_g_rec.data[0]])
        d_loss_log.append([L_d_gan.data[0], L_d_id.data[0], L_d_pose.data[0]])
        # save model in some epoches
        if epoch % args.save_freq == 0:
            save_path_D = os.path.join(save_dir,'epoch{}_D.pth'.format(epoch))
            torch.save(D_model.state_dict(), save_path_D)
            save_path_G = os.path.join(save_dir,'epoch{}_G.pth'.format(epoch))
            torch.save(G_model.state_dict(), save_path_G)
        # swap D and G
        if args.use_switch and epoch>=30 and epoch%2==0:
            print('\nSwitch the encder of D with G!')
            try:
                D_para_buff = D_model.D_enc_convLayers.state_dict()
                G_para_buff = G_model.G_enc_convLayers.state_dict()
                init_weights(D_model, args.init_type)
                init_weights(G_model, args.init_type)
                D_model.D_enc_convLayers.load_state_dict(G_para_buff)
                G_model.G_enc_convLayers.load_state_dict(D_para_buff)
                print("\nSwitch model successfully.\n")
            except:
                print("\nFail to switch models.\n")
                
    # plot loss of G and D after training
    loss_log = np.array(loss_log); g_loss_log = np.array(g_loss_log); d_loss_log = np.array(d_loss_log);
    plot_loss(save_dir+'/loss.png', epo_list, {'d_loss':loss_log[:,0], 'g_loss':loss_log[:,1]})
    plot_loss(save_dir+'/loss_g.png', epo_list, {'L_g_gan':g_loss_log[:,0], 'L_g_id':g_loss_log[:,1], 'L_g_pose':g_loss_log[:,2]})
    plot_loss(save_dir+'/loss_d.png', epo_list, {'L_d_gan':d_loss_log[:,0], 'L_d_id':d_loss_log[:,1], 'L_d_pose':d_loss_log[:,2]})
    imgtogif(save_dir, (args.epochs/args.save_freq))

    return 'train_single_DRGAN successfully'
