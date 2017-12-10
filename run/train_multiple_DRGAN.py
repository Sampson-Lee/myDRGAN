#!/usr/bin/env python
# encoding: utf-8

import os, datetime, random, sys
import numpy as np
import torchvision.utils as vutils
import matplotlib as mpl
mpl.use('Agg')
import torch
from torch import nn, optim
from torch.autograd import Variable

sys.path.append("..")
from util.mylog import log_learning, plot_loss, imgtogif
#from util.myacc import cal_acc
from util.mybuffer import ImageHistoryBuffer
from scheduler import get_scheduler

epoch_set = [6*multi for multi in range(6)]
ratio_set = [2**(power+1)+1 for power in range(6)]

def train_multiple_DRGAN(dataloader, D_model, G_model, args):
    batch_size, Np, Nd, Nz = args.batch_size, args.Np, args.Nd, args.Nz
    image_history_buffer = ImageHistoryBuffer((0, args.channel_num, args.img_size, args.img_size),
                                              args.batch_size * 2, args.batch_size)

    D_model.train()
    G_model.train()

    lr_Adam = args.lr
    beta1_Adam = args.beta1
    beta2_Adam = args.beta2
    eps = 10**-300

    optimizer_D = optim.Adam(D_model.parameters(), lr = lr_Adam, betas=(beta1_Adam, beta2_Adam))
    optimizer_G = optim.Adam(G_model.parameters(), lr = lr_Adam, betas=(beta1_Adam, beta2_Adam))
    if args.lr_policy != 'constant':
        scheduler_D = get_scheduler(optimizer_D, args)
        scheduler_G = get_scheduler(optimizer_G, args)
    CE_loss = nn.CrossEntropyLoss()
    BCE_Loss = nn.BCELoss()

    loss_log = []; g_loss_log = []; d_loss_log = [];
    ratio = 2
    totalsteps = len(dataloader); totalepochs = args.epochs;
    # D_acc = 0
    save_dir = os.path.join(args.save_dir, 'Multi', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.isdir(save_dir): os.makedirs(save_dir);print('make '+ save_dir + '!')

    print("Parameters:")
    with open('{}/Parameters.txt'.format(save_dir),'a') as f:
        for attr, value in sorted(args.__dict__.items()):
            text ="\t{}={}\n".format(attr.upper(), value)
            print(text)
            f.write(text)
        text = 'epoch 0 - ratio 1:1\n'
        print(text)
        f.write(text)
        for epo, rat in zip(epoch_set, ratio_set):
            text = 'epoch '+str(epo)+' - '+'ratio 1:'+str(rat-1)+'\n'
            print(text)
            f.write(text)

    epo_list = range(args.epochs)
    for epoch in epo_list:
        if args.lr_policy != 'constant':scheduler_D.step();scheduler_G.step();
        steps = 0
        sample_index = random.randint(0,len(dataloader))
        if epoch in epoch_set:
            ratio = ratio_set[epoch_set.index(epoch)]
            print('ratio is 1:{} now! \n'.format(ratio-1))
        for i, [batch_image, batch_id_label, batch_pose_label] in enumerate(dataloader):
            D_model.zero_grad()
            G_model.zero_grad()
            
            batch_size = batch_image.size(0)
            batch_size_unique = batch_size // args.images_perID
            print(batch_size_unique)
            batch_id_label_unique = batch_id_label[::args.images_perID]

            # generate noise, pose and id for summed features
            # when use images respectively
            noise = torch.FloatTensor(np.random.uniform(-1,1, (batch_size, Nz)))
            pose_code_label  = np.random.randint(Np, size=batch_size) # get a list of int in range(Np)
            pose_code = np.zeros((batch_size, Np))
            pose_code[range(batch_size), pose_code_label] = 1
            pose_code_label = torch.LongTensor(pose_code_label.tolist())
            pose_code = torch.FloatTensor(pose_code.tolist())
            batch_real_label = torch.ones(batch_size)
            batch_sys_label = torch.zeros(batch_size)
            # when summarize features of the same person
            noise_unique = torch.FloatTensor(np.random.uniform(-1,1, (batch_size_unique, Nz)))
            pose_code_label_unique  = np.random.randint(Np, size=batch_size_unique) # get a list of int in range(Np)
            pose_code_unique = np.zeros((batch_size_unique, Np))
            pose_code_unique[range(batch_size_unique), pose_code_label_unique] = 1
            pose_code_label_unique = torch.LongTensor(pose_code_label_unique.tolist())
            pose_code_unique = torch.FloatTensor(pose_code_unique.tolist())
            batch_real_label_unique = torch.ones(batch_size_unique)
            
            if args.cuda:
                batch_image, batch_id_label, batch_pose_label, batch_real_label, batch_sys_label, \
                noise, pose_code, pose_code_label = \
                    batch_image.cuda(), batch_id_label.cuda(), batch_pose_label.cuda(), batch_real_label.cuda(), batch_sys_label.cuda(), \
                    noise.cuda(), pose_code.cuda(), pose_code_label.cuda()

                batch_id_label_unique, batch_real_label_unique, \
                noise_unique, pose_code_unique, pose_code_label_unique = \
                    batch_id_label_unique.cuda(), batch_real_label_unique.cuda(), \
                    noise_unique.cuda(), pose_code_unique.cuda(), pose_code_label_unique.cuda()

            batch_image, batch_id_label, batch_pose_label, batch_real_label, batch_sys_label, \
            noise, pose_code, pose_code_label = \
                Variable(batch_image), Variable(batch_id_label), Variable(batch_pose_label), Variable(batch_real_label), Variable(batch_sys_label), \
                Variable(noise), Variable(pose_code), Variable(pose_code_label)

            batch_id_label_unique, batch_real_label_unique, \
            noise_unique, pose_code_unique, pose_code_label_unique = \
                Variable(batch_id_label_unique), Variable(batch_real_label_unique), \
                Variable(noise_unique), Variable(pose_code_unique), Variable(pose_code_label_unique)

            # generator forward
            generated = G_model(batch_image, pose_code, noise, single=True)
            generated_unique = G_model(batch_image, pose_code_unique, noise_unique)
            steps += 1

            # Discriminator learning as followed
            if i % ratio == 0:
                # we can use some tricks to training D
                img_history = image_history_buffer.get_from_image_history_buffer()
                image_history_buffer.add_to_image_history_buffer(generated.cpu().data.numpy())
                gen_his = generated
                if len(img_history) and args.use_history:
                    print('using history buffer')
                    img_history = Variable(torch.from_numpy(img_history))
                    if args.cuda:
                        img_history.cuda()
                    gen_his[:args.batch_size // 2] = img_history

                if args.use_softlabel:
                    print('using soft label')
                    batch_real_label = torch.ones(
                        batch_size) * np.random.uniform(0.7, 1.2)
                    batch_sys_label = torch.ones(
                        batch_size) * np.random.uniform(0.0, 0.3)
                    if args.cuda:
                        batch_real_label, batch_sys_label = batch_real_label.cuda(), batch_sys_label.cuda()
                    batch_real_label, batch_sys_label = Variable(
                        batch_real_label), Variable(batch_sys_label)

                if args.use_noiselabel and np.random.randint(50) == 0:
                    print('using noise label')
                    batch_real_label = torch.zeros(batch_size)
                    batch_sys_label = torch.ones(batch_size)
                    if args.cuda:
                        batch_real_label, batch_sys_label = batch_real_label.cuda(), batch_sys_label.cuda()
                    batch_real_label, batch_sys_label = Variable(
                        batch_real_label), Variable(batch_sys_label)
                    
                real_output = D_model(batch_image)
                syn_output = D_model(gen_his.detach()) # .detach() をすることでGeneratorのパラメータを更新しない

                # three loss: L_d_id, L_g_gan, L_d_pose
                L_d_id    = CE_loss(real_output[:, :Nd], batch_id_label)
                L_d_gan   = BCE_Loss(real_output[:, Nd].sigmoid(), batch_real_label) + BCE_Loss(syn_output[:, Nd].sigmoid(), batch_sys_label)
                L_d_pose  = CE_loss(real_output[:, Nd+1:], batch_pose_label)
                d_loss = L_d_id + L_d_gan + L_d_pose

                d_loss.backward()
                optimizer_D.step()
                loss_dict = {'d_loss':d_loss.data[0], 'L_d_id':L_d_id.data[0], 'L_d_gan':L_d_gan.data[0], 'L_d_pose':L_d_pose.data[0]}
                log_learning(epoch, totalepochs, steps, totalsteps, 'MultiDRGAN-D', d_loss.data[0], loss_dict, save_dir)
                # judge whether discriminator is strong or not after D update
                # D_acc = cal_acc(real_output, syn_output, batch_id_label, batch_pose_label, Nd)
                # print(D_acc)
                
            # Generator learning as followed
            else:
                syn_output = D_model(generated)
                syn_output_unique = D_model(generated_unique) # no detach here

                L_g_id    = CE_loss(syn_output[:, :Nd], batch_id_label)
                L_g_gan   = BCE_Loss(syn_output[:, Nd].sigmoid().clamp(min=eps), batch_real_label) # for G, we use real_label in loss
                L_g_pose  = CE_loss(syn_output[:, Nd+1:], pose_code_label)

                L_g_id_unique     = CE_loss(syn_output_unique[:, :Nd], batch_id_label_unique)
                L_g_gan_unique    = BCE_Loss(syn_output_unique[:, Nd].sigmoid().clamp(min=eps), batch_real_label_unique)
                L_g_pose_unique   = CE_loss(syn_output_unique[:, Nd+1:], pose_code_label_unique)

                g_loss = L_g_gan + L_g_id + L_g_pose + L_g_gan_unique + L_g_id_unique + L_g_pose_unique

                g_loss.backward()
                optimizer_G.step()
                loss_dict = {'g_loss':g_loss.data[0], 'L_g_id':L_g_id.data[0], 'L_g_gan':L_g_gan.data[0], 'L_g_pose':L_g_pose.data[0], \
                             'L_g_gan_unique':L_g_gan_unique.data[0], 'L_g_id_unique':L_g_id_unique.data[0], 'L_g_pose_unique':L_g_pose_unique.data[0]}
                log_learning(epoch, totalepochs, steps, totalsteps, 'MultiDRGAN-G', g_loss.data[0], loss_dict, save_dir)
            
            #  save generated images in some epoches
            if epoch % args.save_freq == 0 and i == sample_index:
                save_gen_path = os.path.join(save_dir, 'epoch{}_genbySimage.jpg'.format(epoch))
                vutils.save_image(generated.cpu().data, save_gen_path, normalize=True)
                save_gen_path = os.path.join(save_dir, 'epoch{}_genbyMimage.jpg'.format(epoch))
                vutils.save_image(generated_unique.cpu().data, save_gen_path, normalize=True)
                save_real_path = os.path.join(save_dir, 'epoch{}_realimage.jpg'.format(epoch))
                vutils.save_image(batch_image.cpu().data, save_real_path, normalize=True)
            # if i ==3: break;
            
        # save loss in each epoch
        loss_log.append([d_loss.data[0], g_loss.data[0]])
        g_loss_log.append([L_g_gan.data[0], L_g_id.data[0], L_g_pose.data[0], L_g_gan_unique.data[0], \
                          L_g_id_unique.data[0], L_g_pose_unique.data[0]])
        d_loss_log.append([L_d_gan.data[0], L_d_id.data[0], L_d_pose.data[0]])
        # save model in some epoches
        if epoch % args.save_freq == 0:
            save_path_D = os.path.join(save_dir,'epoch{}_D.pth'.format(epoch))
            torch.save(D_model.state_dict(), save_path_D)
            save_path_G = os.path.join(save_dir,'epoch{}_G.pth'.format(epoch))
            torch.save(G_model.state_dict(), save_path_G)
        # swap D and G
        if args.use_switch and epoch==1:
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
    plot_loss(save_dir+'/loss_g.png', epo_list, {'L_g_gan':g_loss_log[:,0], 'L_g_id':g_loss_log[:,1], \
                                   'L_g_pose':g_loss_log[:,2], 'L_g_gan_unique':g_loss_log[:,3], \
                                   'L_g_id_unique':g_loss_log[:,4], 'L_g_pose_unique':g_loss_log[:,5]})
    plot_loss(save_dir+'/loss_d.png', epo_list, {'L_d_gan':d_loss_log[:,0], 'L_d_id':d_loss_log[:,1], 'L_d_pose':d_loss_log[:,2]})
    imgtogif(save_dir, (args.epochs/args.save_freq))

    return 'train_multiple_DRGAN successfully'