#!/usr/bin/env python
# encoding: utf-8

import argparse
from model.model import initmodel
from data.data import initdataloader
from run.run import initrun

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='DR_GAN')
    # learning parameters
    parser.add_argument('-lr', type=float, default=0.0002, help='initial learning rate [default: 0.0002]')
    parser.add_argument('-lr_policy', type=str, default='constant', help='constant | step | multistep | plateau')
    parser.add_argument('-beta1', type=float, default=0.5, help='adam optimizer parameter [default: 0.5]')
    parser.add_argument('-beta2', type=float, default=0.999, help='adam optimizer parameter [default: 0.999]')
    parser.add_argument('-epochs', type=int, default=20, help='number of epochs for train [default: 1000]')
    parser.add_argument('-batch_size', type=int, default=8, help='batch size for training [default: 8]')
    parser.add_argument('-use_switch', action='store_true', default=False, help='enable the model switch')
    parser.add_argument('-use_history_epoch', action='store_true', default=False, help='enable the hitory buffer')
    parser.add_argument('-use_allhistory_epoch', action='store_true', default=False, help='enable all the hitory buffer')
    parser.add_argument('-use_updatingmore_epoch', action='store_true', default=False, help='enable updating more time per epoch')
    parser.add_argument('-use_softlabel', action='store_true', default=False, help='enable the soft label')
    parser.add_argument('-use_noiselabel', action='store_true', default=False, help='enable the noise label')
    parser.add_argument('-use_rec', action='store_true', default=False, help='enable the reconstruct loss')
    parser.add_argument('-cuda', action='store_true', default=True, help='enable the gpu')
    parser.add_argument('-device', type=str, default='0', help='the number of gpu to use')
    
    # model
    parser.add_argument('-mode', default='trainmulti', help=' trainrandom | trainsingle | trainmulti | gensingle | genmulti | idensingle | idenmulti')
    parser.add_argument('-init_type', type=str, default='normal', help='the type of initializing model: normal | xavier | kaiming | orthogonal')
    parser.add_argument('-images_perID', type=int, default=6, help='number of images per person to input to multi image DR_GAN')
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot(snapshot/{Single or Multiple}/{date}/{epoch}) [default: None]')
    parser.add_argument('-modeldir', type=str, default=None, help='filename of model to test(snapshot/{Single or Multiple}/{date}) [default: None]')
    parser.add_argument('-Np', type=int, default=9, help='Np : Dimension of pose vector')
    parser.add_argument('-Nd', type=int, default=200, help='Nd : Number of identitiy to classify')
    parser.add_argument('-Nf', type=int, default=320, help='Nf : Dimension of features')
    parser.add_argument('-Nz', type=int, default=50, help='Nz : Dimension of noise vector')

    # saving parameters
    parser.add_argument('-save_dir', type=str, default='./snapshot', help='where to save the snapshot')
    parser.add_argument('-save_freq', type=int, default=1, help='save learned model for every "-save-freq" epoch')
    
    # data source
    parser.add_argument('-data_path', type=str, default=None, help='path to filelist')
    parser.add_argument('-channel_num', type=int, default=3, help='input images color channel')
    parser.add_argument('-img_size', type=int, default=96, help='input image size to network')
    
    # model args
    args = parser.parse_args()
    dataloader = initdataloader(args)
    netD, netG = initmodel(args)
    message = initrun(dataloader, netD, netG, args)
    print(message)
