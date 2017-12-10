import single_DR_GAN_model as single_model
import multiple_DR_GAN_model as multi_model
from weights import init_weights
import torch
import torch.nn as nn
import os

def initmodel(args):
    print('creat model!')
    netD = single_model.Discriminator(args)
    netG = single_model.Generator(args)
    if args.mode in ['trainmulti', 'genmulti', 'idenmulti']:
        netD = multi_model.Discriminator(args)
        netG = multi_model.Generator(args)
    
    print('model initialize weights!')
    init_weights(netD, args.init_type)
    init_weights(netG, args.init_type)

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if len(args.device.split(','))>1:
            print('model uses multigpu!')
            netD = nn.DataParallel(netD)
            netG = nn.DataParallel(netG)
        print('model uses cuda!')
        netD.cuda()
        netG.cuda()
            
    if args.snapshot != None:
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            netD.load_state_dict(torch.load('{}_D.pth'.format(args.snapshot)))
            netG.load_state_dict(torch.load('{}_G.pth'.format(args.snapshot)))
            print("\nLoading successfully.")
        except:
            print("Sorry, This snapshot doesn't exist.")
            exit()
    
    return netD, netG