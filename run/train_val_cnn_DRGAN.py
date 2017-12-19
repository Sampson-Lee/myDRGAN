#!/usr/bin/env python
# encoding: utf-8

import sys, datetime, os
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib as mpl
mpl.use('Agg')

sys.path.append("..")
from util.myacc import cal_acc_id
from util.mylog import plot_loss, print_network, print_params
import data.mydataset as mydataset

class args(object):
    Nd=200
    Nf=320
    channel_num=3
    use_Adam=True
    lr_Adam=0.0002
    beta1_Adam=0.5
    beta2_Adam=0.999
    use_SGD=False
    lr_SGD=0.01
    momentum=0.9
    weight_decay=1e-4
    milestones=[10,15,17]
    train_tran='trainaug_transform'
    test_tran='testaug_transform'
    epochs=20
    train_bs=64
    val_bs=64
    cuda=True
    device='9'
    use_ReLU=False
    use_ELU=True
    use_Dropout=0.2
    use_strided=True
    use_maxpool=False
    
    traindata = '/data/lixinpeng/myDRGAN/data/trainsingle_DR_GAN_list.txt'
    valdata_dir = '/data/lixinpeng/myDRGAN/data/'
    gallery = '/data/lixinpeng/myDRGAN/data/iden_gallery_DR_GAN_list.txt'
    save_dir = '/data/lixinpeng/myDRGAN/snapshot/cnn/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    probelist = ['051', '050', '041', '190', '200']
    
class CASIA_Net(nn.Module):
    """Compare supervised CNN with semi-supervised encoder"""
    def __init__(self, args):
        super(CASIA_Net, self).__init__()
        def network(args):
            convLayers = [
                    nn.Conv2d(args.channel_num, 32, 3, 1, 1, bias=False), # Bxchx96x96 -> Bx32x96x96
                    nn.BatchNorm2d(32)]
            if args.use_ReLU:
                convLayers += [nn.ReLU(True)]
            elif args.use_ELU:
                convLayers += [nn.ELU()]
            convLayers += [
                    nn.Conv2d(32, 64, 3, 1, 1, bias=False), # Bx32x96x96 -> Bx64x96x96
                    nn.BatchNorm2d(64)]
            if args.use_ReLU:
                convLayers += [nn.ReLU(True)]
            elif args.use_ELU:
                convLayers += [nn.ELU()]
                
            for i in range(1,5):
                if args.use_strided:
                    convLayers += [
                            nn.ZeroPad2d((0, 1, 0, 1)),   # Bx64*ix96x96 -> Bx64*ix97x97
                            nn.Conv2d(64*i, 64*i, 3, 2, 0, bias=False), # Bx64x97x97 -> Bx64x48x48
                            nn.BatchNorm2d(64*i)]
                    if args.use_ReLU:
                        convLayers += [nn.ReLU(True)]
                    elif args.use_ELU:
                        convLayers += [nn.ELU()]
                elif args.use_maxpool:
                    convLayers += [
                            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)] # Bx64x96x96 -> Bx64x48x48
            
                convLayers += [
                        nn.Conv2d(64*i, 32*(i+1), 3, 1, 1, bias=False), # Bx64x48x48 -> Bx128x48x48
                        nn.BatchNorm2d(32*(i+1))]
                if args.use_ReLU:
                    convLayers += [nn.ReLU(True)]
                if args.use_ELU:
                    convLayers += [nn.ELU()]
                
                convLayers += [
                        nn.Conv2d(32*(i+1), 64*(i+1), 3, 1, 1, bias=False), 
                        nn.BatchNorm2d(64*(i+1))]
                if args.use_ReLU:
                    convLayers += [nn.ReLU(True)]
                if args.use_ELU:
                    convLayers += [nn.ELU()]
                    
            convLayers += [
                        nn.AvgPool2d(6, stride=1)] #  Bx320x6x6 -> Bx320x1x1
            if args.use_Dropout:
	            convLayers += [
	                        nn.Dropout2d(p=args.use_Dropout)] #  Bx320x6x6 -> Bx320x1x1            	
            return convLayers

        self.conv = nn.Sequential(*network(args))
        self.fc = nn.Linear(args.Nf, args.Nd)

    def forward(self, input, extract=False):
        x = self.conv(input)

        x = x.squeeze(2)
        x = x.squeeze(2)
        
        if extract: return x
        x = self.fc(x) # Bx320 -> B x Nd

        return x

def train_cnn(train_loader, model, optimizer, epoch, args):
    """Train cnn and calculate accuracy to identity"""
    CE_loss = nn.CrossEntropyLoss()
    model.train()
    
    acc_=0
    loss_=0
    for i, [batch_image, batch_id_label, _] in enumerate(train_loader):

        if args.cuda: batch_image, batch_id_label=batch_image.cuda(), batch_id_label.cuda()
        batch_image, batch_id_label = Variable(batch_image), Variable(batch_id_label)

        # compute output
        output_id = model(batch_image)
        loss = CE_loss(output_id, batch_id_label)
        acc = cal_acc_id(output_id, batch_id_label)
        loss_ +=loss.data[0]
        acc_ +=acc

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%20 == 0:
        	print('epoch {}\t step {}\t train_loss {}\t train_acc {}\n'.format(epoch, i, loss.data[0], acc))
        # if i == 2: break;
        
    return loss_/(i+1), acc_/(i+1)

def val_cnn(val_loader, model, epoch, args):
    """Train cnn and calculate accuracy to identity"""
    model.eval()

    gallery_features = torch.Tensor(); gallery_idlabel = torch.LongTensor();
    dataset = mydataset.multiPIE(args.gallery, transform_mode='test_transform')
    gallery_loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=0)

    image_number = 0
    for i, [batch_image, batch_id_label, _] in enumerate(gallery_loader):
        image_number += batch_image.size(0)
        print('extracting {} gallery features'.format(image_number))
        if args.cuda: batch_image = batch_image.cuda()
        batch_image = Variable(batch_image)
        gallery_idlabel = torch.cat((gallery_idlabel, torch.LongTensor(batch_id_label)), 0)

        batch_features = model(batch_image, extract=True).cpu().data        
        gallery_features = torch.cat((gallery_features, batch_features), 0)

    image_number = 0
    acc_ = 0
    for i, [batch_image, batch_id_label, _] in enumerate(val_loader):
        image_number += batch_image.size(0)
        print('extracting {} probe features'.format(image_number))
        if args.cuda: batch_image = batch_image.cuda()
        batch_image = Variable(batch_image)
        batch_id_label = torch.LongTensor(batch_id_label)
        batch_features = model(batch_image, extract=True).cpu().data

        Distance = pairwise_distances(gallery_features.numpy(), batch_features.numpy(), metric='cosine', n_jobs=-1)
        maxindex = np.argsort(Distance, axis=0)[0,:]
        match = (gallery_idlabel[torch.LongTensor(maxindex)] == batch_id_label)
        # print(match)
        acc = float(match.sum())/match.shape[0]
        acc_ += acc
        # if i == 1: break;
        
    print('val_acc is {} in {}\n'.format(acc_/(i+1), epoch))
    return acc_/(i+1)


def main():
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    print_params(args, args.save_dir)

    model = CASIA_Net(args)
    print_network(model, name='CASIA_Net')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if len(args.device.split(','))>1:
        print('model uses multigpu!')
        model = nn.DataParallel(model)
    model.cuda()
    
    if args.use_Adam:
    	optimizer = optim.Adam(model.parameters(), lr = args.lr_Adam, betas=(args.beta1_Adam, args.beta2_Adam))
    if args.use_SGD:
    	optimizer = torch.optim.SGD(model.parameters(), lr = args.lr_SGD, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    train_log=[]; val_log=[];
    # train
    for epoch in range(1, args.epochs+1):
        if args.use_SGD: scheduler.step()
        for param_group in optimizer.param_groups:
            print('learning rate is {}'.format(param_group['lr']))
        			 
        train_dataset = mydataset.multiPIE(args.traindata, transform_mode=args.train_tran)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True, num_workers=0)
        train_loss, train_acc = train_cnn(train_loader, model, optimizer, epoch, args)
        train_log.append([train_loss, train_acc])
        torch.save(model.state_dict(), args.save_dir+'/epoch{}'.format(epoch))
    # val
    start=1
    for epoch in range(start, args.epochs+1):
    	# if epoch == 5: break;
        model.load_state_dict(torch.load(args.save_dir+'/epoch{}'.format(epoch)))
        for probe in args.probelist:
            print('validation in {}'.format(probe))
            valdata = args.valdata_dir+'iden_probe'+probe+'_DR_GAN_single_list.txt'
            val_dataset = mydataset.multiPIE(valdata, transform_mode=args.test_tran)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_bs, shuffle=False, num_workers=0)
            val_acc = val_cnn(val_loader, model, epoch, args)
            val_log.append(val_acc)
    
    train_log =np.array(train_log); val_log=np.array(val_log).reshape((-1,5))
    val_log = np.concatenate([val_log, val_log.mean(axis=1).reshape((-1,1))], axis=1)
    np.savetxt(args.save_dir+"/train_result.txt", train_log)
    np.savetxt(args.save_dir+"/val_result.txt", val_log)
      
    plot_loss(args.save_dir+'/cnn_train.png', range(train_log.shape[0]), {'train_loss':train_log[:,0], 'train_acc':train_log[:,1]})
    plot_loss(args.save_dir+'/cnn_val.png', range(val_log.shape[0]), {'051':val_log[:,0], '050':val_log[:,1], 
                                                     '041':val_log[:,2], '190':val_log[:,3], '200':val_log[:,4], 'avg':val_log[:,5]})
main()
