#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import data.mydataset as mydataset
from sklearn.metrics.pairwise import pairwise_distances
import torch
from torch.autograd import Variable

def representation_learning(dataloader, G_model, args):
    """
    Generate_Image with learned Generator

    ### input
    dataloader  : source images and pose code
    G_model     : learned Generator
    args        : options

    ### output
    features    : extracted disentangled features of each image

    """
    if args.cuda: G_model.cuda()

    G_model.eval()
    gallery_features = torch.Tensor(); gallery_idlabel = torch.LongTensor();
    fileList = args.data_path + 'iden_gallery_DR_GAN_list.txt'
    dataset = mydataset.multiPIE(fileList)
    gellery_loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=0)
    
    # Generator encoder extract features
    if args.mode=='idensingle':
        image_number = 0
        for i, [batch_image, batch_id_label, _] in enumerate(gellery_loader):
            image_number += batch_image.size(0)
            print('extracting {} gallery features'.format(image_number))
            if args.cuda: batch_image = batch_image.cuda()
            batch_image = Variable(batch_image)
            gallery_idlabel = torch.cat((gallery_idlabel, torch.LongTensor(batch_id_label)), 0)

            batch_features = G_model(batch_image, extract=True).cpu().data        
            gallery_features = torch.cat((gallery_features, batch_features), 0)

        image_number = 0
        acc = 0
        for i, [batch_image, batch_id_label, _] in enumerate(dataloader):
            image_number += batch_image.size(0)
            print('extracting {} probe features'.format(image_number))
            if args.cuda: batch_image = batch_image.cuda()
            batch_image = Variable(batch_image)
            batch_id_label = torch.LongTensor(batch_id_label)
            batch_features = G_model(batch_image, extract=True).cpu().data

            Distance = pairwise_distances(gallery_features.numpy(), batch_features.numpy(), metric='cosine', n_jobs=-1)
            maxindex = np.argsort(Distance, axis=0)[0,:]
            match = (gallery_idlabel[torch.LongTensor(maxindex)] == batch_id_label)
            # print(match)
            acc += float(match.sum())/match.shape[0]
            # if i == 2: break;

        acc = acc/(i+1)
        print(acc)

    if args.mode=='idenmulti':
        image_number = 0
        for i, [batch_image, batch_id_label, _] in enumerate(gellery_loader):
            image_number += batch_image.size(0)
            print('extracting {} gallery features'.format(image_number))
            if args.cuda: batch_image = batch_image.cuda()
            batch_image = Variable(batch_image)
            gallery_idlabel = torch.cat((gallery_idlabel, torch.LongTensor(batch_id_label)), 0)

            batch_features = G_model(batch_image, single=True, extract=True).cpu().data        
            gallery_features = torch.cat((gallery_features, batch_features), 0)

        image_number = 0
        acc = 0
        for i, [batch_image, batch_id_label, _] in enumerate(dataloader):
            batch_size = batch_image.size(0)
            batch_size_unique = batch_size // args.images_perID
            image_number += batch_size_unique
            print('extracting {} probe features'.format(image_number))
            batch_id_label_unique = batch_id_label[::args.images_perID]
            
            if args.cuda: batch_image = batch_image.cuda()
            batch_image = Variable(batch_image)
            batch_id_label_unique = torch.LongTensor(batch_id_label_unique)
            # Generator generates images
            batch_features = G_model(batch_image, extract=True).cpu().data

            Distance = pairwise_distances(gallery_features.numpy(), batch_features.numpy(), metric='cosine', n_jobs=-1)
            maxindex = np.argsort(Distance, axis=0)[0,:]
            match = (gallery_idlabel[torch.LongTensor(maxindex)] == batch_id_label_unique)
            print(match)
            acc += float(match.sum())/match.shape[0]
            # if i == 2: break;

        acc = acc/(i+1)
        print('{}%'.format(acc*100))

    return acc