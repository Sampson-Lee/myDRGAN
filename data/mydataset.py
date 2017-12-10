# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:20:13 2017

@author: Sampson

1. use it to create image list for dataloader
2. modify to set porper preprocess and provide dataloader iterator for training 

"""
import torch
import numpy as np
import os,random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

channel_num = 3
img_size = 96
Nd=0
Np=0
Nz=50

pose_labels_dict = {
             '041':0,'050':1,'051':2,
             '080':3,'090':4,'130':5,
             '140':6,'190':7,'200':8}

default_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

randomData_transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])

multiPIE_train_transform = transforms.Compose([
                            transforms.CenterCrop(160),
                            transforms.Scale(img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

multiPIE_test_transform = transforms.Compose([
                            transforms.CenterCrop(160),
                            transforms.Scale(img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

def img_loader(img_path):
    img = Image.open(img_path)
    # w, h = img.size
    # th, tw = (160, 160)
    # left = int(round((w - tw) / 2.))
    # upper = int(round((h - th) / 2.))
    # img = img.crop((left, upper, left + tw, upper + th)) # aligning faces is important
    # img = img.resize((110, 110))
    return img

#data=Image.open('./001_01_01_041_00.png')
##plt.imshow(data)
#arr = np.array(data, dtype=float)
##print(data.shape)
#plt.imshow(arr)
#a=data-arr

def PIL_list_reader(fileList):
    imgList = [];idList = [];poseList = [];
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, id_label, pose_label = line.strip().rstrip('\n').split(' ')
            # use
            imgList.append(imgPath)
            idList.append(int(id_label.encode("utf-8")))
            poseList.append(int(pose_label.encode("utf-8")))
            
    Np = int(max(poseList) + 1)
    Nd = int(max(idList) + 1)
    return [imgList, poseList, idList, Np, Nd]

# perID in multiPIE: 3 sessions, 9 poses, 20 illuminations = 540 images
def creat_singleDR_GAN_list(img_dir, img_num=None):

    if not img_dir.endswith('/'):
        img_dir += '/'

    img_list = os.listdir(img_dir)
    img_list.sort()
    if img_num>len(img_list) or img_num == None:
        img_num = len(img_list)
    fw = open('./' + img_dir.split("/")[-2] + 'single_DR_GAN_list.txt','w')
    for i in range(img_num):
        img_name = img_list[i]
        id_label, _, _, pose_label, _=img_name.split("_")
        fw.write(img_dir+img_name + ' ' + str(int(id_label)-1) + ' ' + str(pose_labels_dict[pose_label]) + '\n')
    
    print("generate singleDRGAN txt successfully")
    fw.close()

# data_source='/data5/lixinpeng/dataset/multiPIE/train/'
# creat_singleDR_GAN_list(data_source)

def creat_multiDR_GAN_list(img_dir, images_perID, img_num=None):

    if not img_dir.endswith('/'):
        img_dir += '/'
            
    img_list = os.listdir(img_dir).sort()
    img_list.sort()
    img_num = len(img_list)
    imgList = [];idList = [];poseList = [];
    if img_num>len(img_list) or img_num == None:
        img_num = len(img_list)

    for i in range(img_num):
        img_name = img_list[i]
        imgList.append(img_dir + img_name)
        id_label, _, _, pose_label, _=img_name.split("_")
        idList.append(int(id_label)-1)
        poseList.append(pose_labels_dict[pose_label])
    
    # remove IDs which has images less than images_perID
    id_target = []
    idset = set(idList)
    for item in idset:
        if idList.count(item) > images_perID: id_target.append(item)
        
    index_target = [i for i, id in enumerate(idList) if id in id_target]
    imgList_target = np.array(imgList)[index_target]
    idList_target = np.array(idList)[index_target]
    poseList_target = np.array(poseList)[index_target]
    # sample args.images_perID images for each ID randomely
    fw = open('./' + img_dir.split("/")[-2] + 'multi_DR_GAN_list.txt','w')
    batch = img_num//(images_perID*60)
    for i in range(batch*60):
        perID = random.choice(id_target)
        mask = (idList_target==perID)
        imgs = imgList_target[mask]
        ids = idList_target[mask]
        poses = poseList_target[mask]

        for sample_index in random.sample(range(len(ids)), images_perID):
            image_sample = imgs[sample_index]
            id_label_sample = ids[sample_index]
            pose_label_sample = poses[sample_index]
            fw.write(image_sample+ ' ' + str(id_label_sample) + ' ' + str(pose_label_sample) + '\n')

    print("generate multiDRGAN txt successfully")
    fw.close()

# data_source='/data5/lixinpeng/dataset/multiPIE/train/'
# creat_multiDR_GAN_list(data_source, 6)

# perID in multiPIE: 3 or 4 sessions, 9 poses, 20 illuminations = 540 images
def creat_multiDR_GAN_totallist(img_dir, images_perID, img_num=None):

    if not img_dir.endswith('/'):
        img_dir += '/'

    img_list = os.listdir(img_dir)
    img_list.sort()
    if img_num>len(img_list) or img_num == None:
        img_num = len(img_list)

    # shuffle images and keep images_perID structure
    img_array = np.array(img_list).reshape((-1, 180))
    id_session_num, imgnum = img_array.shape
    imgindex = np.arange(imgnum);random.shuffle(imgindex);
    img_array = img_array[:, imgindex]
    id_session_index = np.arange(id_session_num);random.shuffle(id_session_index);
    img_array = img_array[id_session_index, :]

    img_array = img_array.reshape((-1,images_perID))
    unitnum = img_array.shape[0]
    unitindex = np.arange(unitnum);random.shuffle(unitindex);
    img_array = img_array[unitindex,:]
    img_list = img_array.flatten().tolist()

    fw = open('./' + img_dir.split("/")[-2] + 'multi_DR_GAN_list.txt','w')
    for i in range(img_num):
        img_name = img_list[i]
        id_label, _, _, pose_label, _=img_name.split("_")
        fw.write(img_dir+img_name + ' ' + str(int(id_label)-1) + ' ' + str(pose_labels_dict[pose_label]) + '\n')
    
    print("generate multiDRGAN txt successfully")
    fw.close()
    
# data_source='/data5/lixinpeng/dataset/multiPIE/train/'
# creat_multiDR_GAN_totallist(data_source, 6)

def creat_multiDR_GAN_probelist(fileList, images_perID, img_num=None):
    with open(fileList, 'r') as file:
        img_list=file.readlines()

    if img_num>len(img_list) or img_num == None:
        img_num = len(img_list)

    # shuffle images and keep images_perID structure
    img_array = np.array(img_list).reshape((-1, 20))
    id_session_num, ill_num = img_array.shape
    add_size = images_perID - (20%images_perID)
    imgindex = np.concatenate((np.arange(ill_num), np.random.randint(20, size=add_size)), axis=0);random.shuffle(imgindex);
    img_array = img_array[:, imgindex]
    img_list = img_array.flatten().tolist()

    fw = open(fileList[:-8]+'multi_list.txt','w')
    fw.writelines(img_list)
    
    print("generate multiDRGAN txt successfully")
    fw.close()

# data_source='/home/lixinpeng/myDRGAN/data/iden_probe200_DR_GAN_list.txt'
# creat_multiDR_GAN_probelist(data_source, 6)

class multiPIE(Dataset):
    # 继承Dataset, 重载__init__, __getitem__, __len__
    def __init__(self, fileList, transform_mode=None, list_reader=PIL_list_reader, loader=img_loader):
        self.channel_num = channel_num
        self.image_size = img_size
        self.loader    = loader
        self.Nz = 50
        self.imgList, self.pose_label, self.id_label, self.Np, self.Nd = list_reader(fileList)
        if transform_mode=='train_transform': self.transform = multiPIE_train_transform
        elif transform_mode=='test_transform': self.transform = multiPIE_test_transform
        else: self.transform=None
        
    def __getitem__(self, index):
        imgPath = self.imgList[index]
        img = self.loader(imgPath)

        if self.transform is not None:
            img = self.transform(img)
        return img, self.id_label[index], self.pose_label[index]

    def __len__(self):
        return len(self.imgList)

class randomData(Dataset):
    # 继承Dataset, 重载__init__, __getitem__, __len__
    def __init__(self, transform=randomData_transform):
        self.transform = transform
        self.channel_num = channel_num
        self.image_size = img_size
        self.images, self.id_labels, self.pose_labels, self.Nd, self.Np ,self.Nz= self.create_randomdata()

    def __getitem__(self, index):
        img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.id_labels[index], self.pose_labels[index]

    def __len__(self):
        return len(self.images)
    
    def create_randomdata(self, data_size=64, Nd=200, Np=9, Nz=50):
        """
        Create random data
    
        ### ouput
        images : 4 dimension tensor (the number of image x channel x image_height x image_width)
        id_labels : one-hot vector with Nd dimension
        pose_labels : one-hot vetor with Np dimension
        Nd : the nuber of ID in the data
        Np : the number of discrete pose in the data
        Nz : size of noise vector
        """
        images = np.random.randn(data_size, 230, 230, self.channel_num)
        id_labels = torch.LongTensor(np.random.randint(Nd, size=data_size).tolist())
        pose_labels = torch.LongTensor(np.random.randint(Np, size=data_size).tolist())
    
        return [images, id_labels, pose_labels, Nd, Np, Nz]


