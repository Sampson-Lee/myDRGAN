#!/usr/bin/env python
# encoding: utf-8
import matplotlib.pyplot as plt
from PIL import Image
import os
import imageio

def log_learning(epoch, steps, modelname, save_dir, loss, loss_dict):
    text = "EPOCH : {}, step : {}, model : {},  loss : {} ".format(epoch, steps, modelname, loss)
    for (key, value) in  loss_dict.items():
         text = text + key + ' : ' + str(value) + ' '
    print(text)
    with open(save_dir+'/'+modelname+'_log.txt','a') as f:
        f.write("{}\n".format(text))

def plot_loss(save_dir, x, y):
#    print(loss_log.shape)
    plt.figure()
    for key, value in y.items():
        plt.plot(x, value, label=key)
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(save_dir, bbox_inches='tight')
    plt.show()

def plot_img(img_dir, num):
    import re
    if not img_dir.endswith('/'): img_dir += '/'
    img_list = os.listdir(img_dir)
    for itemname in img_list:
        for e in range(num):
#             epoch20_generatedimage.jpg
            res = re.search(r'^' + 'epoch' + str(e+1) + '_generatedimage' + '.(jpg|png)$', itemname)
            if res != None:
                img = Image.open(img_dir+res.group(0))
                plt.imshow(img)
                plt.axis('off') # 不显示坐标轴
                plt.title('epoch{}'.format(e))
                plt.savefig(img_dir+res.group(0))
                plt.close()

def imgtogif(img_dir, num):
    import re
    if not img_dir.endswith('/'): img_dir += '/'
    img_list = os.listdir(img_dir)
    images = []
    for itemname in img_list:
        for e in range(num):
            # epoch20_generatedimage.jpg
            res = re.search(r'^' + 'epoch' + str(e) + '_genbySimage' + '[A-Za-z_]*' + '.(jpg|png)$', itemname)
            if res != None:
                images.append(imageio.imread(img_dir+res.group(0)))
    imageio.mimsave(img_dir + 'generation_animation.gif', images, fps=1)
    print(type(images))
    
#plot_img('F:/schoolwork/myDRGAN/snapshot/Single/2017-11-17_15-15-28', 20)