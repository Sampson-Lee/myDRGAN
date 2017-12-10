#!/usr/bin/env python
# encoding: utf-8

import torch
from torch import nn

class Discriminator(nn.Module):

    """
    multi-task CNN for identity and pose classification

    ### init
    Nd : Number of identitiy to classify
    Np : Number of pose to classify

    """

    def __init__(self, args):
        super(Discriminator, self).__init__()
        D_enc_convLayers = [
            nn.Conv2d(args.channel_num, 32, 3, 1, 1, bias=False), # Bxchx96x96 -> Bx32x96x96
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), # Bx32x96x96 -> Bx64x96x96
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx64x96x96 -> Bx64x97x97
            nn.Conv2d(64, 64, 3, 2, 0, bias=False), # Bx64x97x97 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx128x48x48
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx128x48x48 -> Bx128x49x49
            nn.Conv2d(128, 128, 3, 2, 0, bias=False), #  Bx128x49x49 -> Bx128x24x24
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 96, 3, 1, 1, bias=False), #  Bx128x24x24 -> Bx96x24x24
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.Conv2d(96, 192, 3, 1, 1, bias=False), #  Bx96x24x24 -> Bx192x24x24
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx192x24x24 -> Bx192x25x25
            nn.Conv2d(192, 192, 3, 2, 0, bias=False), # Bx192x25x25 -> Bx192x12x12
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.Conv2d(192, 128, 3, 1, 1, bias=False), # Bx192x12x12 -> Bx128x12x12
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), # Bx128x12x12 -> Bx256x12x12
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx256x12x12 -> Bx256x13x13
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x13x13 -> Bx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False), # Bx256x6x6 -> Bx160x6x6
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.Conv2d(160, args.Nf+1, 3, 1, 1, bias=False), # Bx160x6x6 -> Bx320x6x6
            nn.BatchNorm2d(args.Nf+1),
            nn.ELU(),
            nn.AvgPool2d(6, stride=1), #  Bx320x6x6 -> Bx320x1x1
        ]

        self.Denc = nn.Sequential(*D_enc_convLayers)
        self.fc = nn.Linear(320, args.Nd+args.Np+1)

    def forward(self, input):
        # 畳み込み -> 平均プーリングの結果 B x 320 x 1 x 1の出力を得る
        x = self.D_enc_convLayers(input)

        # バッチ数次元を消さないように１次元の次元を削除　
        x = x.squeeze(2)
        x = x.squeeze(2)
        
        x = x[:,:-1] # nBx321 -> nBx320
        
        # 全結合
        x = self.fc(x) # Bx320 -> B x (Nd+1+Np)

        return x


## nn.Module を継承しても， super でコンストラクタを呼び出さないと メンバ変数 self._modues が
## 定義されずに後の重み初期化の際にエラーを出す
## self._modules はモジュールが格納するモジュール名を格納しておくリスト

class Crop(nn.Module):

    """
    Generator でのアップサンプリング時に， ダウンサンプル時のZeroPad2d と逆の事をするための関数
    論文著者が Tensorflow で padding='SAME' オプションで自動的にパディングしているのを
    ダウンサンプル時にはZeroPad2dで，アップサンプリング時には Crop で実現

    ### init
    crop_list : データの上下左右をそれぞれどれくらい削るか指定
    """

    def __init__(self, crop_list):
        super(Crop, self).__init__()

        # crop_lsit = [crop_top, crop_bottom, crop_left, crop_right]
        self.crop_list = crop_list

    def forward(self, x):
        B,C,H,W = x.size()
        x = x[:,:, self.crop_list[0] : H - self.crop_list[1] , self.crop_list[2] : W - self.crop_list[3]]

        return x


def WSum_feature(x, n):
    """
    Apply the sigmoid function only to the output part of the weight 
    and add the output results of the n images using the weight 
    Input x: nB x 321 -> Output features_summed: B x 320

    n : 一人にあたり何枚の画像をデータとして渡しているのか
    B : バッチ毎に何人分(１人n枚)の画像をデータとして渡しているのか

    """
    B = x.size(0)//n
    weight = x[:,-1].unsqueeze(1).sigmoid()
    features = x*weight # (nB x 321) element-product (nB x 1) = (nB x 321)
    features = features[:,:-1].split(n, 0) # (nB x 320)->[(n x 320),...] features is a tuple
    features = torch.cat(features,1)    # [(n x 320),...]->(n x 320B)
    features_summed = features.sum(0, keepdim=True) # (n x 320B)->(1 x 320B)
    features_summed = features_summed.view(B,-1)    # (320B)->(B x 320)
    return features_summed

def myWSum_feature(x, n):
    B = x.size(0)//n
    weight = x[:,-1].unsqueeze(1).sigmoid()
    features = x*weight # (nB x 321) element-product (nB x 1) = (nB x 321)
    features = features[:,:-1].view(B, n, -1) # (nB x 320)->(B x n x 320)
    features_summed = features.sum(1, keepdim=True) # (B x n x 320)->(B x 320)
    features_summed = features_summed.view(4,-1)
    return features_summed

#x1 = torch.FloatTensor(np.random.randn(12, 4))
#b = WSum_feature(x1, 3)
#c = WSum_feature(x1, 3)

class Generator(nn.Module):

    """
    Encoder/Decoder conditional GAN conditioned with pose vector and noise vector

    ### init
    Np : Dimension of pose vector (Corresponds to number of dicrete pose classes of the data)
    Nz : Dimension of noise vector
    n  : Number of images per person

    """

    def __init__(self, args):
        super(Generator, self).__init__()
        self.images_perID = args.images_perID

        G_enc_convLayers = [
            nn.Conv2d(args.channel_num, 32, 3, 1, 1, bias=False), # nBxchx96x96 -> nBx32x96x96
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), # nBx32x96x96 -> nBx64x96x96
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # nBx64x96x96 -> nBx64x97x97
            nn.Conv2d(64, 64, 3, 2, 0, bias=False), # nBx64x97x97 -> nBx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), # nBx64x48x48 -> nBx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), # nBx64x48x48 -> nBx128x48x48
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # nBx128x48x48 -> nBx128x49x49
            nn.Conv2d(128, 128, 3, 2, 0, bias=False), #  nBx128x49x49 -> nBx128x24x24
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 96, 3, 1, 1, bias=False), #  nBx128x24x24 -> nBx96x24x24
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.Conv2d(96, 192, 3, 1, 1, bias=False), #  nBx96x24x24 -> nBx192x24x24
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # nBx192x24x24 -> nBx192x25x25
            nn.Conv2d(192, 192, 3, 2, 0, bias=False), # nBx192x25x25 -> nBx192x12x12
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.Conv2d(192, 128, 3, 1, 1, bias=False), # nBx192x12x12 -> nBx128x12x12
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), # nBx128x12x12 -> nBx256x12x12
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # nBx256x12x12 -> nBx256x13x13
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # nBx256x13x13 -> nBx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False), # nBx256x6x6 -> nBx160x6x6
            nn.BatchNorm2d(160),
            nn.ELU(),

            # add one more channel in features before AvgPool to estimate the coefficient w
            nn.Conv2d(160, args.Nf+1, 3, 1, 1, bias=False), # nBx160x6x6 -> nBx321x6x6
            nn.BatchNorm2d(args.Nf+1),
            nn.ELU(),
            nn.AvgPool2d(6, stride=1), #  nBx321x6x6 -> nBx321x1x1

        ]
        self.Genc = nn.Sequential(*G_enc_convLayers)

        G_dec_convLayers = [
            nn.ConvTranspose2d(args.Nf, 160, 3,1,1, bias=False), # Bx320x6x6 -> Bx160x6x6
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.ConvTranspose2d(160, 256, 3,1,1, bias=False), # Bx160x6x6 -> Bx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ConvTranspose2d(256, 256, 3,2,0, bias=False), # Bx256x6x6 -> Bx256x13x13
            nn.BatchNorm2d(256),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(256, 128, 3,1,1, bias=False), # Bx256x12x12 -> Bx128x12x12
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 192,  3,1,1, bias=False), # Bx128x12x12 -> Bx192x12x12
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ConvTranspose2d(192, 192,  3,2,0, bias=False), # Bx128x12x12 -> Bx192x25x25
            nn.BatchNorm2d(192),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(192, 96,  3,1,1, bias=False), # Bx192x24x24 -> Bx96x24x24
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.ConvTranspose2d(96, 128,  3,1,1, bias=False), # Bx96x24x24 -> Bx128x24x24
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 128,  3,2,0, bias=False), # Bx128x24x24 -> Bx128x49x49
            nn.BatchNorm2d(128),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(128, 64,  3,1,1, bias=False), # Bx128x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64,  3,1,1, bias=False), # Bx64x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64,  3,2,0, bias=False), # Bx64x48x48 -> Bx64x97x97
            nn.BatchNorm2d(64),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(64, 32,  3,1,1, bias=False), # Bx64x96x96 -> Bx32x96x96
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.ConvTranspose2d(32, args.channel_num,  3,1,1, bias=False), # Bx32x96x96 -> Bxchx96x96
            nn.Tanh(),
        ]

        self.Gdec = nn.Sequential(*G_dec_convLayers)

        self.dec_fc = nn.Linear(args.Nf+args.Np+args.Nz, 320*6*6)

    def forward(self, input, pose=None, noise=None, single=False, extract=False):

        x = self.Genc(input) # nBx1x96x96 -> Bx321x1x1

        x = x.squeeze(2)
        x = x.squeeze(2)

        # print(x.size())
        if single:
            # 足し合わせない場合
            x = x[:,:-1] # nBx321 -> nBx320
        else:
            # 同一人物の画像の特徴量を重みを用いて足し合わせる
            x = WSum_feature(x, self.images_perID) # nBx321 -> Bx320

        if extract: return x

        # print([x.size(), pose.size(), noise.size()])
        x = torch.cat([x, pose, noise], 1)  # B(nB)x320 -> B(nB) x (320+Np+Nz)

        x = self.dec_fc(x) # B(nB) x (320+Np+Nz) -> B(nB) x (320x6x6)

        x = x.view(-1, 320, 6, 6) # B(nB) x (320x6x6) -> B(nB) x 320 x 6 x 6

        x = self.Gdec(x) #  B(nB) x 320 x 6 x 6 -> B(nB)x1x96x96

        return x
