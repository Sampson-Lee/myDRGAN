#!/usr/bin/env python
# encoding: utf-8
'''
return accuracy of diacriminator
'''
import torch

def cal_acc(real_output, syn_output, id_label_tensor, pose_label_tensor, Nd):
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

def cal_acc(output_id, id_label_tensor):
    _, id_real_ans = torch.max(output_id, 1) # return (max, max_indices)
    id_real_precision = (id_real_ans==id_label_tensor).type(torch.FloatTensor).sum() / real_output.size()[0]
    # Variable(FloatTensor) -> Float
    id_real_precision = id_real_precision.data[0]
    return id_real_precision