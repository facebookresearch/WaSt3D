#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torchvision

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class VGG(torch.nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        # self.req_features= ['0', '5','10','19','28']
        self.req_features= [0, 5, 10 ,19, 28]
        self.req_features= [0, 5, 10, 19, 28]
        #self.req_features=  [0,5,10]# [19] #   #[5,19] # maybe also try 23 it is kinda inbetween 19 and 28
        #Since we need only the 5 layers in the model so we will be dropping all the rest layers from the features of the model
        self.model=torchvision.models.vgg19(pretrained=True).features[:29] #model will contain the first 29 layers
        #for layer_num,layer in enumerate(self.model): print("layer_num:", layer_num, layer)
        # self.model.to(torch.device('cuda'))
        print("\nModel:")
        print(self.model)
        print("\n")


    #x holds the input tensor(image) that will be feeded to each layer
    def forward(self,x):
        #initialize an array that wil hold the activations from the chosen layers
        features=[]
        #Iterate over all the layers of the mode
        for layer_num,layer in enumerate(self.model):
            #print("\n\nlayer_num:", layer_num)
            #activation of the layer will stored in x
            x=layer(x)
            #appending the activation of the selected layers and return the feature array
            #if (str(layer_num) in self.req_features):
            if layer_num in self.req_features:
                features.append(x)

        return features

def get_features(input_tensor, model, preprocessing=True):
    """
    For an input tensor, it returns the features of the input tensor using a pretrained VGG model
    :param input_tensor: input tensor of shape (channels, height, width)
    :param model: pretrained VGG model
    :return: features list of features
    """
    x = input_tensor
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=(112, 112))#(512,512)) # was (224,224)
    x = model(x)
    return x



def content_loss(features_gt, features_pred):
    """
    Compute VGG content loss on features of the input images and compare them with the ground truth image features
    :param features_gt: ground truth image features
    :param features_pred: predicted image features
    :return: content loss
    """

    # for feat in features_gt:
    #     print("feat_gt.shape:", feat.shape)
    # for feat in features_pred:
    #     print("feat_gen.shape:", feat.shape)

    # Compute the content loss
    content_loss = 0
    for i in range(len(features_gt)):
        content_loss += torch.mean((features_gt[i] - features_pred[i]) ** 2)

    return content_loss

# def style_loss(self, network_output, gt):
#     """
#     Compute VGG style loss on features of the input images and compare them with the ground truth image features
#     :param vgg: pretrained VGG model of class VGG.
#     :param network_output: predicted image
#     :param gt: ground truth image
#     :return: style loss
#     """
#     network_output = network_output.unsqueeze(0)
#     gt = gt.unsqueeze(0)

#     # print("gt.shape:", gt.shape)
#     # print("network_output.shape:", network_output.shape)
#     # print("gt_mean:", torch.mean(gt[0], axis=(-1, -2)))
#     # print("gt_std:", torch.std(gt[0], axis=(-1, -2)))
#     # print("network_output_mean:", torch.mean(network_output[0], axis=(-1, -2)))
#     # print("network_output_std:", torch.std(network_output[0], axis=(-1, -2)))

#     def calc_style_loss(gen,style):
#         #Calculating the gram matrix for the style and the generated image
#         channel,height,width=gen.shape

#         G=torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t())
#         A=torch.mm(style.view(channel,height*width),style.view(channel,height*width).t())

#         #Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss
#         style_l=torch.mean((G-A)**2)
#         return style_l

#     features_gt = self.model(gt)
#     features_pred = self.model(network_output)
#     # for feat in features_gt:
#     #     print("feat_gt.shape:", feat.shape)
#     # for feat in features_pred:
#     #     print("feat_gen.shape:", feat.shape)

#     style_loss=0
#     for gen,style in zip(features_pred, features_gt):
#         #extracting the dimensions from the generated image
#         style_loss+=calc_style_loss(gen,style)


#     return style_loss


def style_loss(features_gt, features_pred):
    """
    Compute VGG style loss on features of the input images and compare them with the ground truth image features
    :param features_pred: predicted image features
    :param features_gt: ground truth image features
    :return: style loss
    """

    def calc_style_loss(gen,style):
        #Calculating the gram matrix for the style and the generated image
        _,channel,height,width=gen.shape

        G=torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t())
        A=torch.mm(style.view(channel,height*width),style.view(channel,height*width).t())

        #Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss
        style_l=torch.mean((G-A)**2)
        return style_l



    style_loss=0
    for gen,style in zip(features_pred, features_gt):
        #extracting the dimensions from the generated image
        style_loss+=calc_style_loss(gen,style)


    return style_loss


# def tv_loss(img):
#     return 0.5 * (torch.abs(img[:, 1:, :] - img[:, :-1, :]).mean() +
#                   torch.abs(img[:, :, 1:] - img[:, :, :-1]).mean())


def tv_loss(img):
    return 0.5 * (torch.abs(img[..., 1:, :] - img[..., :-1, :]).mean() +
                  torch.abs(img[..., :, 1:] - img[..., :, :-1]).mean())
