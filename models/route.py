import torch
import torch.nn as nn
import numpy as np
from collections import Counter


class RouteDICE(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, p=90, conv1x1=False, info=None):
        super(RouteDICE, self).__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
        self.p = p
        self.info = info
        self.masked_w = None

    def calculate_mask_weight(self):
        # self.contrib = self.info[None, :] * self.weight.data.cpu().numpy()
        self.contrib = abs(self.info[None, :]) * self.weight.data.cpu().numpy()
        # self.contrib = np.abs(self.contrib)
        # self.contrib = np.random.rand(*self.contrib.shape)
        # self.contrib = self.info[None, :]
        # self.contrib = np.random.rand(*self.info[None, :].shape)
        self.thresh = np.percentile(self.contrib, self.p)
        mask = torch.Tensor((self.contrib > self.thresh))
        self.masked_w = (self.weight.squeeze().cpu() * mask).cuda()

    def forward(self, input):
        if self.masked_w is None:
            self.calculate_mask_weight()
        vote = input[:, None, :] * self.masked_w.cuda()
        if self.bias is not None:
            out = vote.sum(2) + self.bias
        else:
            out = vote.sum(2)
        return out

class RouteLUNCH(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, p=90, conv1x1=False, info=None, clip_threshold = 1e10):
        super(RouteLUNCH, self).__init__(in_features, out_features, bias)
        self.p = p
        self.weight_p = p
        self.clip_threshold = clip_threshold
        self.info = info
        self.masked_w = None
        self.mask_f = None
        self.l_weight = self.weight.data.cuda()

    def calculate_shap_value(self):
        #### naive shap####
        # self.contrib = abs(self.info[None, :]) * self.weight.data.cpu().numpy() #w = [100, 342]
        # self.thresh = np.percentile(self.contrib, self.p)
        # mask = torch.Tensor((self.contrib > self.thresh))
        # self.masked_w = (self.weight.detach().squeeze().cpu() * mask).cuda()
        #### shap matrix only####
        # self.contrib = self.info.T
        # self.contrib = self.info.T / self.info.T.mean(1)[:,np.newaxis]
        #### shap matrix * weight####
        # self.contrib = self.info.T * self.weight.data.cpu().numpy() #w = [100, 342]
        #### standardize ####
        # self.contrib = self.info.T / self.info.T.mean(1)[:,np.newaxis]
        # self.masked_w = torch.zeros((self.out_features,self.out_features,self.in_features))
        # for class_num in range(self.out_features):
        #     self.matrix = self.contrib[class_num,:] * self.weight.data.cpu().numpy()
        #     self.thresh = np.percentile(self.matrix, self.p)
        #     mask = torch.Tensor((self.matrix > self.thresh))
        #     self.masked_w[class_num,:,:] = (self.weight.squeeze().cpu() * mask).cuda()
        #### naive std ####
        # self.contrib = self.info.T.mean(0)[None, :] * self.weight.data.cpu().numpy() #w = [100, 342]
        # self.thresh = np.percentile(self.contrib, self.p)
        # mask = torch.Tensor((self.contrib > self.thresh))
####################### CP ################
        # self.contrib = self.info.T
        # # self.contrib = self.info.T / self.info.T.mean(1)[:,np.newaxis]
        # self.mask = torch.zeros(self.out_features,self.in_features)
        # for i in range(self.out_features):
        #     self.class_thresh = np.percentile(self.contrib[i,:], self.p)
        #     self.mask[i,:] = torch.Tensor((self.contrib[i,:] > self.class_thresh))
######################## CP + DICE ###########################
        self.info = self.info.T
        self.contrib = self.info[10,:,0,:] # last layer cls contribution only
        # self.contrib = self.info.T / self.info.T.mean(1)[:,np.newaxis]
        self.mask_f = torch.zeros(1000,768)
        self.masked_w = torch.zeros((1000,1000,768))

        for class_num in range(self.out_features):
            self.matrix = abs(self.contrib[class_num,:]) * self.weight.data.cpu().numpy()
            self.thresh = np.percentile(self.matrix, self.weight_p)
            mask_w = torch.Tensor((self.matrix > self.thresh))
            self.masked_w[class_num,:,:] = (self.weight.squeeze().cpu() * mask_w).cuda()
            self.class_thresh = np.percentile(self.contrib[class_num,:], self.p)
            self.mask_f[class_num,:] = torch.Tensor((self.contrib[class_num,:] > self.class_thresh))
################################weight###########################    
        # self.thresh = np.percentile(self.weight.detach().squeeze().cpu(), self.p)
        # mask = torch.Tensor((self.weight.detach().squeeze().cpu() > self.thresh))
###########################################################

    def forward(self, input):
          
        if self.masked_w is None:
            self.calculate_shap_value()
        pre = input[:, None, :] * self.weight.data.cuda()
        if self.bias is not None:
            pred = pre.sum(2) + self.bias
        else:
            pred = pre.sum(2)
        pred = torch.nn.functional.softmax(pred, dim=1)   
        preds = np.argmax(pred.cpu().detach().numpy(), axis=1)
       
        counter_cp = 0
        cp = torch.zeros((len(input), self.in_features)).cuda()
        for idx in preds:
            cp[counter_cp,:] = input[counter_cp,:] * self.mask_f[idx,:].cuda()     
            counter_cp = counter_cp + 1

        vote = torch.zeros((len(preds),self.out_features,self.in_features)).cuda()
        counter_dice = 0
        for idx in preds:
            vote[counter_dice,:,:] = cp[counter_dice,:] * self.masked_w[idx,:,:].cuda()
            counter_dice = counter_dice + 1
        
        if self.bias is not None:
            out = vote.sum(2) + self.bias
        else:
            out = vote.sum(2)    
        return out