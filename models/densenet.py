# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.route import *

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        result = nn.Sequential(*layers)
        result.requires_grad_(True)
        return result
    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0, normalizer = None,
                 out_classes = 100, p=None, info=None, LU=False, clip_threshold=1e10):
        super(DenseNet3, self).__init__()

        self.gradients = []
        self.activations = []
        self.handles_list = []
        self.integrad_handles_list = []
        self.integrad_scores = []
        self.integrad_calc_activations_mask = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pruned_activations_mask = []
        self.clip_threshold = clip_threshold

        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = int(n/2)
            block = BottleneckBlock
        else:
            block = BasicBlock
        self.block = block
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.avgp2d = nn.AvgPool2d(8)


        if p is None:
            self.fc = nn.Linear (in_planes, num_classes)
        else:
            if LU:
                self.fc = RouteLUNCH(in_planes, num_classes, p=p, info=info)
            else:
                self.fc = RouteDICE(in_planes, num_classes, p=p, info=info)

        self.in_planes = in_planes
        self.normalizer = normalizer

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def remove_handles(self):
        for handle in self.handles_list:
            handle.remove()
        self.handles_list.clear()
        self.activations = []
        self.gradients = []

    def _compute_taylor_scores(self, inputs, labels):
        self._hook_layers()
        outputs = self._forward(inputs)
        outputs[0, labels.item()].backward(retain_graph=True)

        first_order_taylor_scores = []
        self.gradients.reverse()

        for i, layer in enumerate(self.activations):
            first_order_taylor_scores.append(torch.mul(layer, self.gradients[i]))
        
        self.remove_handles()                
        return first_order_taylor_scores, outputs
    
    def _init_integrad_mask(self, inputs):
        self.integrad_calc_activations_mask = []
        _ = self._forward(inputs)
        for a in self.activations:
            self.integrad_calc_activations_mask.append(torch.ones(a.shape))
    
    def _calc_integrad_scores(self, inputs, labels, iterations):
        def forward_hook_relu(module, input, output):
            output = torch.mul(output, self.integrad_calc_activations_mask[len(self.activations)-1].to(self.device))
            return output

        self._hook_layers()
        initial_output = self._initialize_pruned_mask(inputs)
        output = self._forward(inputs)
        output[0, labels.item()].backward(retain_graph=True)

        original_activations = []
        for a in self.activations:
            original_activations.append(a.detach().clone())

        self._init_integrad_mask(inputs)
        mask_step = 1./iterations
        i = 0
        for module in self.modules():
            if isinstance(module, nn.AvgPool2d):
            # if isinstance(module, nn.ReLU):
               self.integrad_scores.append(torch.zeros(original_activations[i].shape).to(self.device))
               self.integrad_calc_activations_mask[i] = torch.zeros(self.integrad_calc_activations_mask[i].shape)
               self.integrad_handles_list.append(module.register_forward_hook(forward_hook_relu))

               for j in range(iterations+1):
                   self.integrad_calc_activations_mask[i] += j*mask_step
                   output = self._forward(inputs)
                   output[0, labels.item()].backward(retain_graph=True)
                   self.gradients.reverse()
                   self.integrad_scores[len(self.integrad_scores)-1] += self.gradients[i]
               self.integrad_scores[len(self.integrad_scores)-1] = self.integrad_scores[len(self.integrad_scores)-1]/(iterations+1) * original_activations[i]
               self.integrad_calc_activations_mask[i] = torch.ones(self.integrad_calc_activations_mask[i].shape)
               self.integrad_handles_list[0].remove()
               self.integrad_handles_list.clear()
               i += 1
        inte_scores = []
        for layer_scores in self.integrad_scores:
            inte_scores.append(layer_scores)
        self.integrad_scores = []
        self.remove_handles()       
        return inte_scores, output

    def _initialize_pruned_mask(self, inputs):
        output = self._forward(inputs)

        # initializing pruned_activations_mask
        for layer in self.activations:
            self.pruned_activations_mask.append(torch.ones(layer.size()).to(self.device))
        return output
    
    def _hook_layers(self):
        def backward_hook_relu(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].to(self.device))

        def forward_hook_relu(module, input, output):
            # mask output by pruned_activations_mask
            # In the first model(input) call, the pruned_activations_mask
            # is not yet defined, thus we check for emptiness
            if self.pruned_activations_mask:
              output = torch.mul(output, self.pruned_activations_mask[len(self.activations)].to(self.device)) #+ self.pruning_biases[len(self.activations)].to(self.device)
            self.activations.append(output.to(self.device))
            return output
        
        i = 0
        for module in self.modules():
            if isinstance(module, nn.AvgPool2d):
            # if isinstance(module, nn.ReLU):
            # if isinstance(module, resnet.BasicBlock):
                self.handles_list.append(module.register_forward_hook(forward_hook_relu))
                self.handles_list.append(module.register_backward_hook(backward_hook_relu))
            # elif isinstance(module, DenseBlock):
            #     self.handles_list.append(module.register_forward_hook(forward_hook_relu))
            #     self.handles_list.append(module.register_backward_hook(backward_hook_relu))

    def forward(self, x):
        out = self.features(x)
        out = self.avgp2d(out)
        #####clip####
        out = out.clip(max=self.clip_threshold)
        out = out.view(-1, self.in_planes)
        out = self.fc(out)
        return out

    def _forward(self, x):
        self.activations = []
        self.gradients = []
        self.zero_grad()
        
        out = self.features(x)
        out = self.avgp2d(out)
        out = out.view(-1, self.in_planes)
        out = self.fc(out)
        return out

    def features(self, x):
        if self.normalizer is not None:
            x = x.clone()
            x[:, 0, :, :] = (x[:, 0, :, :] - self.normalizer.mean[0]) / self.normalizer.std[0]
            x[:, 1, :, :] = (x[:, 1, :, :] - self.normalizer.mean[1]) / self.normalizer.std[1]
            x[:, 2, :, :] = (x[:, 2, :, :] - self.normalizer.mean[2]) / self.normalizer.std[2]

        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return out

    # function to extact the multiple features
    def feature_list(self, x):
        if self.normalizer is not None:
            x = x.clone()
            x[:,0,:,:] = (x[:,0,:,:] - self.normalizer.mean[0]) / self.normalizer.std[0]
            x[:,1,:,:] = (x[:,1,:,:] - self.normalizer.mean[1]) / self.normalizer.std[1]
            x[:,2,:,:] = (x[:,2,:,:] - self.normalizer.mean[2]) / self.normalizer.std[2]

        out_list = []
        out = self.conv1(x)
        out_list.append(out)
        out = self.trans1(self.block1(out))
        out_list.append(out)
        out = self.trans2(self.block2(out))
        out_list.append(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out_list.append(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)

        return self.fc(out), out_list

    def intermediate_forward(self, x, layer_index):
        if self.normalizer is not None:
            x = x.clone()
            x[:,0,:,:] = (x[:,0,:,:] - self.normalizer.mean[0]) / self.normalizer.std[0]
            x[:,1,:,:] = (x[:,1,:,:] - self.normalizer.mean[1]) / self.normalizer.std[1]
            x[:,2,:,:] = (x[:,2,:,:] - self.normalizer.mean[2]) / self.normalizer.std[2]

        out = self.conv1(x)
        if layer_index == 1:
            out = self.trans1(self.block1(out))
        elif layer_index == 2:
            out = self.trans1(self.block1(out))
            out = self.trans2(self.block2(out))
        elif layer_index == 3:
            out = self.trans1(self.block1(out))
            out = self.trans2(self.block2(out))
            out = self.block3(out)
            out = self.relu(self.bn1(out))
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        if self.normalizer is not None:
            x = x.clone()
            x[:,0,:,:] = (x[:,0,:,:] - self.normalizer.mean[0]) / self.normalizer.std[0]
            x[:,1,:,:] = (x[:,1,:,:] - self.normalizer.mean[1]) / self.normalizer.std[1]
            x[:,2,:,:] = (x[:,2,:,:] - self.normalizer.mean[2]) / self.normalizer.std[2]

        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        penultimate = self.relu(self.bn1(out))
        out = F.avg_pool2d(penultimate, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out), penultimate
