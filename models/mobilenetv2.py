import torch
from torch import nn
from torch import Tensor
from torchvision._internally_replaced_utils import load_state_dict_from_url
from typing import Callable, Any, Optional, List
from models.route import *
from torch.autograd import Variable


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6

        super().__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False),
                         norm_layer(out_planes),
                         activation_layer(inplace=True))
        self.out_channels = out_planes


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        p=None, info=None, clip_threshold=1e10, LU = True
        ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()
        self.gradients = []
        self.activations = []
        self.handles_list = []
        self.integrad_handles_list = []
        self.integrad_scores = []
        self.integrad_calc_activations_mask = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pruned_activations_mask = []
        self.clip_threshold = clip_threshold

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))

        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.classifier = nn.functional.adaptive_avg_pool2d(Variable, (1, 1))
        
        if p is None:
            self.classifier = nn.Sequential(nn.Dropout(0.2),
                nn.Linear(self.last_channel, num_classes))
        else:
            if LU:
                self.classifier = nn.Sequential(nn.Dropout(0.2),
                    RouteLUNCH(self.last_channel, num_classes, p=p, info=info, clip_threshold = clip_threshold))
            else:
                self.classifier = nn.Sequential(nn.Dropout(0.2),
                    RouteDICE(self.last_channel, num_classes, p=p, info=info, clip_threshold = clip_threshold))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward(self, x: Tensor) -> Tensor:
        self.activations = []
        self.gradients = []
        self.zero_grad()        
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = self.avgpool(x)
        # x = apply_ash(x, method=getattr(self, 'ash_method'))
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.clip(max=self.clip_threshold)
        # x = torch.where(x == self.clip_threshold,1,0) # number of clipped values in penultimate layer
        # x = torch.where(x > 0,1,0)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
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



def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
