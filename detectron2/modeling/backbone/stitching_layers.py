import numpy as np
import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation

# def generate_stitch_layer(layer_type)
# _stitch_layer_type = {
#     "linear_conv1x1" : partial(nn.Conv2d, kernel_size=1, stride=1),
#     "linear_fc": partial(nn.Linear, bias=True),
#     "linear_conv1x1_"
# }
from .moe_layer import SimpleStitchMoE
from functools import partial


def generate_moe_layer(in_features, out_features, stitch_layer_classes, noisy_gating=True):
    return SimpleStitchMoE(in_features, out_features, stitch_layer_classes, noisy_gating=noisy_gating)

class FCNormActivation(nn.Module):
    def __init__(self, in_features=None, out_features=None):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm2d(out_features)
        self.act = nn.Hardswish()
    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.act(out)
        return out

class GlobalResponseNorm(nn.Module):
    """ Global Response Normalization layer
    """
    def __init__(self, dim, eps=1e-6, channels_last=True):
        super().__init__()
        print(f"---- Stitch Layer: {self.__class__.__name__}")
        self.eps = eps
        if channels_last:
            self.spatial_dim = (1, 2)
            self.channel_dim = -1
            self.wb_shape = (1, 1, 1, -1)
        else:
            self.spatial_dim = (2, 3)
            self.channel_dim = 1
            self.wb_shape = (1, -1, 1, 1)

        self.weight = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
    
class FCStitchingLayer(nn.Module):
    def __init__(self, in_features=None, out_features=None, cnn_to_vit=False):
        super().__init__()
        print(f"---- Stitch Layer: {self.__class__.__name__}")
        self.transform = nn.Linear(in_features, out_features)
        self.cnn_to_vit = cnn_to_vit
    def init_stitch_weights_bias(self, weight, bias):
        if isinstance(self.transform, nn.Conv2d) and len(weight.shape) == 2:
            weight = weight.view(*weight.shape, 1, 1)
        self.transform.weight.data.copy_(weight)
        self.transform.bias.data.copy_(bias)
    def forward(self, x):
        out = self.transform(x)
        if self.cnn_to_vit:
            B, C, H, W = out.shape
            out = out.reshape(B, C, -1).permute(0, 2, 1)
        return out

class FCBNStitchingLayer(nn.Module):
    def __init__(self, in_features=None, out_features=None, cnn_to_vit=False):
        super().__init__()
        self.transform = nn.Linear(in_features, out_features)
        self.cnn_to_vit = cnn_to_vit
        self.bn = nn.BatchNorm2d(out_features)
        print(f"---- Stitch Layer: {self.__class__.__name__}")
        
    def init_stitch_weights_bias(self, weight, bias):
        if isinstance(self.transform, nn.Conv2d) and len(weight.shape) == 2:
            weight = weight.view(*weight.shape, 1, 1)
        self.transform.weight.data.copy_(weight)
        self.transform.bias.data.copy_(bias)
    
    def forward(self, x):
        out = self.transform(x)
        out = self.bn(out)
        if self.cnn_to_vit:
                B, C, H, W = out.shape
                out = out.reshape(B, C, -1).permute(0, 2, 1)
        return out

class FCGRNStitchingLayer(nn.Module):
    def __init__(self, in_features=None, out_features=None, cnn_to_vit=False):
        super().__init__()
        self.transform = nn.Linear(in_features, out_features)
        self.cnn_to_vit = cnn_to_vit
        self.grn = GlobalResponseNorm(out_features)
        print(f"---- Stitch Layer: {self.__class__.__name__}")
        
    def init_stitch_weights_bias(self, weight, bias):
        if isinstance(self.transform, nn.Conv2d) and len(weight.shape) == 2:
            weight = weight.view(*weight.shape, 1, 1)
        self.transform.weight.data.copy_(weight)
        self.transform.bias.data.copy_(bias)
    
    def forward(self, x):
        out = self.transform(x)
        out = self.grn(out)
        if self.cnn_to_vit:
                B, C, H, W = out.shape
                out = out.reshape(B, C, -1).permute(0, 2, 1)
        return out


class FCGRNHardswishStitchingLayer(nn.Module):
    def __init__(self, in_features=None, out_features=None, cnn_to_vit=False):
        super().__init__()
        self.transform = nn.Linear(in_features, out_features)
        self.cnn_to_vit = cnn_to_vit
        self.grn = GlobalResponseNorm(out_features)
        self.act = nn.Hardswish()
        print(f"---- Stitch Layer: {self.__class__.__name__}")
        
    def init_stitch_weights_bias(self, weight, bias):
        if isinstance(self.transform, nn.Conv2d) and len(weight.shape) == 2:
            weight = weight.view(*weight.shape, 1, 1)
        self.transform.weight.data.copy_(weight)
        self.transform.bias.data.copy_(bias)
    
    def forward(self, x):
        out = self.transform(x)
        out = self.grn(out)
        out = self.act(out)
        if self.cnn_to_vit:
                B, C, H, W = out.shape
                out = out.reshape(B, C, -1).permute(0, 2, 1)
        return out


class FCHardswishStitchingLayer(nn.Module):
    def __init__(self, in_features=None, out_features=None, cnn_to_vit=False):
        super().__init__()
        self.transform = nn.Linear(in_features, out_features)
        self.cnn_to_vit = cnn_to_vit
        self.act = nn.Hardswish()
        print(f"---- Stitch Layer: {self.__class__.__name__}")
        
    def init_stitch_weights_bias(self, weight, bias):
        if isinstance(self.transform, nn.Conv2d) and len(weight.shape) == 2:
            weight = weight.view(*weight.shape, 1, 1)
        self.transform.weight.data.copy_(weight)
        self.transform.bias.data.copy_(bias)
    
    def forward(self, x):
        out = self.transform(x)
        out = self.act(out)
        if self.cnn_to_vit:
                B, C, H, W = out.shape
                out = out.reshape(B, C, -1).permute(0, 2, 1)
        return out

class FCReluStitchingLayer(nn.Module):
    def __init__(self, in_features=None, out_features=None, cnn_to_vit=False):
        super().__init__()
        self.transform = nn.Linear(in_features, out_features)
        self.cnn_to_vit = cnn_to_vit
        self.act = nn.ReLU()
        print(f"---- Stitch Layer: {self.__class__.__name__}")
        
    def init_stitch_weights_bias(self, weight, bias):
        if isinstance(self.transform, nn.Conv2d) and len(weight.shape) == 2:
            weight = weight.view(*weight.shape, 1, 1)
        self.transform.weight.data.copy_(weight)
        self.transform.bias.data.copy_(bias)
    
    def forward(self, x):
        out = self.transform(x)
        out = self.act(out)
        if self.cnn_to_vit:
                B, C, H, W = out.shape
                out = out.reshape(B, C, -1).permute(0, 2, 1)
        return out

class FCGeluStitchingLayer(nn.Module):
    def __init__(self, in_features=None, out_features=None, cnn_to_vit=False):
        super().__init__()
        self.transform = nn.Linear(in_features, out_features)
        self.cnn_to_vit = cnn_to_vit
        self.act = nn.GELU()
        print(f"---- Stitch Layer: {self.__class__.__name__}")
        
    def init_stitch_weights_bias(self, weight, bias):
        if isinstance(self.transform, nn.Conv2d) and len(weight.shape) == 2:
            weight = weight.view(*weight.shape, 1, 1)
        self.transform.weight.data.copy_(weight)
        self.transform.bias.data.copy_(bias)
    
    def forward(self, x):
        out = self.transform(x)
        out = self.act(out)
        if self.cnn_to_vit:
                B, C, H, W = out.shape
                out = out.reshape(B, C, -1).permute(0, 2, 1)
        return out

class FCBNHardswishStitchingLayer(nn.Module):
    def __init__(self, in_features=None, out_features=None, cnn_to_vit=False):
        super().__init__()
        # self.transform = nn.Conv2d(in_features, out_features, 1, 1)
        # self.cnn_to_vit = cnn_to_vit
        # self.bn = nn.BatchNorm2d(out_features)
        # self.act = nn.Hardswish()
        self.cnn_to_vit = cnn_to_vit
        self.layer = FCNormActivation(in_features, out_features)
        print(f"---- Stitch Layer: {self.__class__.__name__}")
        
    def init_stitch_weights_bias(self, weight, bias):
        print("Non initialization need in ConvBnAct Layer")
        pass
        # if isinstance(self.transform, nn.Conv2d) and len(weight.shape) == 2:
        #     weight = weight.view(*weight.shape, 1, 1)
        # self.transform.weight.data.copy_(weight)
        # self.transform.bias.data.copy_(bias)
    
    def forward(self, x):
        out = self.layer(x)
        if self.cnn_to_vit:
                B, C, H, W = out.shape
                out = out.reshape(B, C, -1).permute(0, 2, 1)
        return out

class BNFCBNStitchingLayer(nn.Module):
    def __init__(self, in_features=None, out_features=None, cnn_to_vit=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_features)
        self.transform = nn.Linear(in_features, out_features)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.cnn_to_vit = cnn_to_vit
        print(f"---- Stitch Layer: {self.__class__.__name__}")
    
    def init_stitch_weights_bias(self, weight, bias):
        print("Weight doesn't need to be initialized for this layer")
        pass
    
    def forward(self, x):
        out = self.bn1(x)
        out = self.transform(out)
        out = self.bn2(out)
        if self.cnn_to_vit:
                B, C, H, W = out.shape
                out = out.reshape(B, C, -1).permute(0, 2, 1)
        return out
        
        

class NonLinearBNResidualFCStitchingLayer(nn.Module):
    def __init__(self, in_features=None, out_features=None, cnn_to_vit=False):
        super().__init__()
        self.transform = nn.Linear(in_features, out_features)
        self.cnn_to_vit = cnn_to_vit
        print("--- Adding non linearity")
        self.non_linear_layer = FCNormActivation(in_features, out_features)
        print(f"---- Stitch Layer: {self.__class__.__name__}")
        self.init_nonlinear_layer()
    
    def init_nonlinear_layer(self):
        for m in self.non_linear_layer.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0)

    def init_stitch_weights_bias(self, weight, bias):
        if isinstance(self.transform, nn.Conv2d) and len(weight.shape) == 2:
            weight = weight.view(*weight.shape, 1, 1)
        self.transform.weight.data.copy_(weight)
        self.transform.bias.data.copy_(bias)

    def forward(self, x):
        out = self.transform(x)
        res = self.non_linear_layer(x)
        out = res + out
        if self.cnn_to_vit:
            B, C, H, W = out.shape
            out = out.reshape(B, C, -1).permute(0, 2, 1)
        return out


STITCH_LAYERS = {
    "fc" : FCStitchingLayer,
    "fc_bn": FCBNStitchingLayer,
    "fc_hardswish": FCHardswishStitchingLayer,
    "fc_relu": FCReluStitchingLayer,
    "fc_gelu": FCGeluStitchingLayer,
    "fc_bn_hardswish": FCBNHardswishStitchingLayer,
    "bn_fc_bn": BNFCBNStitchingLayer,
    "residual_fc_bn_hardswish_fc": NonLinearBNResidualFCStitchingLayer,
    "fc_grn" : FCGRNStitchingLayer,
    "fc_grn_hardswish": FCGRNHardswishStitchingLayer,
    "moe_linear_nonlinear": partial(generate_moe_layer, stitch_layer_classes=[FCStitchingLayer, FCGeluStitchingLayer], noisy_gating=False),
    "moe_noisy_linear_nonlinear": partial(generate_moe_layer, stitch_layer_classes=[FCStitchingLayer, FCGeluStitchingLayer], noisy_gating=True),
     "moe_linear_linear": partial(generate_moe_layer, stitch_layer_classes=[FCStitchingLayer, FCStitchingLayer], noisy_gating=False),
}
