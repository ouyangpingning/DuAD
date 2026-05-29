"""WideResNet50-2 主干网络 + _embed 特征聚合。

复用自 torchvision ResNet 实现，添加 WideResNet50-2 的 factory 函数
以及 SimpleNet 论文使用的 _embed 特征聚合流程。
"""

from typing import Any, Type, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


# ========== torchvision ResNet 基础组件 ==========

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)    # /4
        x2 = self.layer2(x1)   # /8
        x3 = self.layer3(x2)   # /16
        x4 = self.layer4(x3)   # /32
        return [x1, x2, x3, x4]

    def forward(self, x):
        return self._forward_impl(x)


model_urls = {
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
}


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


# ========== _embed 特征聚合 (SimpleNet 原版) ==========

def patchify(feature: torch.Tensor, patchsize: int, stride: int):
    padding = int((patchsize - 1) / 2)
    unfolder = torch.nn.Unfold(
        kernel_size=patchsize, stride=stride, padding=padding, dilation=1
    )
    unfolded_features = unfolder(feature)
    number_of_total_patches = []
    for s in feature.shape[-2:]:
        n_patches = int((s + 2 * padding - 1 * (patchsize - 1) - 1) / stride + 1)
        number_of_total_patches.append(n_patches)
    unfolded_features = unfolded_features.reshape(
        *feature.shape[:2], patchsize, patchsize, -1
    )
    unfolded_features = unfolded_features.permute([0, 4, 1, 2, 3])
    return unfolded_features, number_of_total_patches


def Align_patches(feature: torch.Tensor, number_of_total_patches: list, target_patches: int):
    target_size = int(math.sqrt(target_patches))
    _feature = feature.reshape(
        len(feature), number_of_total_patches[0], number_of_total_patches[1],
        *feature.shape[2:]
    )
    _feature = _feature.permute(0, -3, -2, -1, 1, 2)
    temp_feature = _feature.shape
    _feature = _feature.reshape(-1, *_feature.shape[-2:])

    _feature = F.interpolate(
        _feature.unsqueeze(1),
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    _feature = _feature.squeeze(1)
    _feature = _feature.reshape(*temp_feature[:-2], target_size, target_size)
    _feature = _feature.permute(0, -2, -1, 1, 2, 3)
    _feature = _feature.reshape(len(_feature), -1, *_feature.shape[-3:])
    return _feature


def Align_dim(feature: torch.Tensor, target_dim: int):
    _feature = feature.reshape(-1, *feature.shape[-3:])
    _feature = _feature.reshape(len(_feature), 1, -1)
    return F.adaptive_avg_pool1d(_feature, target_dim).squeeze(1)


def _embed(features, layers: list, patchsize: int, stride: int,
           target_patches: int, target_dim: int, output_size: int):
    """SimpleNet 原版特征聚合。

    对指定 layers 的 feature map 依次做:
      patchify → Align_patches → Align_dim
    然后 stack → 展平 → adaptive_avg_pool1d 到 output_size.
    """
    patch_shapes = []
    Align_dim_features = []

    for layer_idx in layers:
        feature = features[layer_idx]
        Unfolded_feature, n_patches = patchify(feature, patchsize, stride)
        patch_shapes.append(n_patches)

        if target_patches is None:
            target_patches = patch_shapes[0][0] ** 2
        Aligned_feature = Align_patches(Unfolded_feature, n_patches, target_patches=target_patches)
        Align_dim_feature = Align_dim(Aligned_feature, target_dim=target_dim)
        Align_dim_features.append(Align_dim_feature)

    _features = torch.stack(Align_dim_features, dim=1)
    _features = _features.reshape(len(_features), 1, -1)
    _features = F.adaptive_avg_pool1d(_features, output_size)
    _features = _features.reshape(len(_features), -1)
    return _features, patch_shapes
