import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import thop
import numpy as np
from einops import rearrange
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft
import math
# from mish_cuda import MishCuda as Mish


import torch
import torch.nn as nn


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

    def __init__(self, kernel_size=7):
        super(HighFreqAttention, self).__init__()
        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1
        # self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        # self.sigmoid = nn.Sigmoid()
        self.patch_factor = 2

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def freq2tensor(self, freq):
        # perform 2D inverse DFT
        if IS_HIGH_VERSION:
            y = torch.fft.ifft2(torch.view_as_complex(freq), norm='ortho')
        else:
            y = torch.irfft(freq, 2, onesided=False, normalized=True)

        patch_factor = self.patch_factor
        patch_list = torch.split(y, 1, 1)

        b, _, _, patch_h, patch_w = patch_list[0].shape
        h = patch_h * patch_factor
        w = patch_w * patch_factor
        x = torch.zeros(y.shape[0], y.shape[2], h, w, device=freq.device)

        for i in range(patch_factor):
            for j in range(patch_factor):
                x[:, :, i * patch_h: (i + 1) * patch_h, j * patch_w: (j + 1) * patch_w] = patch_list[i * patch_factor + j][:, 0, ...]
        return x

    def generate_hanning_window(self, M, N):
        hanning_window_1D_M = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(M) / M))
        hanning_window_1D_N = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(N) / N))
        hanning_window_2D = np.sqrt(np.outer(hanning_window_1D_M, hanning_window_1D_N))
        return torch.tensor(hanning_window_2D)

    def forward(self, x):
        freq = self.tensor2freq(x)
        B, _, _, M, N, _ = freq.shape
        hanning_weight = self.generate_hanning_window(M, N)
        hanning_weight = (1. - hanning_weight).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1, 1, 1)
        attn_freq = freq*hanning_weight#.cuda()
        attn_map = self.freq2tensor(attn_freq)
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat([avg_out, max_out], dim=1)
        # x = self.conv1(x)
        return self.sigmoid(attn_map)

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x): # 输入输出同尺度
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

class GaussianCurvature(nn.Module):
    def __init__(self,in_channels, out_channels, stride=1):
        super(GaussianCurvature, self).__init__()
        self.dilation = 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.gx = torch.tensor([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]]).expand(out_channels, out_channels,-1,-1).float().cuda()
        self.gy = torch.tensor([[[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]]]).expand(out_channels, out_channels,-1,-1).float().cuda()
        self.gxy = torch.tensor([[[[-1, 0, 1], [0, 0, 0], [1, 0, -1]]]]).expand(out_channels, out_channels,-1,-1).float().cuda()
        self.gxx = torch.tensor([[[[1, -1, 1], [1, -1, 1], [1, -1, 1]]]]).expand(out_channels, out_channels,-1,-1).float().cuda()
        self.gyy = torch.tensor([[[[1, 1, 1], [-1, -1, -1], [1, 1, 1]]]]).expand(out_channels, out_channels,-1,-1).float().cuda()

    def forward(self, input):
        x = self.conv(input)
        input_x = F.conv2d(x, self.gx, padding=self.dilation, dilation=self.dilation)
        input_y = F.conv2d(x, self.gy, padding=self.dilation, dilation=self.dilation)
        input_xy = F.conv2d(x, self.gxy, padding=self.dilation, dilation=self.dilation)
        input_xx = F.conv2d(x, self.gxx, padding=self.dilation, dilation=self.dilation)
        input_yy = F.conv2d(x, self.gyy, padding=self.dilation, dilation=self.dilation)
        k = (input_xx*input_yy-input_xy*input_xy)/((1+input_x*input_x+input_y*input_y)*(1+input_x*input_x+input_y*input_y))

        return k # x1*x2*x3

class Freq_Shuffle_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(Freq_Shuffle_Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = out_channels // 2
        # assert (self.stride != 1) or (inp == branch_features << 1)
        self.two_braches = (self.stride > 1) or (in_channels != branch_features << 1)

        if stride != 1 or branch_features != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, branch_features, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(branch_features))
        else:
            self.shortcut = nn.Identity()

        if self.two_braches:  # first block in each stage s=2
            self.branch1 = nn.Sequential(
                self.depthwise_conv(in_channels, in_channels, kernel_size=3, stride=self.stride, padding=1),
                norm_layer(in_channels),
                nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0, bias=False),  # pointwise
                norm_layer(branch_features),
                nn.ReLU(inplace=True),
                self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=1, padding=1),
                norm_layer(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),  # pointwise
                norm_layer(branch_features),
            )
            self.ca = ChannelAttention(branch_features)
            self.sa = SpatialAttention()
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels if self.two_braches else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(branch_features),
            nn.ReLU(inplace=True),
            # RiemannCurvatureExtractor(),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),    # depthwise (or padding=1
            norm_layer(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),    # pointwise
            norm_layer(branch_features),
            nn.ReLU(inplace=True),
        )
        # self.branch2 = LaplacianBranch(in_channels if self.two_braches else branch_features, branch_features, stride)
        # self.branch2 = BasicRFB_a(in_channels if self.two_braches else branch_features, branch_features, stride)

        self.relu = nn.ReLU()


    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        def branch1(x):
            x_ = self.branch1(x)
            x_ = self.ca(x_) * x_
            x_ = self.sa(x_) * x_
            return self.relu(x_ + self.shortcut(x))
        if self.two_braches:
            out = torch.cat((branch1(x), self.branch2(x)), dim=1)  # b 116 10 10
        else:
            x1, x2 = x.chunk(2, dim=1)  # ?? b 116 10 10 -> 2* b 58 10 10   b 464 3 3
            out = torch.cat((x1, self.branch2(x2)), dim=1)  # ?? b 116 10 10
        out = channel_shuffle(out, 2)    # channel shuffle attn
        return out
def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()  # input b 116 10 10
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)       # b 2 58 10 10

    x = torch.transpose(x, 1, 2).contiguous()   # b 58 2 10 10

    # flatten
    x = x.view(batchsize, -1, height, width)     # b 116 10 10

    return x

class RiemannCurvatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RiemannCurvatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        # 定义可学习的滤波器大小
        self.filter_size = 5
        self.learnable_filter_x = nn.Parameter(torch.randn(1, 1, self.filter_size, self.filter_size))
        self.learnable_filter_y = nn.Parameter(torch.randn(1, 1, self.filter_size, self.filter_size))

        # 初始化滤波器为 Sobel 核
        sobel_x = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], dtype=torch.float32)
        self.learnable_filter_x.data = sobel_x
        self.learnable_filter_y.data = sobel_y

    def forward(self, feature_map):
        feature_map = self.conv(feature_map)

        batch_size, channels, width, height = feature_map.size()

        feature_map_fft = torch.fft.fft2(feature_map, dim=(-2, -1))
        filter_x_fft = torch.fft.fft2(self.learnable_filter_x, s=(width, height), dim=(-2, -1))
        filter_y_fft = torch.fft.fft2(self.learnable_filter_y, s=(width, height), dim=(-2, -1))

        grad_x_fft = feature_map_fft * filter_x_fft  # 水平方向的梯度
        grad_y_fft = feature_map_fft * filter_y_fft  # 垂直方向的梯度

        grad_x = torch.fft.ifft2(grad_x_fft, dim=(-2, -1)).real
        grad_y = torch.fft.ifft2(grad_y_fft, dim=(-2, -1)).real

        g_xx = grad_x ** 2  # g_xx
        g_yy = grad_y ** 2  # g_yy
        g_xy = grad_x * grad_y  # g_xy (交叉项)

        second_grad_xx = torch.fft.fft2(grad_x.view(batch_size * channels, 1, width, height), dim=(-2, -1)) * filter_x_fft
        second_grad_yy = torch.fft.fft2(grad_y.view(batch_size * channels, 1, width, height), dim=(-2, -1)) * filter_y_fft

        second_grad_xx = torch.fft.ifft2(second_grad_xx, dim=(-2, -1)).real
        second_grad_yy = torch.fft.ifft2(second_grad_yy, dim=(-2, -1)).real

        second_grad_xx = second_grad_xx.view(batch_size, channels, width, height)
        second_grad_yy = second_grad_yy.view(batch_size, channels, width, height)

        curvature = second_grad_xx + second_grad_yy + g_xx + g_yy - 2 * g_xy
        curvature+=feature_map
        
        return curvature
