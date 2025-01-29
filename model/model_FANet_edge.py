import torch
import torch.nn as nn
from torch.autograd import Variable
import thop
import torch.nn.functional as F
from model.blocks import *
from timm.models.layers import trunc_normal_


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0.0)


class ACmix(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, c1, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        # fully connected layer in Fig.2
        self.fc = nn.Conv2d(3 * self.num_heads, 9, kernel_size=1, bias=True)
        # group convolution layer in Fig.3
        self.dep_conv = nn.Conv2d(9 * dim // self.num_heads, dim, kernel_size=3, bias=True,
                                  groups=dim // self.num_heads, padding=1)
        # rates for both paths
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        ones(self.rate1)
        ones(self.rate2)
        # shift initialization for group convolution
        kernel = torch.zeros(9, 3, 3)
        for i in range(9):
            kernel[i, i // 3, i % 3] = 1.
        kernel = kernel.squeeze(0).repeat(self.dim, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = zeros(self.dep_conv.bias)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, C, H, W) To (B, H, W, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x = x.permute(0, 2, 3, 1)
        _, H, W, _ = x.shape
        qkv = self.qkv(x)

        # fully connected layer
        f_all = qkv.reshape(x.shape[0], H * W, 3 * self.num_heads, -1).permute(0, 2, 1, 3)  # B, 3*nhead, H*W, C//nhead
        f_conv = self.fc(f_all).permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[-1] // self.num_heads, H, W)

        # group conovlution
        out_conv = self.dep_conv(f_conv).permute(0, 2, 3, 1)  # B, H, W, C

        # partition windows
        qkv = window_partition(qkv, self.window_size[0])  # nW*B, window_size, window_size, C

        B_, _, _, C = qkv.shape

        qkv = qkv.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C

        N = self.window_size[0] * self.window_size[1]
        C = C // 3

        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        # merge windows
        x = x.view(-1, self.window_size[0], self.window_size[1], C)
        x = window_reverse(x, self.window_size[0], H, W)  # B H' W' C

        x = self.rate1 * x + self.rate2 * out_conv

        x = self.proj_drop(x).permute(0, 3, 1, 2)
        return x


class FANet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter,deep_supervision=False, learn_loss_mask=False, cam=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.cam=cam
        # self.gc = GaussianCurvature(input_channels, nb_filter[0])
        self.gc = RiemannCurvatureExtractor(input_channels, nb_filter[0])

        self.neck4_e = nn.Sequential(
            nn.Conv2d(nb_filter[0], nb_filter[4], kernel_size=(16,16), stride=16),
            nn.BatchNorm2d(nb_filter[4]),
            nn.ReLU(),
        )
        self.neck22_e = nn.Sequential(
            nn.Conv2d(nb_filter[2], nb_filter[2], kernel_size=(1,1), stride=1),
            nn.BatchNorm2d(nb_filter[2]),
            nn.ReLU(),
        )
        self.neck04_e = nn.Sequential(
            nn.Conv2d(nb_filter[0], nb_filter[0], kernel_size=(1,1), stride=1),
            nn.BatchNorm2d(nb_filter[0]),
            nn.ReLU(),
        )
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0], num_blocks[0], stride=1)
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0], stride=2)
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1], stride=2)
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2], stride=2)
        self.conv4_0 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3], stride=2)# nb_filter[0]+nb_filter[1]+nb_filter[2]+
        self.conv4_e = self._make_layer(block, nb_filter[4]*2, nb_filter[2], num_blocks[2], stride=1)# nb_filter[0]+nb_filter[1]+nb_filter[2]+

        self.neck0_0 = BottleneckCSP(nb_filter[0], nb_filter[0])
        self.neck1_0 = BottleneckCSP(nb_filter[1], nb_filter[1])
        self.neck2_0 = BottleneckCSP(nb_filter[2], nb_filter[2])
        self.neck3_0 = BottleneckCSP(nb_filter[3], nb_filter[3])

        self.neck2_1 = BottleneckCSP(nb_filter[2], nb_filter[2])
        self.neck1_1 = BottleneckCSP(nb_filter[1], nb_filter[1])
        self.neck0_1 = BottleneckCSP(nb_filter[0], nb_filter[0])

        self.neck1_2 = BottleneckCSP(nb_filter[1], nb_filter[1])
        self.neck0_2 = BottleneckCSP(nb_filter[0], nb_filter[0])

        self.neck0_3 = BottleneckCSP(nb_filter[0], nb_filter[0])
        self.neck0_4 = BottleneckCSP(nb_filter[0], nb_filter[0])

        self.up2 = nn.Upsample(None, 2)
        self.up4 = nn.Upsample(None, 4)
        self.conv3_1 = self._make_layer(block, nb_filter[4]+ nb_filter[3],  nb_filter[3], num_blocks[2], stride=1)
        self.conv2_1 = self._make_layer(block, nb_filter[3]+ nb_filter[2],  nb_filter[2], num_blocks[2], stride=1)
        self.conv1_1 = self._make_layer(block, nb_filter[2]+ nb_filter[1],  nb_filter[1], num_blocks[2], stride=1)
        self.conv0_1 = self._make_layer(block, nb_filter[1]+ nb_filter[0],  nb_filter[0], num_blocks[2], stride=1)

        self.conv2_e = self._make_layer(block, nb_filter[2]*2, nb_filter[0], num_blocks[2], stride=1)
        self.conv2_2 = self._make_layer(block, nb_filter[3]*2+nb_filter[2],  nb_filter[2], num_blocks[2], stride=1)
        self.conv1_2 = self._make_layer(block, nb_filter[2]  +nb_filter[1],  nb_filter[1], num_blocks[2], stride=1)
        self.conv0_2 = self._make_layer(block, nb_filter[1]  +nb_filter[0],  nb_filter[0], num_blocks[2], stride=1)
        self.conv1_3 = self._make_layer(block, nb_filter[2]*2+nb_filter[1],  nb_filter[1], num_blocks[2], stride=1)
        self.conv0_3 = self._make_layer(block, nb_filter[1]  +nb_filter[0],  nb_filter[0], num_blocks[2], stride=1)
        self.conv0_4 = self._make_layer(block, nb_filter[1]*2+nb_filter[0],  nb_filter[0], num_blocks[2], stride=1)
        self.conv0_e = self._make_layer(block, nb_filter[0]*2, nb_filter[0], num_blocks[2], stride=1)
        self.acmix = ACmix(nb_filter[0], nb_filter[0], (4, 4), 8)

        self.conv0_4_final =  self._make_layer(block, nb_filter[0]*5, nb_filter[0])
        self.final_edge = nn.Conv2d(nb_filter[0]*2, num_classes, kernel_size=1)
        if self.deep_supervision:
            self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()


    def _make_layer(self, block, input_channels, output_channels, num_blocks=1, stride=1):
        layers = []
        layers.append(block(input_channels, output_channels, stride=stride))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        e0 = self.gc(input)
        x0_0 = self.conv0_0(input)  #  in (b 3 256 256)  x0_0 (b 16 256 256)
        x1_0 = self.conv1_0(x0_0)   # x1_0(b 32 128 128)
        x2_0 = self.conv2_0(x1_0)   # x2_0 (b 64 64 64)
        x3_0 = self.conv3_0(x2_0)    # x3_0 (b 128 32 32)
        x4_0 = self.conv4_0(x3_0)   # x4_0 (b 256 16 16)
        e1 = self.conv4_e(torch.cat([x4_0, self.neck4_e(e0)], 1))

        x3_1 = self.conv3_1(torch.cat([self.neck3_0(x3_0), self.up2(x4_0)], 1))
        x2_1 = self.conv2_1(torch.cat([self.neck2_0(x2_0), self.up2(x3_1)], 1))
        x1_1 = self.conv1_1(torch.cat([self.neck1_0(x1_0), self.up2(x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([self.neck0_0(x0_0), self.up2(x1_1)], 1))

        x2_2 = self.conv2_2(torch.cat([self.neck2_1(x2_1), self.up2(x3_0), self.up2(x3_1)], 1))   #self.up2(torch.cat([x3_0, x3_1],1))
        e2 = self.conv2_e(torch.cat([x2_2, self.neck22_e(self.up4(e1))], 1))
        x1_2 = self.conv1_2(torch.cat([self.neck1_1(x1_1), self.up2(x2_2)], 1))    # x1_3 (b 32 128 128)  -x1_1, x1_2, self.down(x0_3)
        x0_2 = self.conv0_2(torch.cat([self.neck0_1(x0_1), self.up2(x1_2)], 1)) # x0_4 (b 16 256 256)  # -x0_1, x0_2, x0_3,

        x1_3 = self.conv1_3(torch.cat([self.neck1_2(x1_2), self.up2(x2_0), self.up2(x2_2)], 1))    # x1_3 (b 32 128 128)  -x1_1, x1_2, self.down(x0_3)
        x0_3 = self.conv0_3(torch.cat([self.neck0_2(x0_2), self.up2(x1_2)], 1))

        x0_4 = self.conv0_4(torch.cat([self.neck0_3(x0_3), self.up2(x1_0), self.up2(x1_3)], 1))
        e3 = self.conv0_e(torch.cat([x0_4, self.neck04_e(self.up4(e2))], 1))

        x0_4_final = self.conv0_4_final(torch.cat([self.acmix(x0_4), self.neck0_3(x0_3), self.neck0_2(x0_2), self.neck0_1(x0_1), self.neck0_0(x0_0)], 1)) #, x0_3, x0_2, x0_1

        x0_4_final = x0_4_final + self.sigmoid(e3)*x0_4_final

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4_final)
            if self.cam:
                return [output1, output2, output3, output4, e3]
            return [output1, output2, output3, output4, e3] # , maskconv
        else:
            output = self.final(x0_4_final)
            return output



