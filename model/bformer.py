'''
    Dynamic_conv + transformer + ssim + Denosing
'''

import sys
sys.path.append('../')

from model_common import common
from model_common.bformer_transformerlayer import *


def make_model(args):
    return DTSD(args), 1


class DTSD(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DTSD, self).__init__()

        self.scale_idx = 0

        self.args = args

        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        self.sub_mean = common.MeanShift(args.rgb_range)  # sub = 减
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)  # add = 加

        self.head = nn.Sequential(
            conv(args.n_colors, n_feats, kernel_size),  # conv1
            common.ResBlock(conv, n_feats, 5, act=act),  # conv2
            common.ResBlock(conv, n_feats, 5, act=act),  # conv3
        )

        self.window_size = 48
        self.img_size = 96
        self.num_layers = 8
        self.num_heads = 9
        self.patch_dim = 3

        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]  # stochastic depth decay rule

        layers = [
            'horizontal', 'vertical', 'no_shift',
            'horizontal', 'vertical', 'no_shift',
            'horizontal', 'vertical'
        ]

        assert self.num_layers == len(layers)

        self.body1 = CBformerLayer2(dim=n_feats, img_size=self.img_size, win_size=self.img_size,
                                    num_heads=self.num_heads * 4, patch_dim=self.patch_dim * 2)
        self.body2 = nn.Sequential(
            WBformerLayer_with_DMlp(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads,
                                    patch_dim=self.patch_dim, window_size=self.window_size, layers=layers, drop_path=dpr)
        )

        self.body3 = CBformerLayer2(dim=n_feats, img_size=self.img_size, win_size=self.img_size,
                                    num_heads=self.num_heads * 4, patch_dim=self.patch_dim * 2)

        self.tail = conv(n_feats, args.n_colors, kernel_size)

    def forward(self, x):
        y = x

        x = self.head(x)

        x = self.body1(x)
        x = self.body2(x)
        x = self.body3(x)

        out = self.tail(x)
        # print("============")
        return y - out
