import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init, xavier_init)
from torch.utils.checkpoint import checkpoint

from ..builder import NECKS


@NECKS.register_module()
class Deconv_FPN_Neck(nn.Module):
    """

    Args:
        in_channels (list): number of channels for each branch.
        out_channels (int): output channels of feature pyramids.
        num_outs (int): number of output stages.
        pooling_type (str): pooling for generating feature pyramids
            from {MAX, AVG}.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        with_cp  (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        stride (int): stride of 3x3 convolutional layers
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_deconv_layers=2,
                 num_deconv_filters=(256,256),
                 num_deconv_kernels=(4,4),
                 in_index=0,
                 extra=None,
                 input_transform=None,
                 align_corners=False):
        super(Deconv_FPN_Neck, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.align_corners = align_corners
        self.upsample_cfg = dict(mode='nearest')

        self.lateral_convs = nn.ModuleList()
        self.c_convs = nn.ModuleList()
        for i in range(num_deconv_layers+1):
            l_conv = build_conv_layer(
                        dict(type='Conv2d'),
                        in_channels=in_channels[i+1],
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1
                        )
            c_conv = build_conv_layer(
                        dict(type='Conv2d'),
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        )
            self.lateral_convs.append(l_conv)
            self.c_convs.append(c_conv)

        # if num_deconv_layers > 0:
        #     self.deconv_layers = self._make_deconv_layer(
        #         num_deconv_layers,
        #         num_deconv_filters,
        #         num_deconv_kernels,
        #     )

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        """Forward function."""
        x = x[1:]
        laterals = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)
        num = 0
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                # deconv_in = laterals[i]
                # for a in range(3):
                #     deconv_in =self.deconv_layers[a+(3*num)](deconv_in)
            
                # laterals[i - 1] += deconv_in
                # num+=1
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
        outs = [
            self.c_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        out = outs[0]

        return out
        
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding
    
