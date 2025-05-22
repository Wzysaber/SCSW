import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import flop_count_table
from fvcore.nn.flop_count import FlopCountAnalysis

from model_test.DSENet.Decoder import SegFormerHead
from model_test.DSENet import mix_transformer
from model_test.DSENet.CAM_Fusion import CAM_Fusion

import numpy as np

class TSCD(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        # ## initilize encoder
        if pretrained:
            print("Loading pth_model!")
            state_dict = torch.load(
                "/home0/students/master/2022/wangzy/Pycharm-Remote(161)/WSSS_Model2/pretrained/checkpoints/mit_b4.pth")
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.dropout = torch.nn.Dropout2d(0.5)
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        # self.decoder = conv_head.LargeFOV(self.in_channels[-1], out_planes=self.num_classes)

        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes - 1, kernel_size=1,
                                    bias=False)
        self.classifier2 = nn.Conv2d(in_channels=self.in_channels[2], out_channels=self.num_classes - 1, kernel_size=1,
                                     bias=False)

        self.CAM_Fusion = CAM_Fusion(in_features=[self.in_channels[2], self.in_channels[3]])

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.classifier2.weight)
        param_groups[2].append(self.attn_proj.weight)
        param_groups[2].append(self.attn_proj.bias)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    def forward(self, x, cam_only=False, seg_detach=True, aux=False):

        _x, _attns = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x

        segs = self.decoder(_x)

        if cam_only:
            cam = F.conv2d(_x4, self.classifier.weight).detach()
            return cam

        # _x4 = self.dropout(_x4.clone()
        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes - 1)

        aux_cls = self.pooling(_x3, (1, 1))
        aux_cls = self.classifier2(aux_cls)
        aux_cls = aux_cls.view(-1, self.num_classes - 1)

        if aux:
            return cls_x4, segs, _attns

        return cls_x4, aux_cls, segs, _x


if __name__ == "__main__":
    model = TSCD('mit_b1', stride=[4, 2, 2, 1], pooling="gmp", num_classes=6, embedding_dim=256, pretrained=False)

    dummy_input = torch.rand(1, 3, 512, 512)

    cls_x4, aux_cls, segs, _x4 = model(dummy_input)

    # total_params = sum([np.prod(p.size()) for p in model.parameters()])
    # print("Total network parameters (excluding idr):%.2fM" % (total_params / 1e6))
    # flops = FlopCountAnalysis(model, dummy_input)
    # print(flop_count_table(flops))

    print("output:", cls_x4.shape)
    print("output:", aux_cls.shape)
    print("output:", segs.shape)
