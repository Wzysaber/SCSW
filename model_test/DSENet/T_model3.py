"""
(1) MiT-B1+ aux +pre+ 强弱监督（双loss）+ 联合Losser（grid=40）

(2) aux_loss改进
    1） 进行中间阶段融合（stage1 + stage2 +stage3）

(3) 特征基本融合
    1）mid + final
    2）ALL_Map_fusion

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import flop_count_table
from fvcore.nn.flop_count import FlopCountAnalysis

from model_test.DSENet.Decoder import SegFormerHead, Single_SegFormerHead
from model_test.DSENet import mix_transformer
from model_test.DSENet.CAM_Fusion import CAM_ALL_Fusion
from model_test.DSENet.CAM_Fusion import conv1x1

import numpy as np


# class Aux_x(nn.Module):
#     def __init__(self, in_features=[64, 128, 320]):
#         super().__init__()
#         self.attn1_proj = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, bias=True)
#         self.attn2_proj = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, bias=True)
#         self.attn3_proj = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, bias=True)
#         self.attn4_proj = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, bias=True)
#
#     def forward(self, attn_list):
#         for i in range(len(attn_list) / 2):
#             attn_cat = torch.mean(torch.stack(attn_list[i * 2], torch.stack(attn_list[i * 2 + 1]), dim=0)
#             attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)
#
#             attn_pred = self.attn_proj(attn_cat)
#             attn_pred4 = torch.sigmoid(attn_pred4)[:, 0, ...]
#
#         return attn_pred


class Aux_x(nn.Module):
    def __init__(self, in_features=[64, 128, 320]):
        super().__init__()
        self.Con1x1_stage1 = conv1x1(in_features[0], in_features[1])
        self.Con1x1_stage2 = conv1x1(in_features[1], in_features[1])
        self.Con1x1_stage3 = conv1x1(in_features[2], in_features[1])

        self.Con1x1 = conv1x1(in_features[1] * 3, in_features[1])

    def forward(self, x1, x2, x3):
        x1 = self.Con1x1_stage1(x1)
        x1 = F.interpolate(x1, size=(x2.shape[2], x2.shape[3]), mode='bilinear',
                           align_corners=False)

        x2 = self.Con1x1_stage2(x2)

        x3 = self.Con1x1_stage3(x3)
        x3 = F.interpolate(x3, size=(x2.shape[2], x2.shape[3]), mode='bilinear',
                           align_corners=False)

        aux_x = torch.cat([x1, x2, x3], dim=1)
        aux_x = self.Con1x1(aux_x)

        return aux_x


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
                "/home0/students/master/2022/wangzy/Pycharm-Remote(161)/WSSS_Model2/pretrained/checkpoints/mit_b1.pth")
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

        self.Aux_x = Aux_x(in_features=[self.in_channels[0], self.in_channels[1], self.in_channels[2]])

        self.StageAttn_projs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, bias=True),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, bias=True),
            nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, bias=True),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, bias=True)
        ])

        for proj in self.StageAttn_projs:
            nn.init.kaiming_normal_(proj.weight, a=np.sqrt(5), mode="fan_out")

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes - 1, kernel_size=1,
                                    bias=False)
        self.aux_classifier = nn.Conv2d(in_channels=self.in_channels[1], out_channels=self.num_classes - 1,
                                        kernel_size=1,
                                        bias=False)

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.aux_classifier.weight)

        for proj in self.StageAttn_projs:
            param_groups[2].append(proj.weight)
            param_groups[2].append(proj.bias)

        for name, param in self.Aux_x.named_parameters():
            param_groups[2].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    def forward(self, x, cam_only=False, aux=False):

        _x, _attns = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x

        mid_x = self.Aux_x(_x1, _x2, _x3)
        final_x = _x4
        segs = self.decoder(_x)

        attns_groups = [_attns[i:i + 2] for i in range(0, len(_attns), 2)]
        attn_preds = []
        for i in range(4):
            attn_cat = torch.mean(torch.stack(attns_groups[i], dim=0), dim=0)
            # attn_cat = torch.cat(attns_groups[i], dim=1)
            attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)
            # 使用对应的 StageAttn_projs 卷积层
            attn_pred = self.StageAttn_projs[i](attn_cat)
            attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]

            attn_preds.append(attn_pred)

        if cam_only:
            cam = F.conv2d(final_x, self.classifier.weight).detach()
            return cam

        # _x4 = self.dropout(_x4.clone()
        final_x = self.pooling(final_x, (1, 1))
        final_x = self.classifier(final_x)
        final_x = final_x.view(-1, self.num_classes - 1)

        aux_cls = self.pooling(mid_x, (1, 1))
        aux_cls = self.aux_classifier(aux_cls)
        aux_cls = aux_cls.view(-1, self.num_classes - 1)

        if aux:
            return cls_x4, segs, _attns

        return final_x, aux_cls, segs, attn_preds


if __name__ == "__main__":
    model = TSCD('mit_b1', stride=[4, 2, 2, 1], pooling="gmp", num_classes=6, embedding_dim=256, pretrained=False)

    dummy_input = torch.rand(4, 3, 256, 256)

    cls_x4, aux_cls, segs, attn_pred = model(dummy_input, cam_only=False)

    # total_params = sum([np.prod(p.size()) for p in model.parameters()])
    # print("Total network parameters (excluding idr):%.2fM" % (total_params / 1e6))
    # flops = FlopCountAnalysis(model, dummy_input)
    # print(flop_count_table(flops))

    print("output:", cls_x4.shape)
    print("output:", aux_cls.shape)
    print("output:", segs.shape)
