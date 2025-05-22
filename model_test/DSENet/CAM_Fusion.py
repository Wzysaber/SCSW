import torch
import torch.nn as nn
import torch.nn.functional as F


class CAM_ALL_Fusion(nn.Module):
    def __init__(self, in_features=[64, 128, 320, 512]):
        super().__init__()
        self.Constage1_1x1 = conv1x1(in_features[0], in_features[0])
        self.Constage2_1x1 = conv1x1(in_features[1], in_features[0])
        self.Constage3_1x1 = conv1x1(in_features[2], in_features[0])
        self.Constage4_1x1 = conv1x1(in_features[3], in_features[0])

        self.Con1x1 = conv1x1(in_features[0] * 4, in_features[0])

    def forward(self, input_CAM):
        x1, x2, x3, x4 = input_CAM
        w, h = x1.shape[2], x1.shape[3]

        x1 = self.Constage1_1x1(x1)

        x2 = self.Constage2_1x1(x2)
        x2 = F.interpolate(x2, size=(w, h), mode='bilinear', align_corners=False)

        x3 = self.Constage3_1x1(x3)
        x3 = F.interpolate(x3, size=(w, h), mode='bilinear', align_corners=False)

        x4 = self.Constage4_1x1(x4)
        x4 = F.interpolate(x4, size=(w, h), mode='bilinear', align_corners=False)

        Fusion_CAM = torch.cat([x1, x2, x3, x4], dim=1)

        CAM = self.Con1x1(Fusion_CAM)

        return CAM


class CAM_Fusion(nn.Module):
    def __init__(self, in_features=[256, 256]):
        super().__init__()
        self.ConMid_1x1 = conv1x1(in_features[0], in_features[0])
        self.ConFinal_1x1 = conv1x1(in_features[1], in_features[1])

        self.Con1x1 = conv1x1(in_features[1] + in_features[0], in_features[1])

    def forward(self, mid_CAM, final_CAM):
        mid_CAM = self.ConMid_1x1(mid_CAM)

        final_CAM = self.ConFinal_1x1(final_CAM)
        final_CAM = F.interpolate(final_CAM, size=(mid_CAM.shape[2], mid_CAM.shape[3]), mode='bilinear',
                                  align_corners=False)

        CAM = torch.cat([mid_CAM, final_CAM], dim=1)
        CAM = self.Con1x1(CAM)

        return CAM


class conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out
