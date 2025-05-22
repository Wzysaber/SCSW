import torch
import torch.nn as nn
import torch.nn.functional as F


from model_test.resnet50.resnet50 import resnet50

class model_test1(nn.Module):

    def __init__(self):
        super(model_test1, self).__init__()

        self.resnet50 = resnet50(pretrained=True, strides=(2, 2, 2, 1))
        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)

        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 5, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):
        N, C, H, W = x.size()

        # # branch1

        x = self.stage1(x).detach()
        x = self.stage2(x).detach()
        x = self.stage3(x).detach()
        x = self.stage4(x).detach()

        cam = self.classifier(x)
        # cam = F.interpolate(cam, (H, W), mode='bilinear', align_corners=True)


        return cam

    def get_param_groups(self):
        return (
            list(self.backbone.parameters()),
            list(self.newly_added.parameters())
        )


# class CAM(nn.Module):
#
#     def __init__(self):
#         super(CAM, self).__init__()
#
#     def forward(self, x, step=1):
#
#         x_ori = x.clone()
#
#         # branch1
#         if step == 1:
#             x = self.stage1(x)
#             x = self.stage2(x)
#             x = self.stage3(x)
#             x = self.stage4(x)
#
#             cam1 = F.conv2d(x, self.classifier.weight)
#             return cam1
#
#         # # branch2
#         if step == 2:
#             x2 = self.stage2_1(x_ori)
#             x2 = self.stage2_2(x2)
#             x2 = self.stage2_3(x2)
#             x2 = self.stage2_4(x2)
#
#             cam2 = F.conv2d(x2, self.classifier2.weight)
#             return cam2


if __name__ == '__main__':
    model = model_test1()
    model.eval()
    image = torch.randn(2, 3, 256, 256)

    output = model(image)
    print("input:", image.shape)
    print("output:", output.shape)
