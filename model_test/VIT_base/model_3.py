import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_test.VIT_base import backbone as encoder
from model import decoder


class network(nn.Module):
    def __init__(self, backbone, num_classes=None, pretrained=None, init_momentum=None, aux_layer=-3):
        super().__init__()
        self.num_classes = num_classes
        self.init_momentum = init_momentum

        self.encoder = getattr(encoder, backbone)(pretrained=pretrained, aux_layer=aux_layer)

        self.in_channels = [self.encoder.embed_dim] * 4 \
            if hasattr(self.encoder, "embed_dim") else [self.encoder.embed_dims[-1]] * 4

        self.pooling = F.adaptive_max_pool2d

        self.decoder = decoder.LargeFOV(in_planes=self.in_channels[-1], out_planes=self.num_classes, )

        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes - 1,
                                    kernel_size=1, bias=False, )

        self.aux_classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes - 1,
                                        kernel_size=1, bias=False, )

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.aux_classifier.weight)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    def to_2D(self, x, h, w):
        n, hw, c = x.shape
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward(self, x, cam_only=False):

        cls_token, x_patch, all_embeds, attn_weights = self.encoder.forward_features(x)
        xout = all_embeds[0][:, 1:]

        x_aux_patch = all_embeds[-3][:, 1:]

        for i in [3, 7, 11]:
            xout += all_embeds[i][:, 1:]

        h, w = x.shape[-2] // self.encoder.patch_size, x.shape[-1] // self.encoder.patch_size

        _x4 = self.to_2D(xout, h, w)
        _x_aux = self.to_2D(x_aux_patch, h, w)

        if cam_only:
            cam = F.conv2d(_x4, self.classifier.weight).detach()
            aux_cam = F.conv2d(_x_aux, self.aux_classifier.weight).detach()

            return cam, aux_cam

        segs = self.decoder(_x4)

        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)

        aux_cls_x4 = self.pooling(_x_aux, (1, 1))
        aux_cls_x4 = self.aux_classifier(aux_cls_x4)

        cls_logit = cls_x4.view(-1, self.num_classes - 1)
        cls_aux_logit = aux_cls_x4.view(-1, self.num_classes - 1)

        return cls_logit, cls_aux_logit, segs, _x4


if __name__ == '__main__':
    model = network(
        backbone="deit_base_patch16_224",
        pretrained=False,
        num_classes=16,
        init_momentum=0.9,
    )

    print(model)
    model.eval()
    image = torch.randn(5, 3, 256, 256)

    cls_x4, seg, _x4, x_aux = model(image, cam_only=False)
    print("input:", image.shape)
    print("output:", cls_x4.shape)
    print("output:", seg.shape)
