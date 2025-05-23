import pdb
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
import torch.distributed as dist

sys.path.append("./wrapper/bilateralfilter/build/lib.linux-x86_64-3.8")
from bilateralfilter import bilateralfilter, bilateralfilter_batch


def get_aff_loss(inputs, targets):
    pos_label = (targets == 1).type(torch.int16)
    pos_count = pos_label.sum() + 1
    neg_label = (targets == 0).type(torch.int16)
    neg_count = neg_label.sum() + 1
    # inputs = torch.sigmoid(input=inputs)

    pos_loss = torch.sum(pos_label * (1 - inputs)) / pos_count
    neg_loss = torch.sum(neg_label * (inputs)) / neg_count

    return 0.5 * pos_loss + 0.5 * neg_loss, pos_count, neg_count


def get_masked_ptc_loss(inputs, mask):
    b, c, h, w = inputs.shape

    inputs = inputs.reshape(b, c, h * w)

    def cos_sim(x):
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        cos_sim = torch.matmul(x.transpose(1, 2), x)
        return torch.abs(cos_sim)

    inputs_cos = cos_sim(inputs)

    # 所以本质上是将Fmap（进行余玹对成矩阵和中间层的进行Aff相似性对称矩阵进行loss）
    pos_mask = mask == 1
    neg_mask = mask == 0
    loss = 0.5 * (1 - torch.sum(pos_mask * inputs_cos) / (pos_mask.sum() + 1)) + 0.5 * torch.sum(
        neg_mask * inputs_cos) / (neg_mask.sum() + 1)
    return loss


def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label == 0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5


def get_energy_loss(device, img, logit, label, img_box, loss_layer, mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375]):
    pred_prob = F.softmax(logit, dim=1)
    crop_mask = torch.zeros_like(pred_prob[:, 0, ...])

    for idx, coord in enumerate(img_box):
        crop_mask[idx, coord[0]:coord[1], coord[2]:coord[3]] = 1

    # 图像进行归一化处理
    _img = torch.zeros_like(img)
    _img[:, 0, :, :] = img[:, 0, :, :] * std[0] + mean[0]
    _img[:, 1, :, :] = img[:, 1, :, :] * std[1] + mean[1]
    _img[:, 2, :, :] = img[:, 2, :, :] * std[2] + mean[2]

    loss = loss_layer(_img, pred_prob, crop_mask, label.type(torch.uint8).unsqueeze(1), )

    return loss.to(device)


class CTCLoss_neg(nn.Module):
    def __init__(self, ncrops=10, temp=1.0, ):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum
        self.ncrops = ncrops
        # self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        # self.teacher_temp_schedule = np.concatenate((
        #     np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
        #     np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        # ))

    def forward(self, student_output, teacher_output, flags):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        b = flags.shape[0]

        student_out = student_output.reshape(self.ncrops, b, -1).permute(1, 0, 2)
        teacher_out = teacher_output.reshape(2, b, -1).permute(1, 0, 2)

        logits = torch.matmul(teacher_out, student_out.permute(0, 2, 1))
        logits = torch.exp(logits / self.temp)

        total_loss = 0
        for i in range(b):
            neg_logits = logits[i, :, flags[i] == 0]
            pos_inds = torch.nonzero(flags[i])[:, 0]
            loss = 0

            for j in pos_inds:
                pos_logit = logits[i, :, j]
                loss += -torch.log((pos_logit) / (pos_logit + neg_logits.sum(dim=1) + 1e-4))
            else:
                loss += -torch.log((1) / (1 + neg_logits.sum(dim=1) + 1e-4))

            total_loss += loss.sum() / 2 / (pos_inds.shape[0] + 1e-4)

        total_loss = total_loss / b

        return total_loss


class DenseEnergyLossFunction(Function):

    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs, unlabel_region):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        Gate = ROIs.clone().to(ROIs.device)

        ROIs = ROIs.unsqueeze_(1).repeat(1, ctx.K, 1, 1)

        seg_max = torch.max(segmentations, dim=1)[0]
        Gate = Gate - seg_max
        Gate[unlabel_region] = 1
        Gate[Gate < 0] = 0
        Gate = Gate.unsqueeze_(1).repeat(1, ctx.K, 1, 1)

        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs

        densecrf_loss = 0.0
        images = images.cpu().numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        Gate = Gate.cpu().numpy().flatten()
        AS = np.multiply(AS, Gate)
        densecrf_loss -= np.dot(segmentations, AS)

        # averaged by the number of images
        densecrf_loss /= ctx.N

        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2 * grad_output * torch.from_numpy(ctx.AS) / ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None, None


class DenseEnergyLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseEnergyLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor

    def forward(self, images, segmentations, ROIs, seg_label):
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images, scale_factor=self.scale_factor, recompute_scale_factor=True)
        scaled_segs = F.interpolate(segmentations, scale_factor=self.scale_factor, mode='bilinear', align_corners=False,
                                    recompute_scale_factor=True)

        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1), scale_factor=self.scale_factor,
                                    recompute_scale_factor=True).squeeze(1)
        scaled_seg_label = F.interpolate(seg_label, scale_factor=self.scale_factor, mode='nearest',
                                         recompute_scale_factor=True)

        unlabel_region = (scaled_seg_label.long() == 255).squeeze(1)

        return self.weight * DenseEnergyLossFunction.apply(
            scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy * self.scale_factor, scaled_ROIs, unlabel_region)

    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )


class JointLoss(nn.Module):
    def __init__(self, ignore_index=255, ratio=0.2):
        super(JointLoss, self).__init__()

        self.ignore_index = ignore_index
        self.ratio = ratio

    def forward(self, cls_pred, binary_pred, cls_true):
        valid_mask = (cls_true != self.ignore_index)
        fgp = torch.sigmoid(binary_pred)
        clsp = torch.softmax(cls_pred, dim=1)
        joint_prob = torch.clone(clsp)
        joint_prob[:, 0, :, :] = fgp[:, 0, :, :] * clsp[:, 0, :, :]
        joint_prob[:, 1:, :, :] = fgp[:, 1:, :, :] * clsp[:, 1:, :, :]
        Z = torch.sum(joint_prob, dim=1, keepdim=True)
        p_ci = joint_prob / Z
        p_ci = torch.clamp(p_ci, min=1e-7, max=1 - 1e-7)
        losses = F.nll_loss(torch.log(p_ci), cls_true.long(), ignore_index=self.ignore_index, reduction='none')
        return losses.sum() / valid_mask.sum()
