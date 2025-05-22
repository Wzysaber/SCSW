import pdb
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
import torch.distributed as dist

from bilateralfilter import bilateralfilter, bilateralfilter_batch


def get_aff_loss(inputs, targets):
    pos_label = (targets == 1).type(torch.int16)
    pos_count = pos_label.sum() + 1
    neg_label = (targets == 0).type(torch.int16)
    neg_count = neg_label.sum() + 1
    # inputs = torch.sigmoid(input=inputs)

    pos_loss = torch.sum(pos_label * (1 - inputs)) / pos_count
    neg_loss = torch.sum(neg_label * (inputs)) / neg_count

    return 0.5 * pos_loss + 0.5 * neg_loss


def get_masked_ptc_loss(inputs, mask):
    b, c, h, w = inputs.shape

    inputs = inputs.reshape(b, c, h * w)

    def cos_sim(x):
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        cos_sim = torch.matmul(x.transpose(1, 2), x)
        return torch.abs(cos_sim)

    inputs_cos = cos_sim(inputs)

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


def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)


class ContrastiveCorrelationLoss(nn.Module):

    def __init__(self):
        super(ContrastiveCorrelationLoss, self).__init__()
        self.inter_cal = None
        self.intra_cal = None
        self.neg_cal = None

        self.feature_samples = 40

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2, c1, c2):
        with torch.no_grad():
            fd = tensor_correlation(norm(f1), norm(f2))

            old_mean = fd.mean()
            fd -= fd.mean([3, 4], keepdim=True)
            fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))

        loss = - cd.clamp(0) * (fd)

        return loss, cd

    def forward(self,
                orig_feats: torch.Tensor,
                orig_feats_pos: torch.Tensor,
                orig_code: torch.Tensor,
                orig_code_pos: torch.Tensor,
                ):
        coord_shape = [orig_feats.shape[0], self.feature_samples, self.feature_samples, 2]
        coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
        coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = sample(orig_feats, coords1)
        code = sample(orig_code, coords1)

        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)

        pos_intra_loss, pos_intra_cd = self.helper(feats, feats_pos, code, code_pos)

        return pos_intra_loss.mean()


def NT_Xent0(out_1, out_2, temperature):
    batch_size = out_1.shape[0]

    # 将特征图展平为 [B, C, W*H]
    out_1 = out_1.view(batch_size, -1)  # [B, N]
    out_2 = out_2.view(batch_size, -1)  # [B, N]

    out_1 = F.normalize(out_1, p=2, dim=1, eps=1e-8)  # [B, N]
    out_2 = F.normalize(out_2, p=2, dim=1, eps=1e-8)

    # 样本级对齐
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def NT_Xent_pixel_level(out_1, out_2, temperature=0.07):
    """
    NT-Xent Loss for pixel-level comparison in segmentation tasks.

    Args:
        out_1: Tensor of shape [B, C, W, H] - Feature map for the first output.
        out_2: Tensor of shape [B, C, W, H] - Feature map for the second output.
        temperature: Temperature scaling factor.

    Returns:
        loss: Computed NT-Xent loss for pixel-level comparison.
    """
    B, C, W, H = out_1.shape

    # 展平特征图，得到形状为 [B * W * H, C] 的张量
    out_1 = out_1.view(B, C, -1).permute(0, 2, 1)  # [B, W*H, C]
    out_2 = out_2.view(B, C, -1).permute(0, 2, 1)  # [B, W*H, C]

    # # L2归一化每个像素的特征向量
    out_1 = F.normalize(out_1, p=2, dim=-1)  # [B, W*H, C]
    out_2 = F.normalize(out_2, p=2, dim=-1)  # [B, W*H, C]

    # 计算相似度矩阵 [B, W*H, W*H]
    sim_matrix = torch.matmul(out_1, out_2.transpose(1, 2))  # [B, W*H, W*H]
    sim_matrix = torch.exp(sim_matrix / temperature)  # 温度缩放

    # 生成掩码，防止将自己与自己比较
    mask = torch.eye(W * H, device=sim_matrix.device).bool()
    sim_matrix.masked_fill_(mask, 0)  # 将对角线元素置为0，避免自比较

    # 计算正样本的相似度
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)  # [B, W*H]

    # 计算损失：正样本对的相似度与所有其他样本的相似度的比值
    loss = -torch.log(pos_sim / sim_matrix.sum(dim=-1))  # [B, W*H]
    loss = loss.mean()  # 对所有像素取均值

    return loss


def Cut_attn(attn_1, attn_2):
    stage_attn = []
    for i in range(4):
        stage_attn.extend([attn_1[i * 2], attn_1[i * 2 + 1], attn_2[i * 2], attn_2[i * 2 + 1]])

    stage_attn_groups = [stage_attn[i:i + 4] for i in range(0, len(stage_attn), 4)]

    return stage_attn_groups


def attn_aff(attn_1, attn_2):
    stage_attn_groups = Cut_attn(attn_1, attn_2)
    attn_pred_list = []
    for i in range(4):
        attn_cat = torch.mean(torch.stack(stage_attn_groups[i]), dim=0)
        attn_cat = F.interpolate(attn_cat, size=(320, 320), mode='bilinear',
                           align_corners=False)
        attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)

        attn_pred = torch.mean(attn_cat, dim=1, keepdim=True)
        attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]
        attn_pred_list.append(attn_pred)

    return attn_pred_list

