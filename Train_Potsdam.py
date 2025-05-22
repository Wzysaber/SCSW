import argparse
import datetime
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import iSAID
from datasets import potsdam

from utils.losses import get_masked_ptc_loss, get_seg_loss, CTCLoss_neg, DenseEnergyLoss, get_energy_loss, JointLoss, \
    ContrastiveCorrelationLoss, NT_Xent0, get_aff_loss, attn_aff

from torch import autograd
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.PAR import PAR
from utils import evaluate, imutils, optimizer
from thop import profile

# from utils.camutils import (cam_to_label, cams_to_affinity_label, ignore_img_box,
#                             multi_scale_cam, multi_scale_cam_with_aff_mat,
#                             propagte_aff_cam_with_bkg, refine_cams_with_bkg_v2,
#                             refine_cams_with_cls_label)
from utils.camutils_ori import cam_to_label, cam_to_roi_mask2, multi_scale_cam2, multi_scale_cam1, cam_to_label2, \
    label_to_aff_mask, refine_cams_with_bkg_v2, crop_from_roi_neg, cams_to_affinity_label
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger

from model_test.VIT_base.vit_b import network

# from model_test.DSENet.T_model import TSCD
# from model_test.DSENet.T_model_base import TSCD

from model_test.DSENet.T_model3 import TSCD  # 有background

#
# from model_test.DSENet.T_model4 import TSCD  # 无background

# from model_test.VIT_base.model_2 import network

# from model_test.VIT_base.model_3 import network

corr_loss = ContrastiveCorrelationLoss()

parser = argparse.ArgumentParser()

parser.add_argument("--GPU_num", default=9, type=int, help="number of GPU")
parser.add_argument("--information",
                    default='ALL (0.5, 1.0, 1.5, 2)lr=2e-5 warmup_lr=1e-6 (0.7 0.5) 有背景cam(0.6 0.4) ',
                    type=str,
                    help="information")
parser.add_argument("--w_ptc", default=0.3, type=float, help="w_ptc")
parser.add_argument("--w_seg", default=0.1, type=float, help="w_seg")
parser.add_argument("--w_reg", default=0.05, type=float, help="w_reg")

parser.add_argument("--backbone", default='deit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")

parser.add_argument("--data_folder",
                    default="/home0/students/master/2022/wangzy/datasets/Potsdam_WSSS/potsdam_IRRG_wiB_512_256_dl/",
                    type=str, help="dataset folder")
parser.add_argument("--list_folder",
                    default="/home0/students/master/2022/wangzy/Pycharm-Remote(161)/WSSS_Model2/datasets/potsdam/",
                    type=str,
                    help="train/val/test list file")

parser.add_argument("--num_classes", default=6, type=int, help="number of classes")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size in training")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

parser.add_argument("--work_dir",
                    default="/home0/students/master/2022/wangzy/Pycharm-Remote(161)/WSSS_Model2/weight/Potsdam_dataset/",
                    type=str, help="ISAID_Weight")

parser.add_argument("--train_set", default="train", type=str, help="training split")
parser.add_argument("--val_set", default="val", type=str, help="validation split")
parser.add_argument("--test_set", default="test", type=str, help="testing split")
parser.add_argument("--spg", default=8, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=2e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")

parser.add_argument("--max_iters", default=20000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")

parser.add_argument("--high_thre", default=0.7, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="bkg_score")
parser.add_argument("--cam_scales", default=(0.5, 1.0, 1.5, 2), help="multi_scales for cam")

parser.add_argument("--temp", default=0.5, type=float, help="temp")
parser.add_argument("--momentum", default=0.9, type=float, help="temp")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--save_ckpt", default="1", action="store_true", help="save_ckpt")

parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
parser.add_argument("--num_workers", default=10, type=int, help="num_workers")


def get_mask_by_radius(h=10, w=10, radius=8):
    hw = h * w
    # _hw = (h + max(dilations)) * (w + max(dilations))
    mask = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius + 1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius + 1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask


def setting_log(args):
    # 创建初始log
    setup_logger(filename=os.path.join(args.work_dir, 'train.log'))

    # 初始信息print
    # logging.info('\nargs: %s' % args)
    logging.info("GPU num: %s" % args.GPU_num)
    logging.info(args.information)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate(device, model=None, data_loader=None, args=None):
    preds, gts, cams, cams_aux = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()

    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            cls_label = cls_label.to(device)

            inputs = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)

            cls_x4, aux_cls, segs, fmap = model(inputs)

            cls_pred = (cls_x4 > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            _cams = multi_scale_cam1(model, inputs=inputs, scales=args.cam_scales)

            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                                     low_thre=args.low_thre, ignore_index=args.ignore_index)

            cls_pred = (cls_x4 > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores2(gts, preds, 6)
    cam_score = evaluate.scores2(gts, cams, 6)

    model.train()

    tab_results, mIoU = format_tabs([cam_score, seg_score], name_list=["CAM", "Seg_Pred"],
                                    cat_list=potsdam.class_list)
    OA, Average_F1 = seg_score['OA'] * 100, seg_score['F1'] * 100

    return cls_score, tab_results, mIoU, OA, Average_F1


def train(args=None):
    torch.cuda.set_device(args.GPU_num)
    time0 = datetime.datetime.now()

    mIoU = 0.0001
    cam_mIoU = 0.0001

    train_dataset = potsdam.potsdamClsDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.train_set,
        stage='train',
        aug=True,
        # resize_range=cfg.dataset.resize_range,
        rescale_range=args.scales,
        crop_size=args.crop_size,
        img_fliplr=True,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    val_dataset = potsdam.potsdamSegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.test_set,
        stage='test',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        # shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)

    device = torch.device("cuda:%d" % (args.GPU_num))

    model = TSCD('mit_b1', stride=[4, 2, 2, 1], pooling="gmp", num_classes=6, embedding_dim=256, pretrained=True)

    mask_size1 = int(320 // 32)
    mask_size2 = int(320 // 16)

    attn_mask_path1 = get_mask_by_radius(h=mask_size1, w=mask_size1, radius=8)
    attn_mask_path2 = get_mask_by_radius(h=mask_size2, w=mask_size2, radius=8)

    param_groups = model.get_param_groups()

    optim = getattr(optimizer, args.optimizer)(
        params=[
            {
                "params": param_groups[0],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[1],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[2],
                "lr": args.lr * 8,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[3],
                "lr": args.lr * 8,
                "weight_decay": args.wt_decay,
            },
        ],
        lr=args.lr,
        weight_decay=args.wt_decay,
        betas=args.betas,
        warmup_iter=args.warmup_iters,
        max_iter=args.max_iters,
        warmup_ratio=args.warmup_lr,
        power=args.power)

    logging.info('\nOptimizer: \n%s' % optim)

    model.to(device)
    model.train()

    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()

    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
    par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).to(device)
    since = time.time()

    for n_iter in range(args.max_iters):

        try:
            img_name, inputs, cls_label, img_box, crops = next(train_loader_iter)

        except:
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_label, img_box, crops = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        crops = crops.to(device, non_blocking=True)

        # 图像归一化操作
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_label = cls_label.to(device, non_blocking=True)

        cams1 = multi_scale_cam1(model, inputs=inputs, scales=args.cam_scales)
        cams2 = multi_scale_cam1(model, inputs=crops, scales=args.cam_scales)

        cls_logit1, cls_aux_logit1, segs1, attn1 = model(inputs, )
        cls_logit2, cls_aux_logit2, segs2, attn2 = model(crops, )

        loss_er = torch.mean(torch.abs(cams1 - cams2))

        # 构建多组各类loss的构建
        cls_loss1 = F.multilabel_soft_margin_loss(cls_logit1, cls_label)
        aux_cls_loss1 = F.multilabel_soft_margin_loss(cls_aux_logit1, cls_label)

        cls_loss2 = F.multilabel_soft_margin_loss(cls_logit2, cls_label)
        aux_cls_loss2 = F.multilabel_soft_margin_loss(cls_aux_logit2, cls_label)

        cls_loss = cls_loss1 + cls_loss2
        aux_cls_loss = aux_cls_loss1 + aux_cls_loss2

        loss_corr = corr_loss(
            cams1, cams2,
            segs1, segs2,
        )

        # seg_loss & reg_loss
        valid_cam, pseudo_label = cam_to_label(0.6 * cams1.detach() + 0.4 * cams2.detach(), cls_label=cls_label,
                                               img_box=img_box,
                                               ignore_mid=True,
                                               bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                                               low_thre=args.low_thre,
                                               ignore_index=args.ignore_index)

        refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,
                                                       high_thre=args.high_thre, low_thre=args.low_thre,
                                                       ignore_index=args.ignore_index, img_box=img_box,
                                                       background=True)

        segs = F.interpolate(segs1, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        seg_loss = get_seg_loss(segs, refined_pseudo_label.type(torch.long), ignore_index=args.ignore_index)

        reg_loss = get_energy_loss(device, img=inputs, logit=segs, label=refined_pseudo_label, img_box=img_box,
                                   loss_layer=loss_layer)

        aff_label1 = cams_to_affinity_label(refined_pseudo_label, cut=32, mask=attn_mask_path1, ignore_index=255)
        aff_label2 = cams_to_affinity_label(refined_pseudo_label, cut=16, mask=attn_mask_path2, ignore_index=255)

        aux_losses_stage_mid = [get_aff_loss(attn, aff_label1) for attn in attn1[:3]]
        aux_losses_stage_final = get_aff_loss(attn1[3], aff_label2)

        aux_loss = (sum(aux_losses_stage_mid) + aux_losses_stage_final) / 4

        # warmup
        if n_iter <= 2000:
            loss = 1.0 * cls_loss + 1.0 * aux_cls_loss + 0.0 * seg_loss + 0.0 * reg_loss
        else:
            loss = 1.0 * cls_loss + 1.0 * aux_cls_loss + args.w_seg * seg_loss + args.w_reg * reg_loss + 0.1 * loss_er \
                   + 0.1 * loss_corr + 0.3 * aux_loss

        cls_pred = (cls_logit1 > 0).type(torch.int16)
        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])

        avg_meter.add({
            'cls_loss': cls_loss.item(),
            'seg_loss': seg_loss.item(),
            'cls_score': cls_score.item(),

        })

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if (n_iter + 1) % args.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
            cur_lr = optim.param_groups[0]['lr']

            if args.local_rank == 0:
                logging.info(
                    "Iter: %d; Elasped: %s; ETA: %s; lr: %.7f; cls_loss: %.4f,seg_loss: %.4f..." % (
                        n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'),
                        avg_meter.pop('seg_loss')
                    ))

        if (n_iter + 1) % args.eval_iters == 0:

            if args.local_rank == 0:
                logging.info('Validating...')

            val_cls_score, tab_results, mIoU_result, OA, Average_F1 = validate(device, model=model,
                                                                               data_loader=val_loader, args=args)

            if args.save_ckpt and (n_iter + 1) >= 6000 and mIoU_result[1] > mIoU:
                mIoU = mIoU_result[1]
                cam_mIoU = mIoU_result[0]

                ckpt_name = os.path.join(args.ckpt_dir,
                                         "Best mIoU: {}, model: {} model_iter_%d.pth".format(mIoU, "AWTS") % (
                                                 n_iter + 1))
                torch.save(model.state_dict(), ckpt_name)

            if args.local_rank == 0:
                logging.info("val cls score: %.6f" % (val_cls_score))
                logging.info("\n" + tab_results)
                logging.info("Segmentation Results:")
                logging.info(f"OA: {OA:.4f}, MioU: {mIoU_result[1]:.4f}, F1: {Average_F1:.4f}")
                print(" ")

    time_elapsed = time.time() - since
    print('Time in  {:.0f}h:{:.0f}min:{:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))
    print(" ")
    print("best cam_mIoU = %.4f  best seg_mIoU = %.4f" % (cam_mIoU, mIoU))

    return True


if __name__ == "__main__":
    torch.cuda.empty_cache()

    args = parser.parse_args()
    timestamp = "{0:%Y-%m-%d-%H-%M-%S-%f}".format(datetime.datetime.now())
    args.work_dir = os.path.join(args.work_dir, timestamp)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")

    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

    setting_log(args)
    setup_seed(args.seed)
    train(args=args)
