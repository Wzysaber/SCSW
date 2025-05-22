import argparse
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]= '2'
sys.path.append(".")
from collections import OrderedDict

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from model.model_seg_neg_fp import network
# from omegaconf import OmegaConf
from datasets import iSAID
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from utils.camutils_ori import cam_to_label, get_valid_cam, multi_scale_cam2, multi_scale_cam1
from utils.pyutils import AverageMeter, format_tabs
from model_test.DSENet.T_model3 import TSCD

#"/home0/students/master/2022/wangzy/Pycharm-Remote(161)/WSSS_Model2/weight/ISAID_dataset/2025-02-19-21-44-15-556198/checkpoints/Best mIoU: 38.9269609646926, model: AWTS model_iter_18000.pth"

parser = argparse.ArgumentParser()
parser.add_argument("--bkg_thre", default=0.5, type=float, help="work_dir")

parser.add_argument("--model_path", default="/home0/students/master/2022/wangzy/Pycharm-Remote(161)/WSSS_Model2/weight/ISAID_dataset/2025-02-14-09-25-58-793942/checkpoints/Best mIoU: 40.278214500033584, model: AWTS model_iter_16000.pth", type=str, help="model_path")

parser.add_argument("--backbone", default='deit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")

parser.add_argument("--data_folder", default="/home0/students/master/2022/wangzy/datasets/iSAID_WSSS/sampled_process", type=str, help="dataset folder")
parser.add_argument("--list_folder", default="/home0/students/master/2022/wangzy/Pycharm-Remote(161)/CTFA/datasets/iSAID", type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=16, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--infer_set", default="test", type=str, help="infer_set")


def _validate(model=None, data_loader=None, args=None):
    model.eval()

    base_dir = args.model_path.split("checkpoint")[0]
    cam_dir = os.path.join(base_dir, "cam_img", args.infer_set)
    cam_aux_dir = os.path.join(base_dir, "cam_img_aux", args.infer_set)

    os.makedirs(cam_aux_dir, exist_ok=True)
    os.makedirs(cam_dir, exist_ok=True)
    color_map = plt.get_cmap("jet")

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda()

        gts, cams, aux_cams = [], [], []

        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            img = imutils.denormalize_img(inputs)[0].permute(1, 2, 0).cpu().numpy()

            inputs = F.interpolate(inputs, size=[448, 448], mode='bilinear', align_corners=False)
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            ###
            _cams = multi_scale_cam1(model, inputs=inputs, scales=[1.0, 0.5, 1.5])
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)

            cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre)
            resized_cam = get_valid_cam(resized_cam, cls_label)

            cam_np = torch.max(resized_cam[0], dim=0)[0].cpu().numpy()
            cam_rgb = color_map(cam_np)[:, :, :3] * 255

            alpha = 0.6
            cam_rgb = alpha * cam_rgb + (1 - alpha) * img

            imageio.imsave(os.path.join(cam_dir, name[0] + ".jpg"), cam_rgb.astype(np.uint8))

            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

    cam_score = evaluate.scores(gts, cams)

    return format_tabs([cam_score], ["cam"], cat_list=iSAID.class_list)


def validate(args=None):
    val_dataset = iSAID.iSAIDSegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.infer_set,
        stage='test',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=False,
                            drop_last=False)


    model = TSCD('mit_b1', stride=[4, 2, 2, 1], pooling="gmp", num_classes=16, embedding_dim=256, pretrained=True)

    trained_state_dict = torch.load(args.model_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v

    print(model.load_state_dict(state_dict=new_state_dict, strict=False))
    model.eval()

    results = _validate(model=model, data_loader=val_loader, args=args)
    torch.cuda.empty_cache()

    print(results)

    return True


if __name__ == "__main__":
    args = parser.parse_args()

    validate(args=args)

