import argparse
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from collections import OrderedDict
import imageio
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import potsdam
from model.model_seg_neg import network
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from utils.dcrf import DenseCRF
from utils.pyutils import format_tabs

from model_test.Ablation.model1 import model1

parser = argparse.ArgumentParser()
parser.add_argument("--infer_set", default="test", type=str, help="infer_set")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--model_path",
                    default="/home0/students/master/2022/wangzy/Pycharm-Remote(161)/WSSS_Model2/pretrained/CTFA_weight/checkpoints/Potsdam_model_iter_20000.pth",
                    type=str, help="model_path")

parser.add_argument("--backbone", default='deit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--data_folder",
                    default="/home0/students/master/2022/wangzy/datasets/Potsdam_WSSS/potsdam_IRRG_wiB_512_256_dl/",
                    type=str, help="dataset folder")
parser.add_argument("--list_folder",
                    default="/home0/students/master/2022/wangzy/Pycharm-Remote(161)/WSSS_Model2/datasets/potsdam/",
                    type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=6, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--scales", default=(1.0,), help="multi_scales for seg")


def _validate(model=None, data_loader=None, args=None):
    model.eval()

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda()
        gts, seg_pred = [], []
        count = 0
        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            count += 1

            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            _, _, h, w = inputs.shape
            seg_list = []
            for sc in args.scales:
                _h, _w = int(h * sc), int(w * sc)

                _inputs = F.interpolate(inputs, size=[_h, _w], mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                segs = model(inputs_cat, )[1]
                segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

                seg = segs[:1, ...] + segs[1:, ...].flip(-1)

                seg_list.append(seg)

            seg = torch.max(torch.stack(seg_list, dim=0), dim=0)[0]

            seg_pred += list(torch.argmax(seg, dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            os.makedirs(args.logits_dir, exist_ok=True)

            np.save(args.logits_dir + "/" + name[0] + '.npy', {"msc_seg": seg.cpu().numpy()})

    seg_score = evaluate.scores(gts, seg_pred)

    # print(format_tabs([seg_score], ["seg_pred"], cat_list=potsdam.class_list))

    return seg_score


def crf_proc():
    print("crf post-processing...")

    txt_name = os.path.join(args.list_folder, args.infer_set) + '.txt'

    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    images_path = os.path.join(args.data_folder, 'img_dir', 'test')
    labels_path = os.path.join(args.data_folder, 'ann_dir', 'test_new')

    post_processor = DenseCRF(
        iter_max=10,  # 10
        pos_xy_std=1,  # 3
        pos_w=1,  # 3
        bi_xy_std=121,  # 121, 140
        bi_rgb_std=5,  # 5, 5
        bi_w=4,  # 4, 5
    )

    def _job(i):

        name = name_list[i]

        logit_name = args.logits_dir + "/" + name + ".npy"

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit['msc_seg']

        image_name = os.path.join(images_path, name + ".png")
        image = imageio.imread(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + "_instance_color_RGB.png")
        if "test" in args.infer_set:
            label = image[:, :, 0]
        else:
            label = imageio.imread(label_name)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        imageio.imsave(args.segs_dir + "/" + name + ".png", np.squeeze(pred).astype(np.uint8))
        imageio.imsave(args.segs_rgb_dir + "/" + name + ".png",
                       imutils.encode_cmap4potsdam(np.squeeze(pred)).astype(np.uint8))
        return pred, label

    n_jobs = int(os.cpu_count() * 0.8)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(_job)(i) for i in range(len(name_list))])

    preds, gts = zip(*results)

    crf_score = evaluate.scores(gts, preds)
    print(format_tabs([crf_score], ["seg_crf"], cat_list=potsdam.class_list))
    return crf_score


def validate(args=None):
    val_dataset = potsdam.potsdamSegDataset(
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
                            num_workers=16,
                            pin_memory=False,
                            drop_last=False)

    model = network(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=False,
        aux_layer=-3
    )

    trained_state_dict = torch.load(args.model_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    # # new_state_dict.pop("conv.weight")
    # # new_state_dict.pop("aux_conv.weight")
    model.load_state_dict(state_dict=trained_state_dict, strict=False)
    model.eval()

    seg_score = _validate(model=model, data_loader=val_loader, args=args)

    print(seg_score)

    torch.cuda.empty_cache()
    crf_score = crf_proc()

    print(crf_score)
    return True


if __name__ == "__main__":
    args = parser.parse_args()

    base_dir = args.model_path.split("checkpoints")[0]
    args.logits_dir = os.path.join(base_dir, "segs/logits", args.infer_set)
    args.segs_dir = os.path.join(base_dir, "segs/seg_preds", args.infer_set)
    args.segs_rgb_dir = os.path.join(base_dir, "segs/seg_preds_rgb", args.infer_set)

    os.makedirs(args.segs_dir, exist_ok=True)
    os.makedirs(args.segs_rgb_dir, exist_ok=True)
    os.makedirs(args.logits_dir, exist_ok=True)

    print(args)
    validate(args=args)
