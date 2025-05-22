import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import imageio
import matplotlib.pyplot as plt
from model_test.DSENet.T_model3 import TSCD
from utils.dcrf import DenseCRF
from utils import imutils

from tools.Color_palette import ISAID_palette, Potsdam_palette
from PIL import Image
import torchvision.transforms as T
from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = '8'

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str,
                    default="/home0/students/master/2022/wangzy/datasets/iSAID_WSSS/sampled_process/test/images/P2098_6.png",
                    help="Path to the input image")
parser.add_argument("--model_path", type=str,
                    default="/home0/students/master/2022/wangzy/Pycharm-Remote(161)/WSSS_Model2/weight/ISAID_dataset/2025-01-09-15-53-16-388102/checkpoints/Best mIoU: 40.79923434219596, model: AWTS model_iter_14000.pth",
                    help="Path to the trained model")
parser.add_argument("--output_dir", type=str, default="/home0/students/master/2022/wangzy/OR_image/WSSS/ISAID/seg/",
                    help="Directory to save predictions")
parser.add_argument("--backbone", default='deit_base_patch16_224', type=str, help="Backbone network")
parser.add_argument("--num_classes", default=16, type=int, help="Number of classes")
parser.add_argument("--scales", default=(1.0,), help="Multi-scale inference")
parser.add_argument("--use_crf", default=True, action='store_true', help="Apply CRF post-processing")


def load_model(model_path, num_classes):
    model = TSCD('mit_b1', stride=[4, 2, 2, 1], pooling="gmp", num_classes=num_classes, embedding_dim=256,
                 pretrained=False)

    trained_state_dict = torch.load(model_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    # new_state_dict.pop("conv.weight")
    # new_state_dict.pop("aux_conv.weight")
    model.load_state_dict(state_dict=new_state_dict, strict=False)
    model.cuda().eval()

    return model


def predict_single_image(model, image_path, scales, use_crf=False):
    image = Image.open(image_path).convert('RGB')

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(image).cuda()

    _, H, W = img.shape

    img = img.unsqueeze(0)  # 增加batch维
    image_tensor = torch.cat([img, img], dim=0)

    seg_list = []
    for sc in scales:
        _H, _W = int(H * sc), int(W * sc)
        scaled_image = F.interpolate(image_tensor, size=(_H, _W), mode='bilinear', align_corners=False)

        _, _, seg, _ = model(scaled_image)
        seg = F.interpolate(seg, size=(H, W), mode='bilinear', align_corners=False)
        seg_list.append(seg)

    seg = torch.max(torch.stack(seg_list, dim=0), dim=0)[0]
    prediction = torch.argmax(seg, dim=1).cpu().numpy().astype(np.uint8)[0]

    if use_crf:
        post_processor = DenseCRF(iter_max=10, pos_xy_std=1, pos_w=1, bi_xy_std=121, bi_rgb_std=5, bi_w=4)  # 参数可以调整
        prob = F.softmax(seg, dim=1)[0].detach().cpu().numpy()

        image_np = img.squeeze().cpu().numpy().astype(np.uint8)
        image_np = np.transpose(image_np, (1, 2, 0))

        prediction = np.argmax(post_processor(image_np, prob), axis=0)

    return prediction


import matplotlib.pyplot as plt


def save_prediction(image_path, prediction, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path).replace('.png', '')

    # 定义类别颜色映射 (16 类示例)
    palette = ISAID_palette

    # 生成 RGB 伪彩色图
    color_prediction = palette[prediction]  # (H, W, 3)

    # 保存 RGB 图像
    color_output_path = os.path.join(output_dir, f"{base_name}_color.png")
    Image.fromarray(color_prediction).save(color_output_path)

    # 可视化 RGB 图像
    plt.figure(figsize=(8, 8))
    plt.imshow(color_prediction)
    plt.axis('off')  # 去掉坐标轴
    plt.title("Predicted Segmentation (RGB)")
    plt.show()

    print(f"Saved color-mapped prediction: {color_output_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    model = load_model(args.model_path, args.num_classes)
    prediction = predict_single_image(model, args.image_path, args.scales, args.use_crf)
    save_prediction(args.image_path, prediction, args.output_dir)
