import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


def Creat_LineGraph(loss):

    x_loss = range(len(loss))
    y_loss = loss

    # plt.plot(x, y, color="g", label="loss ", linewidth=0.7, marker=',')
    plt.plot(x_loss, y_loss, color="g", label=" loss", linewidth=0.7, marker=',')
    plt.xlabel('Epoch')
    plt.ylabel('Segloss Value')
    plt.show()


def encode_cmap(label):
    cmap = colormap()
    return cmap[label.astype(np.int16), :]


def encode_cmap4potsdam(label):
    cmap = colormap4potsdam()
    return cmap[label.astype(np.int16), :]


def encode_cmap4deepglobe(label):
    cmap = colormap4deepglobe()
    return cmap[label.astype(np.int16), :]


def encode_cmap4iSAID(label):
    cmap = colormap4iSAID()
    return cmap[label.astype(np.int16), :]


def denormalize_img(imgs=None, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    _imgs = torch.zeros_like(imgs)
    _imgs[:, 0, :, :] = imgs[:, 0, :, :] * std[0] + mean[0]
    _imgs[:, 1, :, :] = imgs[:, 1, :, :] * std[1] + mean[1]
    _imgs[:, 2, :, :] = imgs[:, 2, :, :] * std[2] + mean[2]
    _imgs = _imgs.type(torch.uint8)

    return _imgs


def denormalize_img2(imgs=None):
    # _imgs = torch.zeros_like(imgs)
    imgs = denormalize_img(imgs)

    return imgs / 255.0


def minmax_norm(x):
    for i in range(x.shape[0]):
        x[i, ...] = x[i, ...] - x[i, ...].min()
        x[i, ...] = x[i, ...] / x[i, ...].max()
    return x


def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def colormap4iSAID(N=256, normalized=False):
    # Your color table
    color_table = {
        (0, 127, 255): 14,  # plane
        (0, 0, 127): 9,  # small_vehicle
        (0, 127, 127): 8,  # large_vehicle
        (0, 0, 63): 1,  # ship
        (0, 100, 155): 15,  # harbor
        (0, 63, 127): 4,  # tennis_court
        (0, 0, 255): 11,  # swimming_pool
        (0, 63, 255): 3,  # baseball_diamond
        (0, 63, 63): 2,  # stroage_tank
        (0, 127, 63): 7,  # bridge
        (0, 63, 0): 6,  # Ground_Track_Field
        (0, 0, 191): 10,  # helicopter
        (0, 63, 191): 5,  # basketball_court
        (0, 127, 191): 13,  # Soccer_ball_field
        (0, 191, 127): 12  # Roundabout
    }
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)

    for color, index in color_table.items():
        cmap[index] = np.array(color[::-1])  # Reverse the order for BGR format

    cmap = cmap / 255 if normalized else cmap
    return cmap


def tensorboard_image(imgs=None, cam=None, ):
    ## images

    _imgs = denormalize_img(imgs=imgs)
    grid_imgs = torchvision.utils.make_grid(tensor=_imgs, nrow=2)

    cam = F.interpolate(cam, size=_imgs.shape[2:], mode='bilinear', align_corners=False)
    cam = cam.cpu()
    cam_max = cam.max(dim=1)[0]
    cam_heatmap = plt.get_cmap('jet')(cam_max.numpy())[:, :, :, 0:3] * 255
    cam_cmap = torch.from_numpy(cam_heatmap).permute([0, 3, 1, 2])
    cam_img = cam_cmap * 0.5 + _imgs.cpu() * 0.5
    grid_cam = torchvision.utils.make_grid(tensor=cam_img.type(torch.uint8), nrow=2)

    return grid_imgs, grid_cam


def tensorboard_edge(edge=None, n_row=2):
    ##
    edge = F.interpolate(edge, size=[224, 224], mode='bilinear', align_corners=False)[:, 0, ...]
    edge = edge.cpu()
    edge_heatmap = plt.get_cmap('viridis')(edge.numpy())[:, :, :, 0:3] * 255
    edge_cmap = torch.from_numpy(edge_heatmap).permute([0, 3, 1, 2])

    grid_edge = torchvision.utils.make_grid(tensor=edge_cmap.type(torch.uint8), nrow=n_row)

    return grid_edge


def tensorboard_attn(attns=None, size=[224, 224], n_pix=0, n_row=4):
    n = len(attns)
    imgs = []
    for idx, attn in enumerate(attns):

        b, hw, _ = attn.shape
        h = w = int(np.sqrt(hw))

        attn_ = attn.clone()  # - attn.min()
        # attn_ = attn_ / attn_.max()
        _n_pix = int(h * n_pix) * (w + 1)
        attn_ = attn_[:, _n_pix, :].reshape(b, 1, h, w)

        attn_ = F.interpolate(attn_, size=size, mode='bilinear', align_corners=True)

        attn_ = attn_.cpu()[:, 0, :, :]

        def minmax_norm(x):
            for i in range(x.shape[0]):
                x[i, ...] = x[i, ...] - x[i, ...].min()
                x[i, ...] = x[i, ...] / x[i, ...].max()
            return x

        attn_ = minmax_norm(attn_)

        attn_heatmap = plt.get_cmap('viridis')(attn_.numpy())[:, :, :, 0:3] * 255
        attn_heatmap = torch.from_numpy(attn_heatmap).permute([0, 3, 2, 1])
        imgs.append(attn_heatmap)
    attn_img = torch.cat(imgs, dim=0)

    grid_attn = torchvision.utils.make_grid(tensor=attn_img.type(torch.uint8), nrow=n_row).permute(0, 2, 1)

    return grid_attn


def tensorboard_attn2(attns=None, size=[224, 224], n_pixs=[0.0, 0.3, 0.6, 0.9], n_row=4, with_attn_pred=True):
    n = len(attns)
    attns_top_layers = []
    attns_last_layer = []
    grid_attns = []
    if with_attn_pred:
        _attns_top_layers = attns[:-3]
        _attns_last_layer = attns[-3:-1]
    else:
        _attns_top_layers = attns[:-2]
        _attns_last_layer = attns[-2:]

    attns_top_layers = [_attns_top_layers[i][:, 0, ...] for i in range(len(_attns_top_layers))]
    if with_attn_pred:
        attns_top_layers.append(attns[-1])
    grid_attn_top_case0 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[0], n_row=n_row)
    grid_attn_top_case1 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[1], n_row=n_row)
    grid_attn_top_case2 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[2], n_row=n_row)
    grid_attn_top_case3 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[3], n_row=n_row)
    grid_attns.append(grid_attn_top_case0)
    grid_attns.append(grid_attn_top_case1)
    grid_attns.append(grid_attn_top_case2)
    grid_attns.append(grid_attn_top_case3)

    for attn in _attns_last_layer:
        for i in range(attn.shape[1]):
            attns_last_layer.append(attn[:, i, :, :])
    grid_attn_last_case0 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[0], n_row=2 * n_row)
    grid_attn_last_case1 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[1], n_row=2 * n_row)
    grid_attn_last_case2 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[2], n_row=2 * n_row)
    grid_attn_last_case3 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[3], n_row=2 * n_row)
    grid_attns.append(grid_attn_last_case0)
    grid_attns.append(grid_attn_last_case1)
    grid_attns.append(grid_attn_last_case2)
    grid_attns.append(grid_attn_last_case3)

    return grid_attns


def tensorboard_label(labels=None):
    ## labels
    labels_cmap = encode_cmap(np.squeeze(labels))
    labels_cmap = torch.from_numpy(labels_cmap).permute([0, 3, 1, 2])
    grid_labels = torchvision.utils.make_grid(tensor=labels_cmap, nrow=2)

    return grid_labels


def colormap4potsdam(N=256, normalized=False):
    # Your color table
    color_table = {
        (255, 255, 255): 0,  # imp surf white  [255,255,255]
        (255, 0, 0): 1,  # building deep blue [255,0,0]
        (255, 255, 0): 2,  # low vegetation light blue [255,255,0]
        (0, 255, 0): 3,  # tree green [0,255,0]
        (0, 255, 255): 4,  # car yellow [0,255,255]
        (0, 0, 255): 5,  # clutter red [0,0,255]
    }
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)

    for color, index in color_table.items():
        cmap[index] = np.array(color[::-1])  # Reverse the order for BGR format

    cmap = cmap / 255 if normalized else cmap
    return cmap


def colormap4deepglobe(N=256, normalized=False):
    # Your color table
    color_table = {
        (255, 255, 0): 0,  # urban blue
        (0, 255, 255): 1,  # agriculture yellow
        (255, 0, 255): 2,  # range land pink
        (0, 255, 0): 3,  # forest green
        (255, 0, 0): 4,  # water deep blue
        (255, 255, 255): 5,  # barren white [0,0,255]
        (0, 0, 0): 6,  # unknown black
    }
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)

    for color, index in color_table.items():
        cmap[index] = np.array(color[::-1])  # Reverse the order for BGR format

    cmap = cmap / 255 if normalized else cmap
    return cmap
