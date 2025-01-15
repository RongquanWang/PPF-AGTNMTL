<<<<<<< HEAD
import argparse
import os
import pickle
import random
import shutil
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch import optim

from model import Multimodal as MODEL

warnings.filterwarnings('ignore')

def bell_loss(p, y, *args):
    y_p = torch.pow((y - p), 2)
    y_p_div = -1.0 * torch.div(y_p, 162.0)
    exp_y_p = torch.exp(y_p_div)
    loss = 300 * (1.0 - exp_y_p)
    loss = torch.mean(loss)
    return loss

def logcosh(pred, true, *args):
    loss = torch.log(torch.cosh(pred - true))
    return torch.mean(loss)


def lossFunc(p, y, *args):
    return logcosh(p, y) + bell_loss(p, y)

def acc_func(preds, gts):
    return torch.mean(1 - torch.abs(preds - gts), dim=0)

def otherMetirc_func(preds, gts):
    preds = preds.cpu().numpy()
    gts = gts.cpu().numpy()
    pcc_5, ccc_5, R2_5 = [], [], []
    for i in range(5):
        pred, gt = preds[:, i], gts[:, i]
        pcci = np.corrcoef(pred, gt)[0, 1]
        pcc_5.append(pcci)

        mean_p = np.mean(pred).item()
        mean_y = np.mean(gt).item()
        std_p = np.std(pred).item()
        std_y = np.std(gt).item()
        ccci = 2 * std_y * std_p * pcci / (std_y ** 2 + std_p ** 2 + (mean_y - mean_p) ** 2)
        ccc_5.append(ccci)

        r2i = 1 - ((pred - gt) ** 2).sum() / ((gt - mean_y) ** 2).sum()
        R2_5.append(r2i)

    pcc_5 = np.array(pcc_5)
    pcc = np.mean(pcc_5).item()
    ccc_5 = np.array(ccc_5)
    ccc = np.mean(ccc_5).item()
    R2_5 = np.array(R2_5)
    R2 = np.mean(R2_5).item()

    return pcc_5, pcc, ccc_5, ccc, R2_5, R2

def validate2(model, val_dl, epoch=1):
    model.eval()

    preds = torch.empty((0, 5)).to(device)
    gts = torch.empty((0, 5)).to(device)
    for dl in val_dl:
        with torch.no_grad():
            target = dl[0].to(device)  # [b,5]
            clip_v = dl[1].to(device)
            wav2clip = dl[2].to(device)
            clip_t = dl[3].to(device)
            bg = dl[4].to(device)
            out = model(clip_v, wav2clip, clip_t, bg)
            preds = torch.cat((preds, out['m'].detach()), dim=0)
            gts = torch.cat((gts, target), dim=0)

    acc_5 = acc_func(preds, gts).cpu().numpy()
    epoch_acc = np.mean(acc_5).item()
    pcc_5, epoch_pcc, ccc_5, epoch_ccc, R2_5, R2 = otherMetirc_func(preds, gts)
    loss = LOSSFUNC(preds, gts).item()
    return acc_5, epoch_acc, pcc_5, epoch_pcc, loss, ccc_5, epoch_ccc, R2_5, R2

def test(save_path, mod='test'):
    test_dataset = DATASET(mod=mod)

    test_dl = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                              pin_memory=True, num_workers=1)
    model = MODEL(text_len=text_len)
    # nn.DataParallel(model, device_ids=[0]).cuda()

    checkpoint = torch.load(f"{save_path}/ckpt.best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    acc_5, acc, pcc_5, pcc, test_loss, ccc_5, ccc, R2_5, R2 = validate2(model, test_dl)

    acc_5 = np.around(acc_5, 4)
    pcc_5 = np.around(pcc_5, 4)
    ccc_5 = np.around(ccc_5, 4)

    res = f"""{mod}, Acc: {acc:.4f} {acc_5} | PCC: {pcc:.4f} | CCC: {ccc:.4f} | R2: {R2:.4f} mean:{(acc+pcc+ccc+R2)/4:.4f}"""
    print(res)
    return round(acc, 4), round(pcc, 4), round(ccc, 4), round(R2, 4)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('-i', '--ind', type=int, default=0)
    parser.add_argument('-d', '--dataset', type=str, default='CFIv2', choices=['CFIv2', 'UDIVA'])
    args = parser.parse_args()
    ind = args.ind
    dataset_name = args.dataset
    device = torch.device(f'cuda:{ind % 4}')  # 有4个GPU
    times = 3  # 训练次数
    lr = 1e-4

    batch_size = 16
    text_len = 13
    from utils import myDataset_CFIv2 as DATASET

    clip_1608_path = f"data/text_1608_CLIP.pkl"
    with open(clip_1608_path, 'rb') as f:
        clip_p = pickle.load(f)
        clip_p = torch.FloatTensor(clip_p).to(device)

    res = []

    save_path = f'./result/{dataset_name}'
    # save_path = './result/UDIVA'

    LOSSFUNC = lossFunc
    print(LOSSFUNC.__name__)

    if 'UDIVA' in save_path:
        batch_size = 16
        text_len = 452
        from utils import myDataset_UDIVA as DATASET
    print(save_path)

    test(save_path, 'val')
    test(save_path, 'test')

=======
import argparse
import os
import pickle
import random
import shutil
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch import optim

from model import Multimodal as MODEL

warnings.filterwarnings('ignore')

def bell_loss(p, y, *args):
    y_p = torch.pow((y - p), 2)
    y_p_div = -1.0 * torch.div(y_p, 162.0)
    exp_y_p = torch.exp(y_p_div)
    loss = 300 * (1.0 - exp_y_p)
    loss = torch.mean(loss)
    return loss

def logcosh(pred, true, *args):
    loss = torch.log(torch.cosh(pred - true))
    return torch.mean(loss)


def lossFunc(p, y, *args):
    return logcosh(p, y) + bell_loss(p, y)

def acc_func(preds, gts):
    return torch.mean(1 - torch.abs(preds - gts), dim=0)

def otherMetirc_func(preds, gts):
    preds = preds.cpu().numpy()
    gts = gts.cpu().numpy()
    pcc_5, ccc_5, R2_5 = [], [], []
    for i in range(5):
        pred, gt = preds[:, i], gts[:, i]
        pcci = np.corrcoef(pred, gt)[0, 1]
        pcc_5.append(pcci)

        mean_p = np.mean(pred).item()
        mean_y = np.mean(gt).item()
        std_p = np.std(pred).item()
        std_y = np.std(gt).item()
        ccci = 2 * std_y * std_p * pcci / (std_y ** 2 + std_p ** 2 + (mean_y - mean_p) ** 2)
        ccc_5.append(ccci)

        r2i = 1 - ((pred - gt) ** 2).sum() / ((gt - mean_y) ** 2).sum()
        R2_5.append(r2i)

    pcc_5 = np.array(pcc_5)
    pcc = np.mean(pcc_5).item()
    ccc_5 = np.array(ccc_5)
    ccc = np.mean(ccc_5).item()
    R2_5 = np.array(R2_5)
    R2 = np.mean(R2_5).item()

    return pcc_5, pcc, ccc_5, ccc, R2_5, R2

def validate2(model, val_dl, epoch=1):
    model.eval()

    preds = torch.empty((0, 5)).to(device)
    gts = torch.empty((0, 5)).to(device)
    for dl in val_dl:
        with torch.no_grad():
            target = dl[0].to(device)  # [b,5]
            clip_v = dl[1].to(device)
            wav2clip = dl[2].to(device)
            clip_t = dl[3].to(device)
            bg = dl[4].to(device)
            out = model(clip_v, wav2clip, clip_t, bg)
            preds = torch.cat((preds, out['m'].detach()), dim=0)
            gts = torch.cat((gts, target), dim=0)

    acc_5 = acc_func(preds, gts).cpu().numpy()
    epoch_acc = np.mean(acc_5).item()
    pcc_5, epoch_pcc, ccc_5, epoch_ccc, R2_5, R2 = otherMetirc_func(preds, gts)
    loss = LOSSFUNC(preds, gts).item()
    return acc_5, epoch_acc, pcc_5, epoch_pcc, loss, ccc_5, epoch_ccc, R2_5, R2

def test(save_path, mod='test'):
    test_dataset = DATASET(mod=mod)

    test_dl = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                              pin_memory=True, num_workers=1)
    model = MODEL(text_len=text_len)
    # nn.DataParallel(model, device_ids=[0]).cuda()

    checkpoint = torch.load(f"{save_path}/ckpt.best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    acc_5, acc, pcc_5, pcc, test_loss, ccc_5, ccc, R2_5, R2 = validate2(model, test_dl)

    acc_5 = np.around(acc_5, 4)
    pcc_5 = np.around(pcc_5, 4)
    ccc_5 = np.around(ccc_5, 4)

    res = f"""{mod}, Acc: {acc:.4f} {acc_5} | PCC: {pcc:.4f} | CCC: {ccc:.4f} | R2: {R2:.4f} mean:{(acc+pcc+ccc+R2)/4:.4f}"""
    print(res)
    return round(acc, 4), round(pcc, 4), round(ccc, 4), round(R2, 4)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('-i', '--ind', type=int, default=0)
    parser.add_argument('-d', '--dataset', type=str, default='CFIv2', choices=['CFIv2', 'UDIVA'])
    args = parser.parse_args()
    ind = args.ind
    dataset_name = args.dataset
    device = torch.device(f'cuda:{ind % 4}')  # 有4个GPU
    times = 3  # 训练次数
    lr = 1e-4

    batch_size = 16
    text_len = 13
    from utils import myDataset_CFIv2 as DATASET

    clip_1608_path = f"data/text_1608_CLIP.pkl"
    with open(clip_1608_path, 'rb') as f:
        clip_p = pickle.load(f)
        clip_p = torch.FloatTensor(clip_p).to(device)

    res = []

    save_path = f'./result/{dataset_name}'
    # save_path = './result/UDIVA'

    LOSSFUNC = lossFunc
    print(LOSSFUNC.__name__)

    if 'UDIVA' in save_path:
        batch_size = 16
        text_len = 452
        from utils import myDataset_UDIVA as DATASET
    print(save_path)

    test(save_path, 'val')
    test(save_path, 'test')

>>>>>>> origin/main
