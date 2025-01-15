import os
import pickle
import random
import shutil
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from utils import myDataset as DATASET
from tqdm import tqdm, trange
from clip_model import Model as MODEL

print(MODEL.__name__)

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True




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



def validate2(model, val_dl, epoch=None):
    model.eval()
    preds = torch.empty((0, 5)).to(device)
    gts = torch.empty((0, 5)).to(device)
    with torch.no_grad():
        for dl in val_dl:
            target = dl[0].to(device)  # [b,5]
            x = dl[1].to(device)
            out, _ = model(x)
            preds = torch.cat((preds, out.detach()), dim=0)
            gts = torch.cat((gts, target), dim=0)
    acc_5 = acc_func(preds, gts).cpu().numpy()
    epoch_acc = np.mean(acc_5).item()
    pcc_5, epoch_pcc, ccc_5, epoch_ccc, R2_5, epoch_R2 = otherMetirc_func(preds, gts)

    # loss = LOSSFUNC(preds, gts).item()
    loss = 0
    return loss, acc_5, epoch_acc, pcc_5, epoch_pcc, ccc_5, epoch_ccc, R2_5, epoch_R2


def test(save_path, mod='test'):
    test_dataset = DATASET(mod=mod)
    test_dl = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                              pin_memory=True, num_workers=16)
    model = MODEL()
    model = model.to(device)

    checkpoint = torch.load(f"{save_path}/ckpt.best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    test_loss, acc_5, acc, pcc_5, pcc, ccc_5, ccc, R2_5, R2 = validate2(model, test_dl, 100)

    acc_5 = np.around(acc_5, 4)
    pcc_5 = np.around(pcc_5, 4)
    ccc_5 = np.around(ccc_5, 4)
    R2_5 = np.around(R2_5, 4)

    res = f"""{mod}, Acc: {acc:.4f} {acc_5} | PCC: {pcc:.4f} | CCC: {ccc:.4f} | R2: {R2:.4f} mean:{(acc+pcc+ccc+R2)/4:.4f}"""
    print(res)

    return round(acc, 4), round(pcc, 4), round(ccc, 4), round(R2, 4)


def extract_feature(save_path):
    model = MODEL()
    model = model.to(device)

    checkpoint = torch.load(f"{save_path}/ckpt.best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    for mod in ['train', 'val', 'test']:
        dataset = DATASET(mod=mod)
        feats = dict()
        with torch.no_grad():
            for i in trange(len(dataset)):
                data = dataset[i]
                name = dataset.video_list[i].path
                x = data[1].to(device).unsqueeze(0)

                _, ff = model(x)
                feats[name] = ff.cpu().numpy()
        with open(f'./clip_{mod}_feature_emb_ft_udiva.pkl', 'wb') as f:
            pickle.dump(feats, f)
        print(f"Extract Feature For {mod.upper()} Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('-i', '--ind', type=int, default=0)
    args = parser.parse_args()
    ind = args.ind
    device = torch.device(f'cuda:{ind % 4}')  # 有4个GPU
    # device = torch.device('cpu')
    times = 3
    batch_size = 128

    res = []
    TEST5, VAL5, TEST, VAL = [], [], [], []
    save_path = './cat_result/bg_{}/{}'

    print(save_path)
    for i in range(times):
        path = save_path.format(ind, i)

        acc_meanv, pcc_meanv, ccc_meanv, R2_meanv = test(path, 'val')
        acc_mean, pcc_mean, ccc_mean, R2_mean = test(path, 'test')
        print(path)
        res.append((acc_meanv, pcc_meanv, ccc_meanv, R2_meanv, acc_mean, pcc_mean, ccc_mean, R2_mean))
    for r in res:
        meanv = (r[0] + r[1] + r[2] + r[3]) / 4
        meant = (r[4] + r[5] + r[6] + r[7]) / 4
        print(
            f'ACC: {r[0]:.4f}/{r[4]:.4f} | PCC: {r[1]:.4f}/{r[5]:.4f} | CCC: {r[2]:.4f}/{r[6]:.4f} | R2: {r[3]:.4f}/{r[7]:.4f} | mean: {meanv:.4f}/{meant:.4f}')
    for i in range(times):
        path = save_path.format(ind, i)
        print(f'{i} =======================================================================')
        test(path, 'val')
        test(path, 'test')

    print(MODEL.__name__)


    extract_feature(save_path.format(0, 0))
