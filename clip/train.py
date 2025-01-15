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
from torch import optim
from torch.cuda.amp import GradScaler
from datetime import datetime
from utils import myDataset as DATASET
from clip_model import Model as MODEL
print(MODEL.__name__)

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True


def bell_loss(p, y):
    y_p = torch.pow((y - p), 2)
    y_p_div = -1.0 * torch.div(y_p, 162.0)
    exp_y_p = torch.exp(y_p_div)
    loss = 300 * (1.0 - exp_y_p)
    loss = torch.mean(loss)
    return loss


def logcosh(pred, true):
    loss = torch.log(torch.cosh(pred - true))
    return torch.mean(loss)


def lossFunc(p, y):
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


def adjust_learning_rate(optimizer, epoch, lr, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr
    weight_decay = 1e-4
    epochs = 100
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = lr * decay
        decay = weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * lr * (1 + math.cos(math.pi * epoch / epochs))
        decay = weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = decay
        # param_group['lr'] = lr * param_group['lr_mult']
        # param_group['weight_decay'] = decay * param_group['decay_mult']


def save_checkpoint(state, is_best, save_path):
    filename = '{}/ckpt.pth.tar'.format(save_path)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


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

    loss = LOSSFUNC(preds, gts).item()
    return loss, acc_5, epoch_acc, pcc_5, epoch_pcc, ccc_5, epoch_ccc, R2_5, epoch_R2


def train(model, train_dl, val_dl, save_path, lr=0.0001, wd=1e-4, epochs=100):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scaler = GradScaler()

    best_loss = 1e8
    count = 0
    best_epoch = 0
    best_acc = 0
    for epoch in range(epochs):
        start_time = datetime.now()
        model.train()
        adjust_learning_rate(optimizer, epoch, lr, lr_type='cos', lr_steps=[50, 100])
        for dl in train_dl:
            target = dl[0].to(device)  # [b,5]
            x = dl[1].to(device)
            outputs, _ = model(x)
            loss = LOSSFUNC(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()  # 模型评估
        train_loss, _, train_acc, _, train_pcc, _, train_ccc, _, train_R2 = validate2(model, train_dl, epoch)
        val_loss, acc_5, acc, pcc_5, pcc, ccc_5, ccc, R2_5, R2 = validate2(model, val_dl, epoch)

        if val_loss < best_loss:
            is_best = True
            count = 0
            best_loss = val_loss
            best_epoch = epoch
            best_acc = acc
        else:
            is_best = False
            count += 1


        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'val_loss': val_loss,
            'val_pcc': pcc,
            'val_acc': acc,
        }, is_best, save_path)

        end_time = datetime.now()
        cost_time = end_time - start_time
        acc_5 = np.around(acc_5, 4)
        pcc_5 = np.around(pcc_5, 4)
        ccc_5 = np.around(ccc_5, 4)
        res = f"""Epoch: {epoch} | Time: {cost_time} | Best_epoch: {best_epoch} | Best Val Acc: {best_acc:.4f}
\tTrain loss: {train_loss:.4f},  Acc: {train_acc:.4f}, PCC: {train_pcc:.4f}, CCC: {train_ccc:.4f}, R2: {train_R2:.4f}
\tVal loss: {val_loss:.4f} | Acc: {acc:.4f} | PCC: {pcc:.4f} | CCC: {ccc:.4f} | R2: {R2:.4f}"""
        print(res)

        if count == 30:
            break

def main(save_path):
    train_dataset = DATASET(mod='train')
    val_dataset = DATASET(mod='val')

    train_dl = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                               pin_memory=True, num_workers=16)
    val_dl = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16)
    train_dataset = None
    val_dataset = None
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = MODEL()
    model = model.to(device)
    train(model, train_dl, val_dl, save_path, epochs=epochs, lr=lr, wd=1e-4)
    print('done!')
    train_dl, val_dl = None, None
    # test(val_dl,save_path)



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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('-i', '--ind', type=int, default=0)
    args = parser.parse_args()
    ind = args.ind
    device = torch.device(f'cuda:{ind % 4}')  # 有4个GPU
    times = 3
    batch_size = 64
    lr = 2e-5
    epochs = 100

    res = []
    LOSSFUNC = lossFunc
    print(LOSSFUNC.__name__)
    save_path = './result/bg_{}/{}'

    print(save_path)
    for i in range(times):
        path = save_path.format(ind, i)
        print(path)
        # 训练和验证
        main(path)
        # test
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