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
# from model import Multimodal_ablation2 as MODEL
print(MODEL.__name__)

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
            # _, stau_f = model_stau(node_fa, node_fg)  # [b,16,384]
            out = model(clip_v, wav2clip, clip_t, bg)
            preds = torch.cat((preds, out['m'].detach()), dim=0)
            gts = torch.cat((gts, target), dim=0)

    acc_5 = acc_func(preds, gts).cpu().numpy()
    epoch_acc = np.mean(acc_5).item()
    pcc_5, epoch_pcc, ccc_5, epoch_ccc, R2_5, R2 = otherMetirc_func(preds, gts)
    loss = LOSSFUNC(preds, gts).item()
    return acc_5, epoch_acc, pcc_5, epoch_pcc, loss, ccc_5, epoch_ccc, R2_5, R2


class SELF_MM(nn.Module):
    def __init__(self, train_samples, device, lr, wd, epochs):
        super().__init__()
        self.device = device
        self.lr = lr
        self.wd = wd
        self.epochs = epochs

        self.tasks = ['M', 'CC', 'CW', 'T']

        self.label_map = {
            'm': torch.zeros((train_samples, 5), requires_grad=False).to(device),
            'cc1': torch.zeros((train_samples, 5), requires_grad=False).to(device),
            'clip_clip': torch.zeros((train_samples, 5), requires_grad=False).to(device),
            'clip_wav': torch.zeros((train_samples, 5), requires_grad=False).to(device),
            'clip_t': torch.zeros((train_samples, 5), requires_grad=False).to(device),
        }
        self.name_map = {
            'M': 'm',
            'CC': 'clip_clip',
            'CW': 'clip_wav',
            'T': 'clip_t',
        }

    def init_labels(self, indexes, m_labels):
        self.label_map['m'][indexes] = m_labels
        self.label_map['clip_clip'][indexes] = m_labels
        self.label_map['clip_wav'][indexes] = m_labels
        self.label_map['clip_t'][indexes] = m_labels

    def weighted_loss(self, y_pred, y_true, indexes=None, mode='m'):
        if mode == 'm':
            weighted = 0.5 * torch.ones_like(y_pred)
        else:
            weighted = 0.1 * torch.ones_like(y_pred)
            # weighted = torch.tanh(torch.abs(self.label_map[mode][indexes] - self.label_map['m'][indexes]))

        loss = torch.mean(weighted * LOSSFUNC(y_pred, y_true))
        return loss


    def do_train(self, model, train_dl, val_dl, save_path):
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)
        # 初始化各类别标签
        for dl in train_dl:
            labels_m = dl[0].to(self.device)  # [b,5]
            indexes = dl[5].tolist()
            self.init_labels(indexes, labels_m)

        best_loss = 1e8
        count = 0
        best_epoch = 0
        best_acc = 0
        for epoch in range(self.epochs):
            start_time = datetime.now()
            model.train()
            adjust_learning_rate(optimizer, epoch, self.lr, lr_type='cos', lr_steps=[50, 100])
            for dl in train_dl:
                target = dl[0].to(device)  # [b,5]
                clip_v = dl[1].to(device)
                wav2clip = dl[2].to(device)
                clip_t = dl[3].to(device)
                bg = dl[4].to(device)
                index = dl[5].to(device)

                outputs = model(clip_v, wav2clip, clip_t, bg)
                loss = 0.0
                for m in self.tasks:
                    loss += self.weighted_loss(outputs[self.name_map[m]], self.label_map[self.name_map[m]][index],
                                               indexes=index, mode=self.name_map[m])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # print(=ee
            #     'After {} epochs ,mseloss is {:.6f} lr:{}'.format(epoch, loss.item(), optimizer.param_groups[0]["lr"]))

            model.eval()
            _, train_acc, _, train_pcc, train_loss, _, train_ccc, _, train_R2 = validate2(model, train_dl, epoch+1)
            acc_5, acc, pcc_5, pcc, val_loss, ccc_5, ccc, _, R2 = validate2(model, val_dl, epoch+1)

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
            if count == 20:
                break
            # if val_loss < 0.001:
            #     break


def main(save_path):
    train_dataset = DATASET(mod='train')
    val_dataset = DATASET(mod='val')

    train_samples = len(train_dataset)

    train_dl = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                               pin_memory=True, num_workers=1)
    val_dl = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = MODEL(text_len=text_len)

    self_mm = SELF_MM(train_samples=train_samples, device=device, lr=lr, wd=1e-4, epochs=300)
    best_acc = self_mm.do_train(model, train_dl, val_dl, save_path)
    return best_acc


def test(save_path, mod='test'):
    test_dataset = DATASET(mod=mod)

    test_dl = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                              pin_memory=True, num_workers=16)
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
    device = torch.device(f'cuda:{ind % 4}')  # 4 gpu
    times = 3  # times of training
    lr = 1e-4


    if dataset_name == 'UDIVA':
        batch_size = 64
        text_len = 452
        from utils import myDataset_UDIVA as DATASET
    else:
        batch_size = 128
        text_len = 13
        from utils import myDataset_CFIv2 as DATASET

    clip_1608_path = f"./data/text_1608_CLIP.pkl"
    with open(clip_1608_path, 'rb') as f:
        clip_p = pickle.load(f)
        clip_p = torch.FloatTensor(clip_p).to(device)

    res = []
    save_path = './result/{}_tmp_{}/{}'
    # save_path = './result/{}_text-bilstm_{}/{}'
    # save_path = './result/{}_graph-gru_{}/{}'

    LOSSFUNC = lossFunc
    print(LOSSFUNC.__name__)

    print(save_path)
    for i in range(times):
        path = save_path.format(dataset_name, ind, i)
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
        path = save_path.format(dataset_name, ind, i)
        print(f'{i} =======================================================================')
        test(path, 'val')
        test(path, 'test')

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
# from model import Multimodal_ablation2 as MODEL
print(MODEL.__name__)

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
            # _, stau_f = model_stau(node_fa, node_fg)  # [b,16,384]
            out = model(clip_v, wav2clip, clip_t, bg)
            preds = torch.cat((preds, out['m'].detach()), dim=0)
            gts = torch.cat((gts, target), dim=0)

    acc_5 = acc_func(preds, gts).cpu().numpy()
    epoch_acc = np.mean(acc_5).item()
    pcc_5, epoch_pcc, ccc_5, epoch_ccc, R2_5, R2 = otherMetirc_func(preds, gts)
    loss = LOSSFUNC(preds, gts).item()
    return acc_5, epoch_acc, pcc_5, epoch_pcc, loss, ccc_5, epoch_ccc, R2_5, R2


class SELF_MM(nn.Module):
    def __init__(self, train_samples, device, lr, wd, epochs):
        super().__init__()
        self.device = device
        self.lr = lr
        self.wd = wd
        self.epochs = epochs

        self.tasks = ['M', 'CC', 'CW', 'T']

        self.label_map = {
            'm': torch.zeros((train_samples, 5), requires_grad=False).to(device),
            'cc1': torch.zeros((train_samples, 5), requires_grad=False).to(device),
            'clip_clip': torch.zeros((train_samples, 5), requires_grad=False).to(device),
            'clip_wav': torch.zeros((train_samples, 5), requires_grad=False).to(device),
            'clip_t': torch.zeros((train_samples, 5), requires_grad=False).to(device),
        }
        self.name_map = {
            'M': 'm',
            'CC': 'clip_clip',
            'CW': 'clip_wav',
            'T': 'clip_t',
        }

    def init_labels(self, indexes, m_labels):
        self.label_map['m'][indexes] = m_labels
        self.label_map['clip_clip'][indexes] = m_labels
        self.label_map['clip_wav'][indexes] = m_labels
        self.label_map['clip_t'][indexes] = m_labels

    def weighted_loss(self, y_pred, y_true, indexes=None, mode='m'):
        if mode == 'm':
            weighted = 0.5 * torch.ones_like(y_pred)
        else:
            weighted = 0.1 * torch.ones_like(y_pred)
            # weighted = torch.tanh(torch.abs(self.label_map[mode][indexes] - self.label_map['m'][indexes]))

        loss = torch.mean(weighted * LOSSFUNC(y_pred, y_true))
        return loss


    def do_train(self, model, train_dl, val_dl, save_path):
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)
        # 初始化各类别标签
        for dl in train_dl:
            labels_m = dl[0].to(self.device)  # [b,5]
            indexes = dl[5].tolist()
            self.init_labels(indexes, labels_m)

        best_loss = 1e8
        count = 0
        best_epoch = 0
        best_acc = 0
        for epoch in range(self.epochs):
            start_time = datetime.now()
            model.train()
            adjust_learning_rate(optimizer, epoch, self.lr, lr_type='cos', lr_steps=[50, 100])
            for dl in train_dl:
                target = dl[0].to(device)  # [b,5]
                clip_v = dl[1].to(device)
                wav2clip = dl[2].to(device)
                clip_t = dl[3].to(device)
                bg = dl[4].to(device)
                index = dl[5].to(device)

                outputs = model(clip_v, wav2clip, clip_t, bg)
                loss = 0.0
                for m in self.tasks:
                    loss += self.weighted_loss(outputs[self.name_map[m]], self.label_map[self.name_map[m]][index],
                                               indexes=index, mode=self.name_map[m])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # print(=ee
            #     'After {} epochs ,mseloss is {:.6f} lr:{}'.format(epoch, loss.item(), optimizer.param_groups[0]["lr"]))

            model.eval()
            _, train_acc, _, train_pcc, train_loss, _, train_ccc, _, train_R2 = validate2(model, train_dl, epoch+1)
            acc_5, acc, pcc_5, pcc, val_loss, ccc_5, ccc, _, R2 = validate2(model, val_dl, epoch+1)

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
            if count == 20:
                break
            # if val_loss < 0.001:
            #     break


def main(save_path):
    train_dataset = DATASET(mod='train')
    val_dataset = DATASET(mod='val')

    train_samples = len(train_dataset)

    train_dl = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                               pin_memory=True, num_workers=1)
    val_dl = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = MODEL(text_len=text_len)

    self_mm = SELF_MM(train_samples=train_samples, device=device, lr=lr, wd=1e-4, epochs=300)
    best_acc = self_mm.do_train(model, train_dl, val_dl, save_path)
    return best_acc


def test(save_path, mod='test'):
    test_dataset = DATASET(mod=mod)

    test_dl = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                              pin_memory=True, num_workers=16)
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
    device = torch.device(f'cuda:{ind % 4}')  # 4 gpu
    times = 3  # times of training
    lr = 1e-4


    if dataset_name == 'UDIVA':
        batch_size = 64
        text_len = 452
        from utils import myDataset_UDIVA as DATASET
    else:
        batch_size = 128
        text_len = 13
        from utils import myDataset_CFIv2 as DATASET

    clip_1608_path = f"./data/text_1608_CLIP.pkl"
    with open(clip_1608_path, 'rb') as f:
        clip_p = pickle.load(f)
        clip_p = torch.FloatTensor(clip_p).to(device)

    res = []
    save_path = './result/{}_tmp_{}/{}'
    # save_path = './result/{}_text-bilstm_{}/{}'
    # save_path = './result/{}_graph-gru_{}/{}'

    LOSSFUNC = lossFunc
    print(LOSSFUNC.__name__)

    print(save_path)
    for i in range(times):
        path = save_path.format(dataset_name, ind, i)
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
        path = save_path.format(dataset_name, ind, i)
        print(f'{i} =======================================================================')
        test(path, 'val')
        test(path, 'test')

>>>>>>> origin/main
