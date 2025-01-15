import torch
import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import cv2
from tqdm import tqdm


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(float(self._data[-1]))

    @property
    def label(self):
        return torch.tensor([float(self._data[i]) for i in range(1, 6)])  # ocean

class myDataset(Dataset):
    def __init__(self, mod='train'):
        super().__init__()
        self.mod = mod
        self.csv_filepath = f'./data/UDIVA/{mod}_label_UDIVA_v0.5.csv'

        path = f'./clip_{mod}_feature_emb_1608_UDIVA.pkl'
        with open(path, 'rb') as f:
            self.data = pickle.load(f)  # 1*1608

        if not os.path.exists(self.csv_filepath):
            print('缺少标签文件')
            exit(1)
        self._parse_list()

    def _parse_list(self):
        tmp = [x.strip().split(',') for x in open(self.csv_filepath)]
        self.video_list = [VideoRecord(item) for item in tmp]
        self.video_list = [item for item in self.video_list if item.path in self.data.keys()]


    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):

        record = self.video_list[index]

        data = self.data[record.path]
        data = np.array(data)
        data = torch.FloatTensor(data).squeeze(0)  # [256]
        label = record.label
        return label, data


if __name__ == '__main__':
    pass
    # for mod in ['train', 'val', 'test']:
    #     getlabel(mod)