import torch
import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
# import cv2
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


class myDataset_CFIv2(Dataset):  #大五数据集
    def __init__(self, mod='train'):
        super().__init__()
        self.mod = mod
        self.csv_filepath = f'./data/{mod}_label_all.csv'   # 标签文件
        self.max_text_len_seq = 13

        clip_image_path = f'./data/{mod}_clipimage.pkl' # video from clip, 15 frames
        with open(clip_image_path, 'rb') as f:
            self.video = pickle.load(f)

        wav2clip_path = f"./data/{mod}_15_audio_wav2clip.pkl"   # audio wav2clip
        with open(wav2clip_path, 'rb') as f:
            self.wav2clip = pickle.load(f)

        clip_t_path = f"./data/{mod}_clipsentence.pkl"  # clip_text
        with open(clip_t_path, 'rb') as f:
            self.clip_t = pickle.load(f)
        self.clip_t = {k: v for k, v in self.clip_t.items() if len(v) != 0}

        bg_path = f"./data/clip_{mod}_feature_emb_ft.pkl"  #  Scene-descriptions association feature,
                            # BG feature is pre-trained and extracted through code in the 'clip' folder.
                            # In our previous work, we conducted separate experiments and proved that
                            # regardless of whether pre extracted features are used, multimodal models can achieve the same performance.
        with open(bg_path, 'rb') as f:
            self.bg = pickle.load(f)

        if not os.path.exists(self.csv_filepath):
            print('缺少标签文件')
            exit(1)
        self._parse_list()

    def _parse_list(self):
        tmp = [x.strip().split(',') for x in open(self.csv_filepath)]
        self.video_list = [VideoRecord(item) for item in tmp]
        self.video_list = [item for item in self.video_list if item.path + '.mp4' in self.video.keys()]
        self.video_list = [item for item in self.video_list if item.path + '.mp4' in self.wav2clip.keys()]
        self.video_list = [item for item in self.video_list if item.path + '.mp4' in self.clip_t.keys()]
        self.video_list = [item for item in self.video_list if item.path in self.bg.keys()]

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):

        record = self.video_list[index]

        v = self.video[record.path + '.mp4']
        v = torch.stack(v, dim=0).squeeze(0).to(torch.float32)  # [15, 768]
        v = torch.FloatTensor(v)

        wav2clip = self.wav2clip[record.path + '.mp4']
        wav2clip = torch.FloatTensor(wav2clip)  # [15,512]

        t = self.clip_t[record.path + '.mp4']
        t = torch.stack(t, dim=0).to(torch.float32)  # [n, 768]  # 最大长度13
        t = torch.FloatTensor(t)
        t_len = t.shape[0]
        t = F.pad(t, (0, 0, 0, self.max_text_len_seq - t_len), "constant", 0)

        bg = self.bg[record.path]
        bg = np.array(bg)
        bg = torch.FloatTensor(bg).squeeze(0)  # [256]

        label = record.label
        return label, v, wav2clip, t, bg, index

class myDataset_UDIVA(Dataset):
    def __init__(self, mod='train'):
        super().__init__()
        self.mod = mod
        self.csv_filepath = f'./data/{mod}_label_UDIVA_v0.5.csv'
        self.max_text_len_seq = 452

        clip_image_path = f'./data/{mod}_clipimage_UDIVA_v0.5.pkl' # video from clip, 15 frames
        with open(clip_image_path, 'rb') as f:
            self.video = pickle.load(f)

        wav2clip_path = f"./data/{mod}_15_audio_wav2clip_UDIVA_v0.5.pkl" # audio wav2clip
        with open(wav2clip_path, 'rb') as f:
            self.wav2clip = pickle.load(f)

        clip_t_path = f"./data/{mod}_clipsentence_UDIVA_v0.5.pkl" # clip_text
        with open(clip_t_path, 'rb') as f:
            self.clip_t = pickle.load(f)
        self.clip_t = {k: v for k, v in self.clip_t.items() if len(v) != 0}

        bg_path = f"./data/clip_{mod}_feature_emb_ft_udiva.pkl"  #  Scene-descriptions association feature,
                            # BG feature is pre-trained and extracted through code in the 'clip' folder.
                            # In our previous work, we conducted separate experiments and proved that
                            # regardless of whether pre extracted features are used, multimodal models can achieve the same performance.
        with open(bg_path, 'rb') as f:
            self.bg = pickle.load(f)

        if not os.path.exists(self.csv_filepath):
            print('缺少标签文件')
            exit(1)
        self._parse_list()

    def _parse_list(self):
        tmp = [x.strip().split(',') for x in open(self.csv_filepath)]
        self.video_list = [VideoRecord(item) for item in tmp]
        self.video_list = [item for item in self.video_list if item.path in self.video.keys()]
        self.video_list = [item for item in self.video_list if item.path in self.wav2clip.keys()]
        self.video_list = [item for item in self.video_list if item.path in self.clip_t.keys()]
        self.video_list = [item for item in self.video_list if item.path in self.bg.keys()]

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):

        record = self.video_list[index]

        v = self.video[record.path]
        v = np.vstack(v)  # [15, 768]
        v = torch.FloatTensor(v)

        wav2clip = self.wav2clip[record.path]
        wav2clip = torch.FloatTensor(wav2clip)  # [15,512]

        t = self.clip_t[record.path]
        t = torch.FloatTensor(t)
        t_len = t.shape[0]
        t = F.pad(t, (0, 0, 0, self.max_text_len_seq - t_len), "constant", 0)

        bg = self.bg[record.path]
        bg = np.array(bg)
        bg = torch.FloatTensor(bg).squeeze(0)  # [256]

        label = record.label
        return label, v, wav2clip, t, bg, index



if __name__ == '__main__':
    pass
    # for mod in ['train', 'val', 'test']:
    #     getlabel(mod)