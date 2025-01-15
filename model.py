<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseModel import *


class Multimodal(nn.Module):
    def __init__(self, dim=128, head=1, metric='ones', p=0.2, text_len=13):
        super().__init__()
        video_neighbor = 3
        audio_neighbor = 3
        act = nn.ReLU()
        text_len = text_len
        self.rnn_text = RNN(input_size=768, hidden_size=64, num_layers=2, drop=0.2)

        self.att_text = ATT(128, text_len)

        self.lin_clip_clip = nn.Sequential(nn.Dropout(p), nn.Linear(256, dim), act)

        self.video_position = PositionalEncoding(768, 15)
        self.video_gcn = nn.Sequential(GCN(768, video_neighbor, metric=metric), GCN(768, video_neighbor, metric=metric))
        self.video_lin = nn.Sequential(nn.Dropout(p), nn.Linear(768 * 1, dim), act)

        self.wav2clip_position = PositionalEncoding(512, 15)
        self.wav2clip_gcn = nn.Sequential(GCN(512, audio_neighbor, metric=metric),
                                          GCN(512, audio_neighbor, metric=metric))
        self.wav2clip_lin = nn.Sequential(nn.Dropout(p), nn.Linear(512, dim), act)

        self.pooling = nn.AdaptiveMaxPool1d(1)

        # 融合模型
        num = 3
        self.mutliHead = channelATT2_res(dim=dim * num, head=head)

        # 主回归头
        self.lin_m = nn.Sequential(nn.Linear(dim * num * head, dim), nn.ReLU(), nn.Dropout(p))
        self.mlp_m = nn.Sequential(nn.Linear(dim, 5), nn.Sigmoid())

        self.clip_v_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        self.clip_clip_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        self.clip_wav_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        self.clip_t_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())

    def forward(self, clip_video, wav2clip, text, bg):

        clip_clip = self.lin_clip_clip(bg)

        vpos = self.video_position(clip_video)
        clip_v = clip_video + vpos
        clip_v = self.video_gcn(clip_v)
        clip_v = self.video_lin(clip_v)
        clip_v = self.pooling(clip_v.permute(0, 2, 1)).squeeze(2)

        a2pos = self.wav2clip_position(wav2clip)
        wav = wav2clip + a2pos
        wav = self.wav2clip_gcn(wav)
        wav = self.wav2clip_lin(wav)
        wav = self.pooling(wav.permute(0, 2, 1)).squeeze(2)

        clip_wav = cosine_func(clip_v, wav)

        clip_t = self.rnn_text(text)
        clip_t = self.att_text(clip_t)
        x = self.mutliHead(clip_clip, clip_wav, clip_t)
        xm = self.lin_m(x)

        out = self.mlp_m(xm)
        clip_clip_out = self.clip_clip_out(clip_clip)
        clip_wav_out = self.clip_wav_out(clip_wav)
        clip_t_out = self.clip_t_out(clip_t)

        res = {
            'm': out,
            'Feature_m': xm,

            'clip_clip': clip_clip_out,
            'Feature_clip_clip': clip_clip,

            'clip_wav': clip_wav_out,
            'Feature_clip_wav': clip_wav,

            'clip_t': clip_t_out,
            'Feature_clip_t': clip_t,
        }

        return res

class Multimodal_ablation(nn.Module):
    def __init__(self, dim=128, head=1, metric='ones', p=0.2, text_len=13):
        super().__init__()
        video_neighbor = 3
        audio_neighbor = 3
        act = nn.ReLU()
        self.rnn_text = nn.LSTM(input_size=768, hidden_size=64, num_layers=2, batch_first=True,
                               bidirectional=True, dropout=p)

        self.att_text = ATT(128, text_len)

        self.lin_clip_clip = nn.Sequential(nn.Dropout(p), nn.Linear(256, dim), act)

        self.video_position = PositionalEncoding(768, 15)
        self.video_gcn = nn.Sequential(GCN(768, video_neighbor, metric=metric), GCN(768, video_neighbor, metric=metric))
        self.video_lin = nn.Sequential(nn.Dropout(p), nn.Linear(768 * 1, dim), act)

        self.wav2clip_position = PositionalEncoding(512, 15)
        self.wav2clip_gcn = nn.Sequential(GCN(512, audio_neighbor, metric=metric),
                                          GCN(512, audio_neighbor, metric=metric))
        self.wav2clip_lin = nn.Sequential(nn.Dropout(p), nn.Linear(512, dim), act)

        self.pooling = nn.AdaptiveMaxPool1d(1)

        # 融合模型
        num = 3
        self.mutliHead = channelATT2_res(dim=dim * num, head=head)

        # 主回归头
        self.lin_m = nn.Sequential(nn.Linear(dim * num * head, dim), nn.ReLU(), nn.Dropout(p))
        self.mlp_m = nn.Sequential(nn.Linear(dim, 5), nn.Sigmoid())

        self.clip_v_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        self.clip_clip_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        self.clip_wav_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        self.clip_t_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())

    def forward(self, clip_video, wav2clip, text, bg):
        clip_clip = self.lin_clip_clip(bg)

        vpos = self.video_position(clip_video)
        clip_v = clip_video + vpos
        clip_v = self.video_gcn(clip_v)
        clip_v = self.video_lin(clip_v)
        clip_v = self.pooling(clip_v.permute(0, 2, 1)).squeeze(2)

        a2pos = self.wav2clip_position(wav2clip)
        wav = wav2clip + a2pos
        wav = self.wav2clip_gcn(wav)
        wav = self.wav2clip_lin(wav)
        wav = self.pooling(wav.permute(0, 2, 1)).squeeze(2)

        clip_wav = cosine_func(clip_v, wav)

        clip_t, _ = self.rnn_text(text)
        clip_t = self.att_text(clip_t)
        x = self.mutliHead(clip_clip, clip_wav, clip_t)
        xm = self.lin_m(x)

        out = self.mlp_m(xm)
        clip_clip_out = self.clip_clip_out(clip_clip)
        clip_wav_out = self.clip_wav_out(clip_wav)
        clip_t_out = self.clip_t_out(clip_t)

        res = {
            'm': out,
            'Feature_m': xm,

            'clip_clip': clip_clip_out,
            'Feature_clip_clip': clip_clip,

            'clip_wav': clip_wav_out,
            'Feature_clip_wav': clip_wav,

            'clip_t': clip_t_out,
            'Feature_clip_t': clip_t,
        }

        return res


class Multimodal_ablation2(nn.Module):
    def __init__(self, dim=128, head=1, metric='ones', p=0.2, text_len=13):
        super().__init__()
        video_neighbor = 3
        audio_neighbor = 3
        act = nn.ReLU()
        text_len = text_len
        self.rnn_text = nn.GRU(input_size=768, hidden_size=64, num_layers=2, batch_first=True,
                               bidirectional=True, dropout=p)

        self.att_text = ATT(128, text_len)

        self.lin_clip_clip = nn.Sequential(nn.Dropout(p), nn.Linear(256, dim), act)

        self.video_gcn = nn.GRU(input_size=768, hidden_size=768, num_layers=2, batch_first=True,
                                bidirectional=False, dropout=p)
        self.video_lin = nn.Sequential(nn.Dropout(p), nn.Linear(768, dim), act)

        self.wav2clip_gcn = nn.GRU(input_size=512, hidden_size=512, num_layers=2, batch_first=True,
                                   bidirectional=False, dropout=p)
        self.wav2clip_lin = nn.Sequential(nn.Dropout(p), nn.Linear(512, dim), act)

        self.pooling = nn.AdaptiveMaxPool1d(1)

        # 融合模型
        num = 3
        self.mutliHead = channelATT2_res(dim=dim * num, head=head)

        # 主回归头
        self.lin_m = nn.Sequential(nn.Linear(dim * num * head, dim), nn.ReLU(), nn.Dropout(p))
        self.mlp_m = nn.Sequential(nn.Linear(dim, 5), nn.Sigmoid())

        self.clip_v_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        self.clip_clip_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        self.clip_wav_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        self.clip_t_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())

    def forward(self, clip_video, wav2clip, text, bg):
        clip_clip = self.lin_clip_clip(bg)

        clip_v, _ = self.video_gcn(clip_video)
        clip_v = self.video_lin(clip_v)
        clip_v = self.pooling(clip_v.permute(0, 2, 1)).squeeze(2)

        wav, _ = self.wav2clip_gcn(wav2clip)
        wav = self.wav2clip_lin(wav)
        wav = self.pooling(wav.permute(0, 2, 1)).squeeze(2)

        clip_wav = cosine_func(clip_v, wav)

        clip_t, _ = self.rnn_text(text)
        clip_t = self.att_text(clip_t)
        x = self.mutliHead(clip_clip, clip_wav, clip_t)
        xm = self.lin_m(x)

        out = self.mlp_m(xm)
        clip_clip_out = self.clip_clip_out(clip_clip)
        clip_wav_out = self.clip_wav_out(clip_wav)
        clip_t_out = self.clip_t_out(clip_t)

        res = {
            'm': out,
            'Feature_m': xm,

            'clip_clip': clip_clip_out,
            'Feature_clip_clip': clip_clip,

            'clip_wav': clip_wav_out,
            'Feature_clip_wav': clip_wav,

            'clip_t': clip_t_out,
            'Feature_clip_t': clip_t,
        }

        return res


if __name__ == '__main__':
    pass


=======
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseModel import *


class Multimodal(nn.Module):
    def __init__(self, dim=128, head=1, metric='ones', p=0.2, text_len=13):
        super().__init__()
        video_neighbor = 3
        audio_neighbor = 3
        act = nn.ReLU()
        text_len = text_len
        self.rnn_text = RNN(input_size=768, hidden_size=64, num_layers=2, drop=0.2)

        self.att_text = ATT(128, text_len)

        self.lin_clip_clip = nn.Sequential(nn.Dropout(p), nn.Linear(256, dim), act)

        self.video_position = PositionalEncoding(768, 15)
        self.video_gcn = nn.Sequential(GCN(768, video_neighbor, metric=metric), GCN(768, video_neighbor, metric=metric))
        self.video_lin = nn.Sequential(nn.Dropout(p), nn.Linear(768 * 1, dim), act)

        self.wav2clip_position = PositionalEncoding(512, 15)
        self.wav2clip_gcn = nn.Sequential(GCN(512, audio_neighbor, metric=metric),
                                          GCN(512, audio_neighbor, metric=metric))
        self.wav2clip_lin = nn.Sequential(nn.Dropout(p), nn.Linear(512, dim), act)

        self.pooling = nn.AdaptiveMaxPool1d(1)

        # 融合模型
        num = 3
        self.mutliHead = channelATT2_res(dim=dim * num, head=head)

        # 主回归头
        self.lin_m = nn.Sequential(nn.Linear(dim * num * head, dim), nn.ReLU(), nn.Dropout(p))
        self.mlp_m = nn.Sequential(nn.Linear(dim, 5), nn.Sigmoid())

        self.clip_v_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        self.clip_clip_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        self.clip_wav_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        self.clip_t_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())

    def forward(self, clip_video, wav2clip, text, bg):

        clip_clip = self.lin_clip_clip(bg)

        vpos = self.video_position(clip_video)
        clip_v = clip_video + vpos
        clip_v = self.video_gcn(clip_v)
        clip_v = self.video_lin(clip_v)
        clip_v = self.pooling(clip_v.permute(0, 2, 1)).squeeze(2)

        a2pos = self.wav2clip_position(wav2clip)
        wav = wav2clip + a2pos
        wav = self.wav2clip_gcn(wav)
        wav = self.wav2clip_lin(wav)
        wav = self.pooling(wav.permute(0, 2, 1)).squeeze(2)

        clip_wav = cosine_func(clip_v, wav)

        clip_t = self.rnn_text(text)
        clip_t = self.att_text(clip_t)
        x = self.mutliHead(clip_clip, clip_wav, clip_t)
        xm = self.lin_m(x)

        out = self.mlp_m(xm)
        clip_clip_out = self.clip_clip_out(clip_clip)
        clip_wav_out = self.clip_wav_out(clip_wav)
        clip_t_out = self.clip_t_out(clip_t)

        res = {
            'm': out,
            'Feature_m': xm,

            'clip_clip': clip_clip_out,
            'Feature_clip_clip': clip_clip,

            'clip_wav': clip_wav_out,
            'Feature_clip_wav': clip_wav,

            'clip_t': clip_t_out,
            'Feature_clip_t': clip_t,
        }

        return res

class Multimodal_ablation(nn.Module):
    def __init__(self, dim=128, head=1, metric='ones', p=0.2, text_len=13):
        super().__init__()
        video_neighbor = 3
        audio_neighbor = 3
        act = nn.ReLU()
        self.rnn_text = nn.LSTM(input_size=768, hidden_size=64, num_layers=2, batch_first=True,
                               bidirectional=True, dropout=p)

        self.att_text = ATT(128, text_len)

        self.lin_clip_clip = nn.Sequential(nn.Dropout(p), nn.Linear(256, dim), act)

        self.video_position = PositionalEncoding(768, 15)
        self.video_gcn = nn.Sequential(GCN(768, video_neighbor, metric=metric), GCN(768, video_neighbor, metric=metric))
        self.video_lin = nn.Sequential(nn.Dropout(p), nn.Linear(768 * 1, dim), act)

        self.wav2clip_position = PositionalEncoding(512, 15)
        self.wav2clip_gcn = nn.Sequential(GCN(512, audio_neighbor, metric=metric),
                                          GCN(512, audio_neighbor, metric=metric))
        self.wav2clip_lin = nn.Sequential(nn.Dropout(p), nn.Linear(512, dim), act)

        self.pooling = nn.AdaptiveMaxPool1d(1)

        # 融合模型
        num = 3
        self.mutliHead = channelATT2_res(dim=dim * num, head=head)

        # 主回归头
        self.lin_m = nn.Sequential(nn.Linear(dim * num * head, dim), nn.ReLU(), nn.Dropout(p))
        self.mlp_m = nn.Sequential(nn.Linear(dim, 5), nn.Sigmoid())

        self.clip_v_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        self.clip_clip_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        self.clip_wav_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        self.clip_t_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())

    def forward(self, clip_video, wav2clip, text, bg):
        clip_clip = self.lin_clip_clip(bg)

        vpos = self.video_position(clip_video)
        clip_v = clip_video + vpos
        clip_v = self.video_gcn(clip_v)
        clip_v = self.video_lin(clip_v)
        clip_v = self.pooling(clip_v.permute(0, 2, 1)).squeeze(2)

        a2pos = self.wav2clip_position(wav2clip)
        wav = wav2clip + a2pos
        wav = self.wav2clip_gcn(wav)
        wav = self.wav2clip_lin(wav)
        wav = self.pooling(wav.permute(0, 2, 1)).squeeze(2)

        clip_wav = cosine_func(clip_v, wav)

        clip_t, _ = self.rnn_text(text)
        clip_t = self.att_text(clip_t)
        x = self.mutliHead(clip_clip, clip_wav, clip_t)
        xm = self.lin_m(x)

        out = self.mlp_m(xm)
        clip_clip_out = self.clip_clip_out(clip_clip)
        clip_wav_out = self.clip_wav_out(clip_wav)
        clip_t_out = self.clip_t_out(clip_t)

        res = {
            'm': out,
            'Feature_m': xm,

            'clip_clip': clip_clip_out,
            'Feature_clip_clip': clip_clip,

            'clip_wav': clip_wav_out,
            'Feature_clip_wav': clip_wav,

            'clip_t': clip_t_out,
            'Feature_clip_t': clip_t,
        }

        return res


class Multimodal_ablation2(nn.Module):
    def __init__(self, dim=128, head=1, metric='ones', p=0.2, text_len=13):
        super().__init__()
        video_neighbor = 3
        audio_neighbor = 3
        act = nn.ReLU()
        text_len = text_len
        self.rnn_text = nn.GRU(input_size=768, hidden_size=64, num_layers=2, batch_first=True,
                               bidirectional=True, dropout=p)

        self.att_text = ATT(128, text_len)

        self.lin_clip_clip = nn.Sequential(nn.Dropout(p), nn.Linear(256, dim), act)

        self.video_gcn = nn.GRU(input_size=768, hidden_size=768, num_layers=2, batch_first=True,
                                bidirectional=False, dropout=p)
        self.video_lin = nn.Sequential(nn.Dropout(p), nn.Linear(768, dim), act)

        self.wav2clip_gcn = nn.GRU(input_size=512, hidden_size=512, num_layers=2, batch_first=True,
                                   bidirectional=False, dropout=p)
        self.wav2clip_lin = nn.Sequential(nn.Dropout(p), nn.Linear(512, dim), act)

        self.pooling = nn.AdaptiveMaxPool1d(1)

        # 融合模型
        num = 3
        self.mutliHead = channelATT2_res(dim=dim * num, head=head)

        # 主回归头
        self.lin_m = nn.Sequential(nn.Linear(dim * num * head, dim), nn.ReLU(), nn.Dropout(p))
        self.mlp_m = nn.Sequential(nn.Linear(dim, 5), nn.Sigmoid())

        self.clip_v_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        self.clip_clip_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        self.clip_wav_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        self.clip_t_out = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())

    def forward(self, clip_video, wav2clip, text, bg):
        clip_clip = self.lin_clip_clip(bg)

        clip_v, _ = self.video_gcn(clip_video)
        clip_v = self.video_lin(clip_v)
        clip_v = self.pooling(clip_v.permute(0, 2, 1)).squeeze(2)

        wav, _ = self.wav2clip_gcn(wav2clip)
        wav = self.wav2clip_lin(wav)
        wav = self.pooling(wav.permute(0, 2, 1)).squeeze(2)

        clip_wav = cosine_func(clip_v, wav)

        clip_t, _ = self.rnn_text(text)
        clip_t = self.att_text(clip_t)
        x = self.mutliHead(clip_clip, clip_wav, clip_t)
        xm = self.lin_m(x)

        out = self.mlp_m(xm)
        clip_clip_out = self.clip_clip_out(clip_clip)
        clip_wav_out = self.clip_wav_out(clip_wav)
        clip_t_out = self.clip_t_out(clip_t)

        res = {
            'm': out,
            'Feature_m': xm,

            'clip_clip': clip_clip_out,
            'Feature_clip_clip': clip_clip,

            'clip_wav': clip_wav_out,
            'Feature_clip_wav': clip_wav,

            'clip_t': clip_t_out,
            'Feature_clip_t': clip_t,
        }

        return res


if __name__ == '__main__':
    pass


>>>>>>> origin/main
