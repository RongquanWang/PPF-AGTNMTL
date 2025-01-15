r"""
使用clip提取视频帧和1608个人格描述符的特征并计算相似度得到1维联合特征，之后将送去使用clip_model训练并提取新特征
Use clip to extract features from video frames and 1608 personality descriptors,
calculate similarity to obtain 1D joint features, and then send them to be trained using clip_model to extract new features
"""
import pickle
import torch
import clip
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import glob

def get_textlabel(f_path):
    textlabel=[]
    for line in open(f_path,'r'):
        line=line.strip()
        textlabel.append(line)
    return textlabel

def extract_bf_feature():
    textlabel = get_textlabel('./1608_descriptors.txt')
    # print(len(textlabel))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)  # ViT-L/14@336px,ViT-B/32
    text = clip.tokenize(textlabel).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    for mod in ['val', 'train', 'test']:
    # for mod in ['test']:
        train_path = "/mnt/sdb/UDIVA_v0.5/UDIVA_v0.5/frames1/{}_frames".format(mod) # Folder containing extracted 15 frames of images
        save_path = './clip_{}_feature_emb_1608_UDIVA.pkl'.format(mod)
        vp_list = glob.glob(f'{train_path}/**/*')
        print(len(vp_list))
        dic = {}
        for filepath in tqdm(list(vp_list)):
            record = filepath.split('/')[-2]
            mp4 = filepath.split('/')[-1]
            if mod == 'test' and 'T' in mp4 and 'unmasked' not in mp4:
                continue
            img_list = [os.path.join(filepath, f'{record}-{mp4}-{i}.jpg') for i in range(0, 15)]
            # img_list = [train_path + num + '/face_xz/{}.jpg'.format(i) for i in range(1, 16)]
            name = f'{record}_{mp4[2]}_{mp4[4]}'

            seqs = []
            for image_path in img_list:
                if os.path.isfile(image_path):
                    # flag = 1
                    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                    # print(image.shape)
                    # exit()
                    with torch.no_grad():
                        image_features = model.encode_image(image)  # [1,768]
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        similarity = image_features @ text_features.T  #[1,1608]

                        if name not in dic.keys():
                            dic[name] = []
                        dic[name].append(similarity.cpu().numpy().tolist())
            dic[name] = np.array(dic[name]).mean(axis=0)
        with open(save_path, 'wb') as f:
            pickle.dump(dic, f)

if __name__=="__main__":
    pass
    extract_bf_feature()














