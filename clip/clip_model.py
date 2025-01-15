<<<<<<< HEAD
r"""
使用clip提取视频帧和1608个人格描述符的特征并计算相似度得到1维联合特征后使用该模型训练，然后提取训练后的特征作为official/untils.py中的bg特征
Use clip to extract features from video frames and 1608 personality descriptors, calculate similarity to obtain 1D joint features,
and train the model. Then, extract the trained features as bg features in official/untimes.py
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_feature=1608):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_feature, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(256, 5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x:[b,1608]
        x = self.dropout(self.relu1(self.bn1(self.linear1(x))))
        x = self.dropout(self.relu2(self.bn2(self.linear2(x))))
        f = x.clone()
        x = self.sigmoid(self.linear3(x))

        return x, f    #[b,5]

if __name__=="__main__":
    x=torch.rand(2,1608)
    model=Model()
    with torch.no_grad():
        y,_=model(x)
        print(y.shape)
=======
r"""
使用clip提取视频帧和1608个人格描述符的特征并计算相似度得到1维联合特征后使用该模型训练，然后提取训练后的特征作为official/untils.py中的bg特征
Use clip to extract features from video frames and 1608 personality descriptors, calculate similarity to obtain 1D joint features,
and train the model. Then, extract the trained features as bg features in official/untimes.py
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_feature=1608):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_feature, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(256, 5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x:[b,1608]
        x = self.dropout(self.relu1(self.bn1(self.linear1(x))))
        x = self.dropout(self.relu2(self.bn2(self.linear2(x))))
        f = x.clone()
        x = self.sigmoid(self.linear3(x))

        return x, f    #[b,5]

if __name__=="__main__":
    x=torch.rand(2,1608)
    model=Model()
    with torch.no_grad():
        y,_=model(x)
        print(y.shape)
>>>>>>> origin/main
