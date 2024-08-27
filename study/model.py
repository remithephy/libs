import torch.nn as nn
import torch

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, stride=2),

                                    nn.Conv2d(32, 64, 5, stride=1, padding=2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, stride=2),

                                    nn.Conv2d(64, 128, 3, stride=1, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, stride=2))
        self.fc1 = nn.Sequential(nn.Linear(8192, 4096),
                                    nn.ReLU(),
                                    nn.Dropout(0.01),
                                    nn.Linear(4096, 1024),
                                    nn.ReLU(),
                                    nn.Dropout(0.01),
                                    nn.Linear(1024, 1)) 
    def forward(self, y):
        x = y.reshape(-1, 1, 64, 64).clone()
        x = self.layer1(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        return x
    
class CNN2(torch.nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        # 定义卷积层
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        
        # 计算卷积层输出的尺寸 (假设输入是 1x3600x56)
        # 输入尺寸通过每层卷积和池化操作进行变换
        # Conv1: 输入 1x3600x56 -> 输出 32x1800x28 (经过 MaxPool2d)
        # Conv2: 输入 32x1800x28 -> 输出 64x900x14 (经过 MaxPool2d)
        # Conv3: 输入 64x900x14 -> 输出 128x450x7 (经过 MaxPool2d)
        conv_output_size = 128 * 450 * 7
        
        # 定义全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(conv_output_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )

    def forward(self, y):
        # 将输入 reshape 为 1 个通道的 2D 图像 (假设输入为 1x3600x56)
        x = y.reshape(-1, 1, 3600, 56).clone()
        x = self.layer1(x)
        # 将卷积层的输出展平为一维向量
        x = x.view(x.size(0), -1)
        # 通过全连接层
        x = self.fc1(x)
        return x
    

class BPNN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1,n_hidden2, n_output):
        super(BPNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_feature,n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1,n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2,n_output))            

    def forward(self, x):
        out = self.fc(x)
        return out
    
class perturbation(torch.nn.Module):
    def __init__(self,n_feat):
        super(perturbation, self).__init__()
        self.sigma = nn.Parameter(torch.ones(n_feat)/2)
    def forward(self, x):
        ave = torch.mean(x)
        sigma = torch.sigmoid(self.sigma.clone())
        p_p = (sigma * x + (1 - sigma) * ave).clone()
        p_m = (sigma * ave + (1 - sigma) * x).clone()

        return p_p, p_m, sigma
