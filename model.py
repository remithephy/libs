import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleTaskInceptionResNet(nn.Module):
    def __init__(self, num_outputs_task1=1, ):
        super(MultiTaskInceptionResNet, self).__init__()

        # Stem Layer with two Conv2D layers
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Single Inception-ResNet-A block
        self.inception_resnet_a = InceptionResNetA(64)

        self.dropout = nn.Dropout(0.5)

        # Task 1: Classification (dense-dropout-dense-output)
        self.fc_task1 = nn.Sequential(
            nn.Linear(64 * 32 * 32, 1024),  # Adjusted input size
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout layer
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs_task1)  # Final output
        )

        # # Regularization parameter
        # self.lambda_reg = lambda_reg

    def forward(self, x1):
        # Shared feature extraction
        x = x1.reshape(-1, 1, 64, 64).clone()
        x = self.stem(x)
        x = self.dropout(x)
        x = self.inception_resnet_a(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.dropout(x)
        # Task outputs
        task1_out = self.fc_task1(x)  # Regression1
    
        return task1_out
    
class MultiTaskInceptionResNet(nn.Module):
    def __init__(self, num_outputs_task1=1, num_outputs_task2=1, lambda_reg=0.01):
        super(MultiTaskInceptionResNet, self).__init__()

        # Stem Layer with two Conv2D layers
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Single Inception-ResNet-A block
        self.inception_resnet_a = InceptionResNetA(64)

        self.dropout = nn.Dropout(0.5)

        # Task 1: Classification (dense-dropout-dense-output)
        self.fc_task1 = nn.Sequential(
            nn.Linear(64 * 32 * 32, 1024),  # Adjusted input size
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout layer
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs_task1)  # Final output
        )
        
        # Task 2: Regression (dense-dropout-dense-output)
        self.fc_task2 = nn.Sequential(
            nn.Linear(64 * 32 * 32, 1024),  # Adjusted input size
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout layer
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs_task2)  # Final output
        )
        
        # # Regularization parameter
        # self.lambda_reg = lambda_reg

    def forward(self, x1):
        # Shared feature extraction
        x = x1.reshape(-1, 1, 64, 64).clone()
        x = self.stem(x)
        x = self.dropout(x)
        x = self.inception_resnet_a(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.dropout(x)
        # Task outputs
        task1_out = self.fc_task1(x)  # Regression1
        task2_out = self.fc_task2(x)  # Regression2
        
        return task1_out, task2_out
    
class DynamicWeightedLoss(nn.Module):

    def __init__(self):
        super(DynamicWeightedLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.prev_val_losses = None  # To store the validation losses from the previous epoch
        self.smoothing_factor = 0.95  # 平滑因子
        self.min_weight = 0.1  # 权重最小值
        self.max_weight = 10   # 权重最大值
        self.epsilon = 1e-4  # Small value to avoid division by zero

    def forward(self, input_Cs, input_Rb, Cs_lab, Rb_lab, val_losses):
        if self.prev_val_losses is None:
            # Initialize prev_val_losses with the current epoch's validation losses
            self.prev_val_losses = val_losses.clone().detach() 
        
        # 计算权重并应用平滑
        lambda_Cs = self.smoothing_factor * torch.log1p(torch.abs((val_losses[0] - self.prev_val_losses[0]) / (self.prev_val_losses[0]) + self.epsilon))
        lambda_Rb = self.smoothing_factor * torch.log1p(torch.abs((val_losses[1] - self.prev_val_losses[1]) / (self.prev_val_losses[1]) + self.epsilon))

        # 限制权重的范围
        lambda_Cs = torch.clamp(lambda_Cs, self.min_weight, self.max_weight)
        lambda_Rb = torch.clamp(lambda_Rb, self.min_weight, self.max_weight)
        lambda_Cs = lambda_Cs/(lambda_Rb + lambda_Cs)
        lambda_Rb = lambda_Rb/(lambda_Rb + lambda_Cs)
        # Compute individual task losses
        ls_c = self.loss(input_Cs, Cs_lab)
        ls_r = self.loss(input_Rb, Rb_lab)

        
        # 用当前历元的val_losses值更新self.prev-val_loss，以在下一个epoch进行比较
        self.prev_val_losses = val_losses.clone().detach()
        
        return (lambda_Rb * ls_c),(lambda_Cs * ls_r)
  
    
# Inception-ResNet-A module (block35)

class InceptionResNetA(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super(InceptionResNetA, self).__init__()
        
        # Branch 1
        self.branch1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        
        # Branch 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        
      
        # Concatenate branches
        self.conv2d = nn.Conv2d(64, in_channels, kernel_size=1)
        self.scale = scale
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
 
        
        out = torch.cat([branch1, branch2], 1)
        out = self.conv2d(out)
        
        return x + self.scale * out

    



# #Rb
# class Rb(nn.Module):
#     def __init__(self):
#         super(Rb, self).__init__()
#         self.layer2 = nn.Sequential(
#                                     nn.Linear(128 * 38, 1024),
#                                     nn.ReLU(),
#                                     nn.Dropout(0.5),#在前向传播的时候，让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征。
#                                     #比如有1000个神经元，p=0.4，训练的时候这一层神经元经过Dropout后，1000个神经元中会有大约400个的值被置为0。
#                                     nn.Linear(1024, 128),
#                                     #nn.Sigmoid(),
#                                     nn.ReLU(),
#                                     nn.Dropout(0.5),
#                                     nn.Linear(128, 1))


#     def forward(self, x):
#         # 搭建模型
#         x = x.flatten(1)
#         #print(f"cnn{x.shape}")
#         x = self.layer2(x)
#         #print(f"cnn{x.shape}")
#         return x

# #Cs
# class Cs(nn.Module):
#     def __init__(self):
#         super(Cs, self).__init__()
#         self.layer2 = nn.Sequential(
#                                     nn.Linear(128 * 38, 1024),
#                                     nn.ReLU(),
#                                     nn.Dropout(0.5),#在前向传播的时候，让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征。
#                                     #比如有1000个神经元，p=0.4，训练的时候这一层神经元经过Dropout后，1000个神经元中会有大约400个的值被置为0。
#                                     nn.Linear(1024, 128),
#                                     #nn.Sigmoid(),
#                                     nn.ReLU(),
#                                     nn.Dropout(0.5),
#                                     nn.Linear(128, 1))


#     def forward(self, x):
#         # 搭建模型
#         x = x.flatten(1)
#         x = self.layer2(x)

#         return x    


    
# 六通道
class Model_1D(nn.Module):
    def __init__(self):
        super(Model_1D, self).__init__()

        self.shared_convolution = SharedLayer_1D()
        self.regression_part1 = Cs()
        self.regression_part2 = Rb()

    def forward(self, x):
        
        shared_features = self.shared_convolution(x)

        Flatness_output = self.regression_part1(shared_features)
        Concen_output = self.regression_part2(shared_features)
 
        all_output = torch.cat([Concen_output, Flatness_output], dim=1)    
        
        return all_output
    
    
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
