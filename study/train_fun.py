'''
:@Author: 张朝
:@Date: Created on Wed Aug 18 15:59:47 2021
:@LastEditors: Remi
:@LastEditTime: 2023/12/17 13:51:45
:Description:
'''
from sklearn.metrics import r2_score
import torch
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils import data
import random
import data_prepare as dpp
import torch.nn.functional as f
from torch.utils.data import Dataset
from tqdm import tqdm
from tensorboardX import SummaryWriter
#from autogluon.tabular import TabularDataset, TabularPredictor


###############################自动回归器#################################################
def auto_reg(train_data, test_data, label):
    train_data = TabularDataset(train_data)
    test_data = TabularDataset(test_data)
    reg = TabularPredictor(label=label).fit(train_data)
    test_lab = test_data[label]
    test_data_nolab = test_data.drop(label)
    test_pred_lab = reg.predict(test_data_nolab)
    perf = reg.evaluate_predictions(y_true=test_lab, y_pred=test_pred_lab, auxiliary_metrics=True)

    return 0


###############################coral损失函数##############################################
class loss_coral(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    @staticmethod
    def forward(source, target):
        d = source.data.shape[1]
        ns, nt = source.data.shape[0], target.data.shape[0]
        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (ns - 1)

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (nt - 1)

        # frostbitten norm between source and target
        loss = torch.mul((xc - xct), (xc - xct))
        loss = torch.sum(loss) / (4 * d * d)
        return loss


###############################空模型#################################################
class empty_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential()

    def forward(self, x):
        x = self.layer1(x)
        return x


###############################判别器#################################################
class spec_disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 384, 3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(384, 384, 3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(384, 128, 3, stride=1),
                                    )
        self.layer2 = nn.Sequential(nn.Linear(1152, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(512, 1),
                                    nn.Sigmoid())
        self.layer3 = nn.Sequential(nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 1),
                                    nn.Sigmoid())

    def forward(self, x):
        # x = self.layer1(x)
        # x = x.view(-1, 1152)
        # x = self.layer2(x)
        y = self.layer3(x)
        return y


###############################GAN模型搭建#################################################
class gene(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 256)
        self.fc2 = nn.Linear(256, 784)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.tanh(self.fc2(x))
        return x


class disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.sigmoid(self.fc2(x))
        return x


###############################CNN模型#################################################
class cnn_model(torch.nn.Module):
    def __init__(self):
        super(cnn_model, self).__init__()
        # 模型的结构定义
        self.layer1 = nn.Sequential(nn.Conv2d(1, 96, 11, stride=4, padding=0),
                                    nn.ReLU(),
                                    nn.MaxPool2d(3, stride=2),
                                    nn.Conv2d(96, 256, 5, stride=1, padding=2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(3, stride=2),
                                    nn.Conv2d(256, 384, 3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(384, 384, 3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(384, 256, 3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(3, stride=2))
        self.layer2 = nn.Sequential(nn.Linear(256 * 5 * 5, 4096),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),#在前向传播的时候，让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征。
                                    #比如有1000个神经元，p=0.4，训练的时候这一层神经元经过Dropout后，1000个神经元中会有大约400个的值被置为0。
                                    nn.Linear(4096, 1024),
                                    #nn.Sigmoid(),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(1024, 6))


    def forward(self, x):
        # 搭建模型
        x = self.layer1(x)
        #print(f"cnn{x.shape}")
        x = x.flatten(1)
        #print(f"cnn{x.shape}")
        x = self.layer2(x)
        #print(f"cnn{x.shape}")
        return x
    


###############################MRE损失函数##############################################
class loss_mre(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(loss_mre, self).__init__()

    @staticmethod
    def forward(y, y_hat):
        result = (y - y_hat) / y_hat
        return result.mean()


###############################BPNN模型#################################################
class BPNN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(BPNN, self).__init__()
        self.fc1 = nn.Linear(n_feature,n_hidden)#左边是特征数量
        self.fc2 = nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x


###############################小型模型训练#################################################
def small_model_train(net, train_data, train_features, train_labels, test_features, test_labels, device, log):
    ###################优化算法和参数设置#################################################
    loss = nn.MSELoss()
    lr = 0.005
    #optimizer = torch.optim.SGD(net.parameters(), lr=lr)#加了L2正则防过拟合
    #optimizer_Adamax = torch.optim.Adamax(net.parameters(), lr=lr)#adagrad学习率衰减，要把学习率先调大点
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #optimizer_AdamW = torch.optim.AdamW(net.parameters(), lr=lr)
    #optimizer_Asgd = torch.optim.ASGD(net.parameters(), lr=lr)
    #optimizer_Adadelta = torch.optim.Adadelta(net.parameters(), lr=lr)
    #optimizer = torch.optim.NAdam(net.parameters(), lr=lr, betas=(0.9,0.99), eps=1e-08, weight_decay=1e-12,momentum_decay=0.12)

    num_epochs = 5000
    step_epoch = 10
    ###################模型输出类型和中间变量#################################################
    all_net = []
    mse_loss = []
    train_loss = []
    test_loss = []
    train_r2 = []
    test_r2 = []
    train_pred = []
    test_pred = []
    train_rmse = []
    test_rmse = []
    train_mae = []
    test_mae = []
    net = net.to(device)
    ##################模型训练###################################################
    for epoch in range(num_epochs):
        loop = tqdm(train_data, total=len(train_data))
        for x, y in loop:
            l = loss(net(x), y).to(device)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            loop.set_description('epoch: [%d/%d]' % (epoch, num_epochs))
        if epoch % step_epoch == 0:
            net.eval()
            with torch.no_grad():
                train_pred = net(train_features)
                test_pred = net(test_features)
                train_loss.append((torch.abs((train_pred - train_labels)) / train_labels).mean())
                test_loss.append((torch.abs((test_pred - test_labels)) / test_labels).mean())
                mse_loss.append(loss(train_pred, train_labels))
                train_rmse.append(np.sqrt(((train_pred - train_labels) ** 2).mean()))
                test_rmse.append(np.sqrt(((test_pred - test_labels) ** 2).mean()))
                train_mae.append(np.abs((train_pred - train_labels)/(6 * train_labels)).mean())        ####################################需要改
                test_mae.append(np.abs((test_pred - test_labels)/(6 * test_labels)).mean())
                all_net.append(net)
                net.train()
            # 省略其他代码
        # if test_loss[i]<0.06 and i>1500:
        #    break
    ################确定最佳的epoch###########################################################
    best_epoch = dpp.optim_epoch(test_loss, num_epochs, step_epoch)
    print('best_epoch=%d' % best_epoch)
    best_net = all_net[best_epoch]
    #################绘图#####################################################################
    x = np.linspace(min(train_labels.numpy()), max(train_labels.numpy()), 2)
    y = x
    best_net.eval()
    with torch.no_grad():
        ##ctrl kc  ku 
        plt.figure(1)
        plt.scatter(train_labels, best_net(train_features), label="train set")
        plt.scatter(test_labels, best_net(test_features), label="test set")
        plt.plot(y, x)
        plt.grid()
        plt.xlabel("Reference", fontsize=16)
        plt.ylabel("Predict", fontsize=16)
        plt.legend(fontsize=16)
        plt.show()
        np.savetxt('/train.csv',np.hstack((train_labels,best_net(train_features))),delimiter=',',fmt='%.04f')
        np.savetxt('/test.csv',np.hstack((test_labels,best_net(test_features))),delimiter=',',fmt='%.04f')

        plt.figure(2)
        plt.plot(train_loss, label='train_loss')
        plt.plot(test_loss, label='test_loss')
        plt.grid()
        plt.xlabel("epoch", fontsize=16)
        plt.ylabel("mean relative error", fontsize=16)
        plt.legend(fontsize=16)
        plt.show()

        plt.figure(3)
        plt.plot(mse_loss, label='mse_loss')
        plt.grid()
        plt.xlabel("epoch", fontsize=16)
        plt.ylabel("mean square error", fontsize=16)
        plt.legend(fontsize=16)
        plt.show()

        train_r2.append(r2_score(train_labels, train_pred))
        test_r2.append(r2_score(test_labels, test_pred))

    print("Train R2 scores:", train_r2)
    print("Test R2 scores:", test_r2)
    print("Train RMSE scores:", np.max(train_rmse),np.min(train_rmse))
    print("Test RMSE scores:",  np.max(test_rmse),np.min(test_rmse))

    all_p = [train_r2,np.min(train_rmse),np.min(test_rmse),np.min(train_mae),np.min(test_mae)]
    return best_net,all_p


###############################大型模型训练#################################################
def  big_model_train(net, train_data, train_features, train_labels, test_features, test_labels, device, log):
    ###################优化算法和参数设置#################################################
    loss = nn.MSELoss()
    lr = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    num_epochs = 2000
    step_epoch = 10
    ###################模型输出类型和中间变量#################################################
    all_net = []
    mse_loss = []
    train_loss = []
    all_test_loss = []
    train_pre_lab = []
    test_pre_lab = []
    writer = SummaryWriter(log)
    ##################模型训练###################################################
    net.to(device)
    batch_size = 128
    test_iter = data.DataLoader(data.TensorDataset(test_features, test_labels), batch_size, shuffle=True)
    for epoch in range(num_epochs):
        loop = tqdm(train_data, total=len(train_data))#loop 是使用 tqdm 创建的一个进度条，用于可视化每个轮次内部的循环进度。
        for x, y in loop:
            x = x.to(device)
            y = y.to(device)
            l = loss(net(x), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            loop.set_description('epoch: [%d/%d]' % (epoch, num_epochs))
        if epoch % step_epoch == 0:
            net.eval()
            with torch.no_grad():
                sum_train_loss = 0
                sum_test_loss = 0
                sum_mse_loss = 0
                for x, y in train_data:
                    x = x.to(device)
                    y = y.to(device)
                    train_loss = (torch.abs((net(x) - y)) / y).mean()
                    sum_train_loss = sum_train_loss + train_loss
                    mse_loss = loss(net(x), y).mean()
                    sum_mse_loss = sum_mse_loss + mse_loss
                for x, y in test_iter:
                    x = x.to(device)
                    y = y.to(device)
                    test_loss = (torch.abs((net(x) - y)) / y).mean()
                    sum_test_loss = sum_test_loss + test_loss

                # train_loss.append((torch.abs((net(train_features) - train_labels)) / train_labels).mean())
                all_test_loss.append(sum_test_loss / len(test_iter))
                # mse_loss.append(loss(net(train_features), train_labels))
                # train_pre_lab.append(np.array(net(train_features).detach()).ravel())
                # test_pre_lab.append(np.array(net(test_features).detach()).ravel())
                all_net.append(net)
                # writer.add_scalar('loss/train_loss', torch.abs((net(train_features) - train_labels)) / train_labels).mean())
                writer.add_scalar('loss/test_loss', sum_test_loss / len(test_iter), epoch)
                writer.add_scalar('loss/train_loss', sum_train_loss / len(train_data), epoch)
                writer.add_scalar('loss/mse_loss', sum_mse_loss / len(train_data), epoch)

                net.train()
        # if test_loss[i]<0.06 and i>1500:
        #    break
    ################确定最佳的epoch###########################################################
    best_epoch = dpp.optim_epoch(all_test_loss, num_epochs, step_epoch)
    print('best_epoch=%d' % best_epoch)
    best_net = all_net[best_epoch]
    #################绘图#####################################################################
    '''
    x = np.linspace(min(train_labels.numpy()), max(train_labels.numpy()), 2)
    y = x
    best_net.to('cpu')
    best_net.eval()
    train_labels = train_labels.to('cpu')
    train_features = train_features.to('cpu')
    test_labels = test_labels.to('cpu')
    test_features = test_features.to('cpu')
    with torch.no_grad():
        plt.figure(1)
        plt.scatter(train_labels, best_net(train_features), label="train set")
        plt.scatter(test_labels, best_net(test_features), label="test set")
        plt.plot(y, x)
        plt.grid()
        plt.xlabel("Reference size.(um)", fontsize=16)
        plt.ylabel("Predict size.(um)", fontsize=16)
        plt.legend(fontsize=16)
        plt.show()
    '''
    # plt.savefig(r'model_final_test/curve_'+name+'_'+str(in_num)+'_'+str(hiden_num)+'.pdf', dpi=200)

    # plt.close('all')
    # print('模型的训练集误差为：%f' %(train_loss[-1]))
    # print('模型的测试集误差为：%f' %(test_loss[-1]))

    return best_net


###############################交叉验证#####################################
def cross_vali(net, feat, lab, num_fold, batch_size):
    num_spec = len(lab) // num_fold
    all_model_performance = []
    all_sd_mean = []
    for i in range(num_fold):
        #################划分训练集和测试集###########################################
        test_index = np.arange(i * num_spec, (i + 1) * num_spec)
        train_lab = np.delete(lab, test_index, axis=0)
        train_feat = np.delete(feat, test_index, axis=0)
        test_lab = lab[test_index, :]
        test_feat = feat[test_index, :]
        ################数据格式转换与模型训练##########################################
        train_feat = a2t(train_feat)
        train_lab_tensor = a2t(train_lab)
        test_feat = a2t(test_feat)
        test_lab_tensor = a2t(test_lab)
        train_data = load_array((train_feat, train_lab_tensor), batch_size)
        optim_model = small_model_train(net, train_data, train_feat,
                                        train_lab_tensor, test_feat, test_lab_tensor)
        ################模型性能评估#################################################
        train_pred_lab = optim_model(train_feat).detach().numpy()
        test_pred_lab = optim_model(test_feat).detach().numpy()
        slope = dpp.compute_slope(train_lab, train_pred_lab)
        r2 = dpp.compute_r2(train_lab, train_pred_lab)
        rec = dpp.compute_mre(train_lab, train_pred_lab)
        rep = dpp.compute_mre(test_lab, test_pred_lab)
        rsd = dpp.compute_rsd(test_pred_lab, 1)
        all_model_performance.append(np.array([slope, r2, rec, rep, rsd]))
        print('%d作为测试集时，训练集相对误差为%f，测试集相对误差为%f' % (test_lab[0] / 50.0, rec, rep))
        ################校准曲线绘制#################################################
        train_lab = train_lab / 50
        test_lab = test_lab / 50
        train_pred_lab = train_pred_lab / 50
        test_pred_lab = test_pred_lab / 50
        sd_mean = dpp.get_sd_mean(train_lab, train_pred_lab, test_lab, test_pred_lab, 9)
        all_sd_mean.append(sd_mean)
        fig = dpp.plot_errbar_curve(train_lab, train_pred_lab, test_lab, test_pred_lab, 9)
        plt.title('calibration curve ' + str(test_lab[0]))
        plt.savefig('cali_curve_' + str(test_lab[0]) + '.jpg', dpi=600)
    all_model_performance = np.array(all_model_performance)
    all_sd_mean = np.array(all_sd_mean)
    all_sd_mean = all_sd_mean.reshape(-1, 3)
    pd.DataFrame(all_sd_mean).to_csv('all_sd_mean.csv', header=None, index=False)
    return all_model_performance


###############################vgg-11训练函数#####################################
def train_fun_vgg11(train_data, train_features, train_labels, test_features, test_labels):
    epochs = 3000

    lr = 0.004
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    net = vgg(conv_arch)

    optimizer = torch.optim.Adam(net.parameters(), lr)
    loss = nn.MSELoss()
    loss_1 = []
    train_loss = []
    test_loss = []
    train_pre_lab = []
    test_pre_lab = []
    for i in range(epochs):
        net.to(device='cuda')
        for x, y in train_data:
            x = x.to(device='cuda')
            y = y.to(device='cuda')
            l = loss(net(x), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        print(l)
        net.to(device='cpu')

        train_loss.append(
            (torch.abs((net(train_features).detach() - train_labels)) / train_labels).mean())
        test_loss.append((torch.abs((net(test_features).detach() - test_labels)) / test_labels).mean())
        loss_1.append(loss(net(train_features), train_labels)).detach()
        train_pre_lab.append(net(train_features).detach())
        test_pre_lab.append(net(test_features).detach())
        # if test_loss[i]<0.03 and train_loss[i]<0.01:
        #    break
    loss_1 = np.array(loss_1)
    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    torch.save(net.state_dict(), 'vgg11.pth')
    x = np.linspace(0, 450, 450)
    y = x

    pd.DataFrame(loss_1).to_csv('loss.csv', header=None, index=False)
    pd.DataFrame(train_loss).to_csv('mre_train.csv', header=None, index=False)
    pd.DataFrame(test_loss).to_csv('mre_test.csv', header=None, index=False)
    pd.DataFrame(train_pre_lab).to_csv('train_pre_lab.csv', header=None, index=False)
    pd.DataFrame(test_pre_lab).to_csv('test_pre_lab.csv', header=None, index=False)
    '''
    plt.figure(1)
    plt.plot(train_loss,label="train set")
    plt.plot(test_loss,label="test set")
    plt.xlabel("epoch",fontsize=16)
    plt.ylabel("mean relative error",fontsize=16)
    plt.legend(fontsize=16)
    '''
    '''
    plt.figure(2)
    plt.plot(loss_1,label="loss_fuction")
    plt.xlabel("epoch",fontsize=16)
    plt.ylabel("MSE",fontsize=16)
    plt.legend(fontsize=16)
    '''
    '''
    plt.figure(3)
    plt.scatter(train_labels,net(train_features).detach(),label="train set")
    plt.scatter(test_labels,net(test_features).detach(),label="test set")
    plt.plot(y,x)
    plt.grid()
    plt.xlabel("Reference size.(um)",fontsize=16)
    plt.ylabel("Predict size.(um)",fontsize=16)
    plt.legend(fontsize=16)
    '''

    return net


###############################ANN的训练函数#################################################
def train_fun_ann(train_data, train_features, train_labels, test_features, test_labels, hiden_num, name):
    in_num = train_features.shape[1]
    out_num = 1
    net = nn.Sequential(nn.Linear(in_num, hiden_num),
                        nn.Tanh(),
                        # nn.Dropout(0.1),
                        nn.Linear(hiden_num, out_num),

                        # nn.Dropout(0.1),
                        )
    ###################优化算法和参数设置#################################################
    loss = nn.MSELoss()
    lr = 0.005
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    num_epochs = 3000
    ###################模型输出类型和中间变量#################################################
    all_net = []
    loss_1 = []
    train_loss = []
    test_loss = []
    train_pre_lab = []
    test_pre_lab = []
    ##################模型训练###################################################
    for i in range(num_epochs):
        for x, y in train_data:
            l = loss(net(x), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        if i % 20 == 0:
            train_loss.append((torch.abs((net(train_features) - train_labels)) /
                               train_labels).mean().detach())
            test_loss.append((torch.abs((net(test_features) - test_labels)) /
                              test_labels).mean().detach())
            loss_1.append(loss(net(train_features), train_labels).detach())
            train_pre_lab.append(np.array(net(train_features).detach()).ravel())
            test_pre_lab.append(np.array(net(test_features).detach()).ravel())
            all_net.append(net)
        # if test_loss[i]<0.06 and i>1500:
        #    break
    ################确定最佳的epoch###########################################################
    best_epoch = dpp.optim_epoch(test_loss, num_epochs)
    '''
################模型性能评估###########################################################
    model_performance=[]
    train_lab=np.array(train_labels).ravel()
    test_lab=np.array(test_labels).ravel()
    slope=dpp.compute_slope(train_lab, train_pre_lab[best_epoch])
    r2=dpp.compute_r2(train_lab,train_pre_lab[best_epoch])
    train_loss=np.array(train_loss)
    test_loss=np.array(test_loss)
    rec=train_loss[best_epoch]
    rep=test_loss[best_epoch]
    rsd=dpp.compute_rsd(test_pre_lab[best_epoch])
    model_performance.append(slope,r2,rec,rep,rsd)
    model_performance=np.array(model_performance)
    pd.DataFrame(model_preformance).to_csv('LIBS-data/model_performance.csv',header=None,index=False)
    '''
    #################绘图#####################################################################
    x = np.linspace(min(train_labels.numpy()), max(train_labels.numpy()), 2)
    y = x

    plt.figure(1)
    plt.plot(train_loss, label="train set")
    plt.plot(test_loss, label="test set")
    plt.xlabel("epoch", fontsize=16)
    plt.ylabel("mean relative error", fontsize=16)
    plt.grid()
    plt.legend(fontsize=16)
    # plt.savefig(r'model_final_test/lose_'+name+'_'+str(in_num)+'_'+str(hiden_num)+'.pdf', dpi=200)

    plt.figure(2)
    plt.plot(loss_1, label="loss_fuction")
    plt.xlabel("epoch", fontsize=16)
    plt.ylabel("MSE", fontsize=16)
    plt.legend(fontsize=16)

    plt.figure(3)
    plt.scatter(train_labels, net(train_features).detach(), label="train set")
    plt.scatter(test_labels, net(test_features).detach(), label="test set")
    plt.plot(y, x)
    plt.grid()
    plt.xlabel("Reference size.(um)", fontsize=16)
    plt.ylabel("Predict size.(um)", fontsize=16)
    plt.legend(fontsize=16)
    # plt.savefig(r'model_final_test/curve_'+name+'_'+str(in_num)+'_'+str(hiden_num)+'.pdf', dpi=200)

    # plt.close('all')
    # print('模型的训练集误差为：%f' %(train_loss[-1]))
    # print('模型的测试集误差为：%f' %(test_loss[-1]))

    '''
    pd.DataFrame(train_pre_lab).to_csv(r'model_final_test/train_pre_lab_'+name+'_'
                                       +str(in_num)+'_'+str(hiden_num)+'.csv'
                                       ,header=None,index=False)
    pd.DataFrame(test_pre_lab).to_csv(r'model_final_test/test_pre_lab_'+name+'_'
                                       +str(in_num)+'_'+str(hiden_num)+'.csv'
                                       ,header=None,index=False)

    pd.DataFrame(train_loss).to_csv(r'model_final_test/train_mre_'+name+'_'
                                       +str(in_num)+'_'+str(hiden_num)+'.csv'
                                       ,header=None,index=False)
    pd.DataFrame(test_loss).to_csv(r'model_final_test/test_mre_'+name+'_'
                                       +str(in_num)+'_'+str(hiden_num)+'.csv'
                                       ,header=None,index=False)

    pd.DataFrame(loss_1).to_csv(r'model_final_test/train_loss_'+name+'_'
                                       +str(in_num)+'_'+str(hiden_num)+'.csv'
                                       ,header=None,index=False)
    '''
    return all_net[best_epoch]


###############################ANN交叉验证的训练函数#################################################
def train_fun_ann_cv(train_data, train_features, train_labels, test_features, test_labels, hiden_num, name):
    in_num = train_features.shape[1]
    out_num = 1
    net = nn.Sequential(nn.Linear(in_num, hiden_num),
                        nn.Tanh(),
                        # nn.Dropout(0.1),
                        nn.Linear(hiden_num, out_num),

                        # nn.Dropout(0.1),
                        )

    loss = nn.MSELoss()
    lr = 0.0045
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    num_epochs = 5000
    loss_1 = []
    train_loss = []
    test_loss = []
    train_pre_lab = []
    test_pre_lab = []

    for i in range(num_epochs):
        for x, y in train_data:
            l = loss(net(x), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_loss.append((torch.abs((net(train_features) - train_labels)) /
                           train_labels).mean().detach())
        test_loss.append((torch.abs((net(test_features) - test_labels)) / test_labels).mean().detach())
        loss_1.append(loss(net(train_features), train_labels).detach())
        train_pre_lab.append(np.array(net(train_features).detach()).ravel())
        test_pre_lab.append(np.array(net(test_features).detach()).ravel())
        # if test_loss[i]<0.03 and train_loss[i]<0.01:
        #    break
    loss_1 = np.array(loss_1)
    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    x = np.linspace(0, 450, 450)
    y = x

    plt.figure(1)
    plt.plot(train_loss, label="train set")
    plt.plot(test_loss, label="test set")
    plt.xlabel("epoch", fontsize=16)
    plt.ylabel("mean relative error", fontsize=16)
    plt.grid()
    plt.legend(fontsize=16)
    plt.savefig(r'cv_fig/loss_' + str(test_labels[0, :].item()) + '_' + name +
                '_' + str(in_num) + '_' + str(hiden_num) + '.jpg', dpi=200)

    plt.figure(2)
    plt.plot(loss_1, label="loss_fuction")
    plt.xlabel("epoch", fontsize=16)
    plt.ylabel("MSE", fontsize=16)
    plt.legend(fontsize=16)

    plt.figure(3)
    plt.scatter(train_labels, net(train_features).detach(), label="train set")
    plt.scatter(test_labels, net(test_features).detach(), label="test set")
    plt.plot(y, x)
    plt.grid()
    plt.xlabel("Reference size.(um)", fontsize=16)
    plt.ylabel("Predict size.(um)", fontsize=16)
    plt.legend(fontsize=16)
    '''
    plt.savefig(r'cv_fig/curve_'+str(test_labels[0,:].item())+'_'+name+
                '_'+str(in_num)+'_'+str(hiden_num)+'.jpg', dpi=200)

    plt.close('all')
    '''
    print(train_loss[-1])
    print(test_loss[-1])
    '''
    pd.DataFrame(train_pre_lab).to_csv(r'ann_cv/train_pre_lab_mtinfo_'
                                       +str(test_labels[0,:].item())+'.csv'
                                       ,header=None,index=False)
    pd.DataFrame(test_pre_lab).to_csv(r'ann_cv/test_pre_lab_mtinfo_'
                                      +str(test_labels[0,:].item())+'.csv'
                                       ,header=None,index=False)

    pd.DataFrame(train_loss).to_csv(r'ann_cv/'+name+'_cv/train_mre'+'_'+name+'_'
                                    +str(test_labels[0,:].item())
                                    +'_'+str(in_num)+'_'+str(hiden_num)+'.csv'
                                       ,header=None,index=False)
    pd.DataFrame(test_loss).to_csv(r'ann_cv/'+name+'_cv/test_mre_'+name+'_'
                                   +str(test_labels[0,:].item())
                                   +'_'+str(in_num)+'_'+str(hiden_num)+'.csv'
                                       ,header=None,index=False)

    pd.DataFrame(loss_1).to_csv(r'ann_cv/train_loss_mtinfo_'
                                +str(test_labels[0,:].item())+'.csv'
                                       ,header=None,index=False)
    '''
    return net



##############################不分层按照7：3划分训练集和测试集#######################
def nolayer_divid(x, y):
    num_spec = x.shape[0]
    num_test = num_spec // 3
    index_delet = random.sample(range(num_spec), num_test)
    index_delet = np.array(index_delet).flatten()
    test_x = x[index_delet, :]
    test_y = y[index_delet, :]
    train_x = np.delete(x, index_delet, axis=0)
    train_y = np.delete(y, index_delet, axis=0)
    return train_x, train_y, test_x, test_y


##############################按照光谱划分数据集###############################
def stand_spec(x, y):
    num_sample = 10
    index_test = 6
    index_delet = []
    for i in range(num_sample):
        index_delet.append(i * num_sample + index_test + 0)
        index_delet.append(i * num_sample + index_test + 1)

    test_x = x[index_delet, :]
    test_y = y[index_delet, :]
    train_x = np.delete(x, index_delet, axis=0)
    train_y = np.delete(y, index_delet, axis=0)
    return train_x, train_y, test_x, test_y


##############################数组转张量##############################################
def a2t(arr):
    return torch.Tensor(arr)


##############################数据打包 train data###############################################
def load_array(features, batch_size, is_train=True):
    data_set = data.TensorDataset(*features)
    return data.DataLoader(data_set, batch_size, shuffle=is_train)


##############################vgg块#############################################
def vgg_block(num_covns, channls_in, channls_out):
    layers = []
    for i in range(num_covns):
        layers.append(nn.Conv2d(channls_in, channls_out, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        channls_in = channls_out
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


##############################vgg##############################################
def vgg(conv_arch):
    block_bulk = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        block_bulk.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(Reshape(), *block_bulk, nn.Flatten(),
                         nn.Linear(6 * 6 * out_channels, 4096), nn.ReLU(), nn.Linear(4096, 2048), nn.ReLU(), nn.Linear(2048, 1))


##############################reshape#############################################
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 200,
                      -200)
