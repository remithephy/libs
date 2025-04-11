import os
import torch
import data_prepare as dpp
from torch import nn
from tqdm import tqdm
import numpy as np
import pandas as pd
from model import perturbation,Predictor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
path = r'D:\20231017(仿真卤水)\for_ml\weighted\all.csv'
label_path = r'D:\20231017(仿真卤水)\for_ml\weighted\alllab.csv'
wl_path = r'D:\20231017(仿真卤水)\for_ml\weighted\wl.csv'
save_path = r'D:\20231017(仿真卤水)\for_ml\result'

raw_feat = pd.read_csv(path,header=None).values.astype(np.float32)[:,11500:12200]###only select wl range 700-800
raw_label = pd.read_csv(label_path,header=None).values.astype(np.float32)
raw_wl = pd.read_csv(wl_path,header=None).values.astype(np.float32)[11500:12200]

# 取测试集
test_index = np.isin(raw_label,raw_label[-1]).reshape(-1)
predict_data = raw_feat[test_index,:]
predict_label = raw_label[test_index]
raw_feat = raw_feat[~test_index,:]
raw_label = raw_label[~test_index]

# random seeding
random_seed = 2024  
np.random.seed(random_seed)  
random_index = [i for i in range(len(raw_label))]
np.random.shuffle(random_index)

raw_feat = raw_feat[random_index,:]
raw_label = raw_label[random_index,:]



# normalization
feat_scaler = MinMaxScaler()
lab_scaler = MinMaxScaler()
norm_feat = feat_scaler.fit_transform(raw_feat)
norm_lab = lab_scaler.fit_transform(raw_label)

# select the best features
scores = dpp.score(norm_feat, norm_lab)
num_feat = 20
index_feat = dpp.select_best(scores, num_feat)
norm_feat = norm_feat [:, index_feat]
select_wl = raw_wl[index_feat]



# split train and test
x_train,x_test,y_train,y_test = train_test_split(
    norm_feat,norm_lab, test_size=0.2,shuffle = False)

batch_size = 256

train_data = torch.tensor(x_train, dtype=torch.float32)
train_label = torch.tensor(y_train, dtype=torch.float32)
train_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_data, train_label), batch_size=batch_size, shuffle=True)

test_data = torch.tensor(x_test, dtype=torch.float32)
test_label = torch.tensor(y_test, dtype=torch.float32)
test_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_data, test_label), batch_size=batch_size, shuffle=False)
'''remember to set shuffle to False'''



# parameter define

if torch.cuda.is_available():
    device = 'cuda'
    print("Using GPU !")
else:
    device = 'cpu'

logdir = os.getcwd()

model = Predictor(num_feat,12,1)
perturb = perturbation(num_feat)
criterion = nn.MSELoss()


lr = 5e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
perturb_optimizer = torch.optim.Adam(perturb.parameters(), lr=lr)

num_epochs = 8000
step_epoch = 10

# variable define
all_sigma = []
all_net = []
mse_loss = []
all_mse_loss = []
train_loss = []
all_train_rmse = []
all_train_loss = []
all_test_rmse = []
all_test_loss = []
train_pre_lab = []
test_pre_lab = []



## 模型训练
model.to(device)

for epoch in range(num_epochs):
    loop = tqdm(train_dataloader, total=len(train_dataloader))#loop 是使用 tqdm 创建的一个进度条，用于可视化每个轮次内部的循环进度。
    
    for x, y in loop:
        x = x.to(device)
        y = y.to(device)
        preds = model(x)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        loop.set_description('epoch: [%d/%d]' % (epoch, num_epochs))
    
    if epoch % step_epoch == 0: # 每step_epoch次epoch存一次数据
        model.eval()
        with torch.no_grad():
            sum_mse_loss = 0
            sum_train_loss = 0
            sum_test_loss = 0
            sum_train_rmse = 0
            sum_test_rmse = 0
            
            for x, y in train_dataloader:
                x = x.to(device)
                y = y.to(device)
                preds = model(x)
                # 计算损失
                loss = torch.zeros_like(y)  # 创建一个与 y 相同形状的张量来存储损失
                abs_loss_mask = (y == 0.0)  # 布尔掩码，用于标记 y 中等于零的部分

                # 对等于零的部分使用绝对值损失
                loss[abs_loss_mask] = torch.abs(preds[abs_loss_mask] - y[abs_loss_mask])

                # 对不等于零的部分使用均方误差损失
                mse_loss_mask = ~abs_loss_mask  # 取反，用于标记 y 中不等于零的部分
                loss[mse_loss_mask] = torch.mean(torch.abs(preds[mse_loss_mask] - y[mse_loss_mask])) / y[mse_loss_mask]

                train_loss = loss.mean()

                sum_train_loss = sum_train_loss + train_loss
                mse_loss = criterion(preds, y).mean()
                sum_mse_loss = sum_mse_loss + mse_loss

                # 评估指标
                y_denormalized = lab_scaler.inverse_transform(y.cpu())
                preds_denormalized = lab_scaler.inverse_transform(preds.cpu())
                train_rmse = (np.sqrt(((preds_denormalized - y_denormalized) ** 2).mean())) # 数据存到cpu
                sum_train_rmse = sum_train_rmse + train_rmse

            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)
                preds = model(x)
                # 计算损失
                loss = torch.zeros_like(y)  # 创建一个与 y 相同形状的张量来存储损失
                abs_loss_mask = (y == 0.0)  # 布尔掩码，用于标记 y 中等于零的部分

                # 对等于零的部分使用绝对值损失
                loss[abs_loss_mask] = torch.abs(preds[abs_loss_mask] - y[abs_loss_mask])

                # 对不等于零的部分使用均方误差损失
                mse_loss_mask = ~abs_loss_mask  # 取反，用于标记 y 中不等于零的部分
                loss[mse_loss_mask] = torch.mean(torch.abs(preds[mse_loss_mask] - y[mse_loss_mask])) / y[mse_loss_mask]

                test_loss = loss.mean()

                sum_test_loss = sum_test_loss + test_loss

                # 评估指标
                y_denormalized = lab_scaler.inverse_transform(y.cpu())
                preds_denormalized = lab_scaler.inverse_transform(preds.cpu())

                test_rmse = (np.sqrt(((preds_denormalized - y_denormalized) ** 2).mean()))
                sum_test_rmse = sum_test_rmse + test_rmse

            sum_train_loss = sum_train_loss.cpu().detach().numpy()#数据存到cpu=
            sum_test_loss = sum_test_loss.cpu().detach().numpy()
            sum_mse_loss = sum_mse_loss.cpu().detach().numpy()
            
            all_train_loss.append(sum_train_loss / len(train_dataloader))
            all_test_loss.append(sum_test_loss / len(test_dataloader))
            all_mse_loss.append(sum_mse_loss / len(train_dataloader))
            all_train_rmse.append(sum_train_rmse / len(train_dataloader))
            all_test_rmse.append(sum_test_rmse / len(test_dataloader))
            all_net.append(model)

            model.train()



# best epoch
best_epoch = all_test_loss.index(min(all_test_loss)) + 1 

print('best_epoch=%d' % best_epoch*step_epoch)
if best_epoch == len(all_net):
    best_net = all_net[best_epoch-1]
else:
    best_net = all_net[best_epoch]


# perturbation train

best_net.cpu()
best_net.eval()
perturb.cpu()
another_epoch = int(num_epochs/10)

test_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_data, test_label), batch_size=32, shuffle=False)

for epoch in range(another_epoch):
    loop = tqdm(test_dataloader, total=len(test_dataloader))
    for x, y in loop:

        x = x.cpu()
        pp,pm,sigma = perturb(x)
        sigma = sigma.cpu().detach()#记得换个位置存
        p = best_net(x)
        pp = best_net(pp)
        pm = best_net(pm)
        pp = torch.abs(pp - p)
        pm = torch.abs(pm - p)
        const = torch.full([pp.size()[0],1],torch.max(p).item())
        const_0 = torch.full([pp.size()[0],1],torch.min(p).item())

        l1 = criterion(pp,const_0)
        l2 = criterion(pm,const)
        l3 = criterion(sigma,torch.zeros(num_feat))

        loss = l1 + l2 + l3


        perturb_optimizer.zero_grad()
        loss.backward()

        perturb_optimizer.step()
        
        loop.set_description('epoch: [%d/%d]' % (epoch, another_epoch))

    if epoch % step_epoch == 0: # 每step_epoch次epoch存一次数据
        save_sigma = sigma.clone().numpy()
        all_sigma.append(save_sigma)
# save



# 所有参数 
all_mse_loss = np.array(all_mse_loss).reshape(-1,1)
all_train_loss = np.array(all_train_loss).reshape(-1,1)
all_test_loss = np.array(all_test_loss).reshape(-1,1)
all_train_rmse = np.array(all_train_rmse).reshape(-1,1)
all_test_rmse = np.array(all_test_rmse).reshape(-1,1)
name_result = np.array(['all_mse_loss','all_train_loss','all_test_loss','all_train_rmse','all_test_rmse']).reshape(-1,1).T
exp_result = np.hstack((all_mse_loss,all_train_loss,all_test_loss,all_train_rmse,all_test_rmse))
result = np.vstack((name_result,exp_result))

torch.save(best_net,save_path + '/best_net.pth')
np.savetxt(save_path + '/result.csv',result,delimiter=',',fmt='%s')
np.savetxt(save_path + '/sigma.csv',np.array(all_sigma),delimiter=',',fmt='%s')
np.savetxt(save_path + '/select_wl.csv',select_wl,delimiter=',',fmt='%s')


# draw
import matplotlib.pyplot as plt
cutoff = 50
plt.figure(1)
plt.plot(all_train_loss[cutoff:], label='train_loss')
plt.plot(all_test_loss[cutoff:], label='test_loss')
plt.xlabel("epochs(10 epoch)")
plt.ylabel("all_train_loss(a.u.)")
plt.show()

plt.figure(2)
plt.plot(all_mse_loss[cutoff:], label='mse_loss')
plt.xlabel("epochs(10 epoch)")
plt.ylabel("all_mse_loss(a.u.)")
plt.show()

plt.figure(3)
plt.plot(all_train_rmse[cutoff:], label='train_rmse')
plt.plot(all_test_rmse[cutoff:], label='test_rmse')
plt.xlabel("epochs(10 epoch)")
plt.ylabel("all_train_rmse(a.u.)")
plt.show()

plt.figure(4)
x = y = [0,1000]
plt.plot(x,y)
plt.scatter(lab_scaler.inverse_transform(train_label), 
            lab_scaler.inverse_transform(best_net(train_data).detach().numpy()),
            label="train set",c = 'r')
plt.scatter(predict_label, 
            lab_scaler.inverse_transform(
                best_net(
                    torch.tensor(feat_scaler.fit_transform(predict_data[:, index_feat]))).detach().numpy()),
                label="test set",c = 'b')

plt.show()

perturb.eval()
p1,p2,sigma = perturb(test_data[0])
p1 = p1.detach().numpy()
p2 = p2.detach().numpy()

spec = raw_feat[0,:].copy()
sp1 = spec.copy()
sp2 = spec.copy()

sp1[index_feat] = p1
sp2[index_feat] = p2

plt.figure(5)
plt.plot(sp1,label='p+',c='r')
plt.plot(sp2,label='p-',c='b')
plt.plot(spec,label='origin',c='g')
plt.show()

plt.figure(6)
plt.plot(p1,label='p+',c='r')
plt.plot(p2,label='p-',c='b')
plt.plot(spec[index_feat],label='p+',c='g')
plt.plot(sigma.detach().numpy(),label='sigma',c='y')
plt.show()

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

X = lab_scaler.inverse_transform(test_label)
Y = lab_scaler.inverse_transform(best_net(test_data).detach().numpy())
print('R_2 = ',r2_score(X,Y))
print('Mae = ',mean_absolute_error(X,Y))
print('Rmse = ',np.sqrt(mean_squared_error(X,Y)))             
