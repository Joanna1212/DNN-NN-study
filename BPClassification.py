import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden1,n_output):
        super(Net,self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature,n_hidden1)
        self.out = torch.nn.Linear(n_hidden1,n_output)
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.out(x)
        return x
net = Net(n_feature=2, n_hidden1=10,n_output=2)
# print(net)
optimizer = torch.optim.SGD(net.parameters(),lr = 0.02)
loss_func = torch.nn.CrossEntropyLoss() # torch.nn.MSELoss()

n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data,1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data,1)
y1 = torch.ones(100)

x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1), ).type(torch.LongTensor)

x,y = Variable(x),Variable(y)


for i in range(100):
    out = net(x)
    loss = loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i %2 ==0:
        plt.cla()
        prediction = torch.max(F.softmax(out),1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()  # 停止画图
plt.show()