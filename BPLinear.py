import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())
x, y = Variable(x),Variable(y)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1,n_hidden2, n_output):
        super(Net,self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature,n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1,n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2,n_output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x= F.relu(self.hidden2(x))
        x = self.predict(x)
        return x



net = Net(n_feature=1,n_hidden1=10, n_hidden2=8, n_output=1)

optimizer = torch.optim.SGD(net.parameters(),lr=0.7)
loss_func = torch.nn.MSELoss()


plt.ion()


for i in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 5 ==0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5,0, 'Loss=%.4f'%loss.data[0],fontdict = {'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()