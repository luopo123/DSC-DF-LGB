import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
root_path = "F:\\China_XCO2\\"
dataset_path = root_path + "dataset_v2\\"
total_path = dataset_path + "total\\"

class CNN(nn.Module):
    def __init__(self, in_channels):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),  # 21+1-7 = 15
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(3)) # 15/3 = 5
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3), # 5+1-3 = 3
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(3)) # 3/3 = 1
        self.layer3 = nn.Conv2d(16, 2, kernel_size=1)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_feature(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class DSC(nn.Module):
    def __init__(self, in_channels):
        super(DSC, self).__init__()
        self.depthwise_1 = nn.Conv2d(in_channels, in_channels, 5, groups=in_channels) # 19+1-5 = 15
        self.batchNorm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
        self.pool = nn.AvgPool2d(3)  # 15/3 = 5
        self.depthwise_2 = nn.Conv2d(in_channels, in_channels, 3, groups=in_channels)
        # self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, x):
        x = self.depthwise_1(x)
        x = self.batchNorm(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.depthwise_2(x)
        x = self.pool(x)
        # out = self.pointwise(out)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_feature(self, x):
        x = self.depthwise_1(x)
        x = self.batchNorm(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.depthwise_2(x)
        x = self.pool(x)
        # out = self.pointwise(out)
        x = x.view(x.size(0), -1)
        return x

class RegressionDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        self.outputs = torch.tensor(outputs, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

if __name__== "__main__" :
    # 判断是否有可用GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"There are {torch.cuda.device_count()} GPU(s) available.")
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("No GPU available, using the CPU instead.")

    # 读入数据集, 打乱顺序
    surfaceDataset = np.load(total_path + "CNN_inputdata.npy")
    pointDataset = np.load(total_path + "outputdata.npy")
    indices = np.arange(surfaceDataset.shape[0])
    np.random.shuffle(indices)
    surfaceDataset = surfaceDataset[indices]
    pointDataset = pointDataset[indices]

    # 8:2划分训练集和测试集
    split_index = int(0.8 * surfaceDataset.shape[0])
    train_inputs = surfaceDataset[:split_index]
    train_outputs = pointDataset[:split_index]
    test_inputs = surfaceDataset[split_index:]
    test_outputs = pointDataset[split_index:]

    train_dataset = RegressionDataset(train_inputs, train_outputs)
    test_dataset = RegressionDataset(test_inputs, test_outputs)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = DSC(4).to(device)

    # 设置些超参数
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    Loss = nn.L1Loss()
    MSELoss = nn.MSELoss()

    # 训练模型
    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = MSELoss(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))

    # 测试模型
    def test():
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += MSELoss(output.squeeze(), target).item()
        test_loss /= len(test_loader)
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
        return test_loss

    for epoch in range(1, 100):
        train(epoch)
        test_loss = test()
        if test_loss < 25:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': Loss,
            }, 'checkpoint\\DSC_checkpoint.tar'  # 这里的后缀名官方推荐使用.tar
            )
            break
