import torch
from torch import nn, optim
from tqdm import tqdm

import wrn_mixup_model
from torchvision.transforms import transforms
from oceanDataSet import oceanDataSet
from torch.utils.data import DataLoader

data_dir = "/root/dataSet/Ocean/"
num_classes = 64  # 分类数
batch_size = 128  # 批处理大小
max_epochs = 10  # 最大微调轮数

# 使用GPU进行微调，如没有GPU，则使用cup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = wrn_mixup_model.wrn28_10(num_classes=num_classes, drop_rate=0.5)
model.load_state_dict(torch.load('./checkpoint/bestModel.pth'))
model.to(device)

isTrain = True
trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize([80, 80])

])
train_set = oceanDataSet(data_dir, isTrain, transform=trans)

train_loder = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


def training(model, loder):
    # 将模型设置为训练模式
    model.train()
    total_loss = 0.0  # 总损失
    total_acc = 0.0
    for data in tqdm(loder, bar_format="{l_bar}{bar} {n_fmt}/{total_fmt}"):
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        # 前向传播
        _, outputs = model(imgs)
        loss = criterion(outputs, targets)
        # 计算损失
        total_loss += loss.item()
        # 计算精度
        _, pred = torch.max(outputs, dim=1)
        acc = torch.sum(pred == targets) / batch_size
        total_acc += acc

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(loder), total_acc / len(loder)  # 返回平均损失,返回平均精确度


if __name__ == "__main__":
    max_acc = 0.0
    for epoch in range(max_epochs):
        epoch_loss, epoch_acc = training(model, train_loder)
        print(f'第{epoch + 1}轮的损失为：{epoch_loss}')
        # 保存微调后的模型
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoint/train_ocean_epoch_{epoch + 1}.pth")
        if epoch_acc > max_acc:
            max_acc = epoch_acc
            torch.save(model.state_dict(), "checkpoint/bestModel.pth")
        print(f'第{epoch + 1}轮的精确度为：{epoch_acc}')
