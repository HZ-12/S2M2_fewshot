import pickle

import torch
from tqdm import tqdm
import numpy as np

import wrn_mixup_model
from torchvision.transforms import transforms
from oceanDataSet import oceanDataSet
from torch.utils.data import DataLoader

data_dir = "/root/autodl-tmp/dataSet/Ocean/"
base_num_classes = 64
novel_num_classes = 20
batch_size = 256

# 使用GPU进行微调，如没有GPU，则使用cup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = wrn_mixup_model.wrn28_10(num_classes=64, drop_rate=0.5)

model.load_state_dict(torch.load('/root/autodl-tmp/checkpoint/bestModel.pth'))

model.to(device)

isTrain = True
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize([80, 80])

])

base_set = oceanDataSet(data_dir, isTrain=isTrain, transform=trans)
novel_set = oceanDataSet(data_dir, isTrain=False, transform=trans)

base_loder = DataLoader(base_set, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)
novel_loder = DataLoader(novel_set, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)


def save(model, loder, isBase=True):
    model.eval()
    features = {}
    arr = np.empty((0, 640))

    with torch.no_grad():
        for data in tqdm(loder, bar_format="{l_bar}{bar} {n_fmt}/{total_fmt}"):
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            # 前向传播
            feature, _ = model(imgs)
            arr = np.concatenate((arr, feature.cpu().numpy()), axis=0)

    if isBase:
        for i in range(64):
            features[str(i)] = arr[i*600:(i + 1) * 600].tolist()
        with open("/root/autodl-tmp/features/base_feature.pickle", "wb") as file:
            # 使用 pickle 序列化字典并写入文件
            pickle.dump(features, file)
    else:
        for i in range(20):
            features[str(i)] = arr[i * 600:(i + 1) * 600].tolist()
        with open("/root/autodl-tmp/features/novel_feature.pickle", "wb") as file:
            # 使用 pickle 序列化字典并写入文件
            pickle.dump(features, file)


if __name__ == "__main__":
    # save(model, base_loder, isBase=True)
    save(model, novel_loder, isBase=False)
