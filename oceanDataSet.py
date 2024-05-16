from PIL import Image
from torch.utils.data import Dataset
import os


class oceanDataSet(Dataset):
    def __init__(self, root_dir, isTrain, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.img_list = []
        self.img_label = []
        self.data_dir = os.path.join(self.root_dir, "images")
        if isTrain:
            with open(self.root_dir+"train.csv", 'r') as f:
                lines = f.readlines()
                for line in lines:
                    self.img_path = os.path.join(self.data_dir, line.split(',')[0])
                    self.img_list.append(self.img_path)
                    self.img_label.append(int(line.split(',')[1].strip()))
                f.close()
        else:
            with open(self.root_dir+"novel.csv", 'r') as f:
                lines = f.readlines()
                for line in lines:
                    self.img_path = os.path.join(self.data_dir, line.split(',')[0])
                    self.img_list.append(self.img_path)
                    self.img_label.append(int(line.split(',')[1].strip()))
                f.close()

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = self.img_label[index]
        return img, label

    def __len__(self):
        return len(self.img_list)
