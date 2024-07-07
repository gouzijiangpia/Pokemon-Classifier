import csv
import glob
import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Pokemon(Dataset):
    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()
        self.root = root
        self.resize = resize
        self.mode = mode
        self.name2label = {}  # "sq...":0
        # 遍历根目录10ClassesDataset下的文件和子目录，若是文件则跳过，若是子目录则将子目录名作为索引、字典当前的已有键的数量为标签（值）存入字典
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        # 将images及labels存入csv后再取出，images存放每个图片的绝对路径, labels存放对应的标签(0,1,……,9)
        self.images, self.labels = self.load_csv('images.csv')
        if mode == 'train':  # 60%
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
            print("TrainSize:", len(self.images))
        elif mode == 'val':  # 20% = 60%->80%
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
            print("ValidationSize:", len(self.images))
        elif mode == 'test':  # 20% = 80%->100%
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]
            print("TestSize:", len(self.images))
        else:
            print("Test1Size:", len(self.images))
            pass

    def load_csv(self, filename):
        # 若csv文件不存在则创建它，filename指的是csv的名字,这里将映射的csv文件存储在root目录下
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            # glob.glob文件名匹配模式，查找所有.png和.jpg后缀的文件加入到images中
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
            # 打乱顺序
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    # images中已经包含了label,这里通过split来读取出来,即取图片路径的倒数第二个文件名
                    # images: 如'10ClassesDataset\\Bulbasaur\\00000000.png',取Bulbasaur
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # 按 '10ClassesDataset\\Bulbasaur\\00000000.png', Bulbasaur对应标签0,然后写入csv中
                    writer.writerow([img, label])
                print('writen into csv file successful:', filename)

        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                images.append(img)
                labels.append(int(label))

        assert len(images) == len(labels)
        # 打乱顺序
        paired_data = list(zip(images, labels))
        random.shuffle(paired_data)
        shuffle_images = [item[0] for item in paired_data]
        shuffle_labels = [item[1] for item in paired_data]
        return shuffle_images, shuffle_labels

    def __len__(self):
        return len(self.images)

    # 按索引取图片,进行数据增强后再返回
    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        trans = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path= > image data,打开图像文件并将其转换为 RGB 格式。
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),  # 先放大到原来的1.25倍
            transforms.RandomRotation(15),  # 随机旋转图像最多 15 度。
            transforms.CenterCrop(self.resize),  # 再居中裁剪图像到原来的大小。
            transforms.ToTensor(),  # 将 PIL图像或NumPy ndarray转换为FloatTensor,并将图像的数值范围从[0, 255]转换为 [0.0, 1.0]。
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化处理
                                 std=[0.229, 0.224, 0.225])
        ])
        img = trans(img)
        label = torch.tensor(label)
        return img, label

    # 逆标准化
    def denormalize(self, x_hat):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # x.size = [c, h, w]
        # mean和std需要与x同维度，于是resize变为3维张量:[3]->[3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x
