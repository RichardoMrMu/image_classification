# -*- coding:utf-8 -*-
# @Time     : 2019-09-24 19:54
# @Author   : Richardo Mu
# @FILE     : hustdata.PY
# @Software : PyCharm
import torch
import os,glob
import random,csv
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class HustData(Dataset):
    def __init__(self,root,resize,mode):
        super(HustData, self).__init__()

        self.root = root
        self.resize = resize
        self.name2label = {} 
        for name in sorted(os.listdir(root)):
            if not os.path.isdir(os.path.join(root,name)):
                continue
            self.name2label[name] = len(self.name2label.keys())

        # print(self.name2label)
        # self.load_csv('image.csv')
        # image + label save as csv
        self.images,self.labels = self.load_csv('images.csv')

        if mode=='train':
            self.images = self.images[:int(0.6*len(self.images))]
            self.labels = self.labels[:int(0.6*len(self.labels))]
        elif mode == 'valid':
            self.images = self.images[int(0.6*len(self.images)):int(0.8*len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:#
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]


    def load_csv(self,filename):
        name2label_file = 'name2label.csv'
        if not os.path.exists(os.path.join(self.root, name2label_file)):
            with open(os.path.join(self.root,name2label_file),mode='w',newline='') as f:
                writer = csv.writer(f)
                for name_,label_ in enumerate(self.name2label):
                    writer.writerow([name_,label_])
        if not os.path.exists(os.path.join(self.root,filename)):
            images = []
            for name in self.name2label.keys():
                # 'hustdata\\statue_Mao\\0001.jpg
                images += glob.glob(os.path.join(self.root,name,'*.png'))
                images += glob.glob(os.path.join(self.root,name,'*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            """298 'E:\\debug\\pyCharmdeBug\\image_classification\\HustData\\Dinning_Hall\\IMG_20190920_113424.jpg',"""
            # print(len(images),images)
            random.shuffle(images)
            with open(os.path.join(self.root,filename),mode='w',newline='') as f:
                writer = csv.writer(f)
                for img in images:#'E:\\debug\\pyCharmdeBug\\image_classification\\HustData\\Dinning_Hall\\IMG_20190920_113424.jpg'
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img,label])
                print('writen into csv file  ',filename)
        # read from csv file
        images,labels = [],[]
        with open(os.path.join(self.root,filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img , label = row
                label = int(label)
                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)
        return images,labels


    def __len__(self):
        return len(self.images)


    def denormalize(self,x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # x_hot = (x-mean)/std
        # x  = x_hot*std = mean
        # mead:[3 ] => [3,1,1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x


    def __getitem__(self, idex):
        # idex [0-len]
        # self.images self.labels
        # img :E:\\debug\\pyCharmdeBug\\image_classification\\HustData\\Dinning_Hall\\IMG_20190920_113424.jpg'
        # label : 0
        img, label = self.images[idex],self.labels[idex]
        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),
           
            transforms.Resize((int(self.resize*1.25),int(self.resize*1.25))),
            
            transforms.RandomRotation(15),
           
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) 
        ])
        #                            
        img = tf(img)
        label = torch.tensor(label)
        return img, label


def main(folder):
    import visdom
    import time
    viz = visdom.Visdom()
    db = HustData(folder,1024,"train")
    # x,y = next(iter(db))
    # print(next(iter(db)))
    # print('sampel:',x.shape,y.shape,y)
    # viz.image(db.denormalize(x),win='sample_x',opts=dict(title='sample_x'))
    loader = DataLoader(db,batch_size=1,shuffle=True)
    for x,y in loader:
        viz.images(db.denormalize(x),nrow=1,win='batch',opts=dict(title='batch'))
        viz.text(str(y.numpy()),win='label',opts=dict(title='batch-y'))
        time.sleep(10)


if __name__ == '__main__':
    # print(os.path)
    filename = 'HustData'
    # filename = 'ascall'
    cwd = os.getcwd()
    print(cwd)
    folder = os.path.join(cwd,filename)
    # folder = 'E:\\debug\\pyCharmdeBug\\image_classification\\HustData'
    main(folder)
