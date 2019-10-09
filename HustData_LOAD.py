#  加载图片
import torch
import os, glob
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset,DataLoader
from torchvision import  transforms
import csv,random

ImageFile.LOAD_TRUNCATED_IMAGES = True


class HustData_LOAD(Dataset):
    def __init__(self, root, resize):
        super(HustData_LOAD, self).__init__()

        self.root = root
        self.resize = resize
        self.images = self.load_img()


    def load_img(self):
        images = []
        images += glob.glob(os.path.join(self.root,  '*.png'))
        images += glob.glob(os.path.join(self.root,  '*.jpg'))
        images += glob.glob(os.path.join(self.root,  '*.jpeg'))
        """298 'E:\\debug\\pyCharmdeBug\\image_classification\\HustData\\Dinning_Hall\\IMG_20190920_113424.jpg',"""
        return images


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idex):
        # idex [0-len]
        # self.images self.labels
        # img :E:\\debug\\pyCharmdeBug\\image_classification\\HustData\\Dinning_Hall\\IMG_20190920_113424.jpg'
        # label : 0
        img = self.images[idex]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),

            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),

            transforms.RandomRotation(15),

            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        return img


def main(folder):
    import visdom
    import time
    viz = visdom.Visdom()
    db = HustData_LOAD(folder, 1024, "train")
    # x,y = next(iter(db))
    # print(next(iter(db)))
    # print('sampel:',x.shape,y.shape,y)
    # viz.image(db.denormalize(x),win='sample_x',opts=dict(title='sample_x'))
    loader = DataLoader(db, batch_size=1, shuffle=True)
    for x, y in loader:
        viz.images(db.denormalize(x), nrow=1, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
        time.sleep(10)


if __name__ == '__main__':
    # print(os.path)
    filename = 'HustData'
    # filename = 'ascall'
    cwd = os.getcwd()
    print(cwd)
    folder = os.path.join(cwd, filename)
    # folder = 'E:\\debug\\pyCharmdeBug\\image_classification\\HustData'
    main(folder)