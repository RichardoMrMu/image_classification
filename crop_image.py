# -*- coding:utf-8 -*-
# @Time     : 2019-09-29 9:26
# @Author   : Richardo Mu
# @FILE     : crop_image.PY
# @Software : PyCharm
# crop image
import os
import glob
from PIL import Image
import re
root = 'E:\\debug\\pyCharmdeBug\\image_classification\\root'
def load_img(root):
    img = []
    img += glob.glob(os.path.join(root,'*.jpg'))
    img += glob.glob(os.path.join(root, '*.png'))
    img += glob.glob(os.path.join(root, '*.jpeg'))
    return img
def main(save_path):
    image = load_img(root)
    a = 0.2
    b = 0.1
    c = 0.8
    d = 0.9
    a = root.replace('\\', '\\\\')
    for img in image:
        name = re.sub(a,'',img)
        name = re.sub('.jpg','',name)
        name = re.sub('\\\\','',name)
        im = Image.open(img)
        x,y = im.size
        region = im.crop((x*a,y*b,x*c,y*d))
        # [2347,2517]
        # 第一个为宽，第二个为�?
        # img_size.append(list(im.size))
        # region = im.crop((x, y, x+w, y+h))

        if not os.path.exists(os.path.join(root,save_path)):
            os.makedirs(os.path.join(root,save_path))
        region.save(os.path.join(root,save_path,'chage_'+name+'_'+'.jpg'))


if __name__ == '__main__':

    path = input('请输入需要操作的文件夹名�?')
    root = os.path.join(root,path)
    print(root)
    save_path = 'statue_mao2'
    main(save_path)