# -*- coding:utf-8 -*-
# @Time     : 2019-09-26 8:45
# @Author   : Richardo Mu
# @FILE     : train_transfer.PY
# @Software : PyCharm
# -*- coding:utf-8 -*-
# @Time     : 2019-09-25 16:01
# @Author   : Richardo Mu
# @FILE     : train_scratch.PY
# @Software : PyCharm
import torch
# import torchvision
from torch import optim,nn
import visdom
from torch.utils.data import DataLoader
from hustdata import HustData
# from resnet import ResNet18
import os
from torchvision.models import resnet50
from utils import Flatten
from HustData_LOAD import HustData_LOAD
# 超参管理
import argparse
#
parser = argparse.ArgumentParser(description="Image Classification -ResNet18")
parser.add_argument('--epochs',type=int,default=10,
                    help='epochs limit (default 10)')
parser.add_argument('--lr',type=float,default=1e-3,
                    help='initial learning rate (default :1e-3)')
parser.add_argument('--seed',type=int,default=1234,
                    help='random seed (default: 1234)')
parser.add_argument('--batchsz',type=int,default=32,
                    help='data batch_size(default=32)')
parser.add_argument('--num_class',type=int,default=11,
                    help='number of classification u want to ')
# parser.add_argument("")
args = parser.parse_args()
# batchsz = 32
# lr = 1e-3
# epochs = 10
torch.manual_seed(args.seed)
# 设置cuda使用的id
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda')

cwd = os.getcwd()
# print(cwd)
folder = os.path.join(cwd,'HustData')

# model = ResNet18(10).to(device)
trained_model = resnet50(pretrained=True)
model = nn.Sequential(*list(trained_model.children())[:-1],  # [b,512,1,1]
                      Flatten(),  # [b,512,1,1]=>[b,512]
                      nn.Linear(512, args.num_class)
                      ).to(device)
# x = torch.randn(2,3,224,224)
# print(model(x).shape)

optimizer = optim.Adam(model.parameters(),lr=args.lr)
criteon = nn.CrossEntropyLoss()



# viz = visdom.Visdom()


def evaluate(loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x,y in loader:
        x,y = x.to(device),y.to(device)
        with torch.no_grad():
            #print(total)
            logits = model(x)
            #print("x:",x)
            pred = logits.argmax(dim=1)
            #print("torch.eq(pred,y):",torch.eq(pred,y))
            #print("torch.eq(pred,y).sum()",torch.eq(pred,y).sum())
        correct += torch.eq(pred,y).sum().float().item()

    return  correct/total

def test_img():
    model.eval()
    folder1 = os.path.join(cwd,'folder2')
    test_img_db = HustData_LOAD(root=folder1,resize=224)
    test_img_dataloder = DataLoader(test_img_db,batch_size=args.batchsz,shuffle=False,
                                    num_workers=2)
    for x in test_img_dataloder:
        x = x.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            print(pred)

def train(epoch,train_loader):
    model.train()
    correct_train = 0
    for step, (x,y) in enumerate(train_loader):
        # x : [b,3,224,224], y:[b]
        x,y = x.to(device),y.to(device)
        logits = model(x)
        loss = criteon(logits,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=1)
        correct_train += torch.eq(pred,y).sum().float().item()
    total = len(train_loader.dataset)
    print("epoch:%d    loss:%f  acc:%f"%(epoch,loss,correct_train/total))
    print('--'*50)

def main():
    train_db = HustData(folder, 224, mode='train')
    valid_db = HustData(folder, 224, mode='valid')
    test_db = HustData(folder, 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=args.batchsz, shuffle=True,num_workers=4)
    valid_loader = DataLoader(valid_db, batch_size=args.batchsz,num_workers=2)
    test_loader = DataLoader(test_db, batch_size=args.batchsz,num_workers=2)

   
    best_acc,best_epoch = 0, 0

    # global_step = 0
    # viz.line([0],[-1],win='loss',opts=dict(title='loss'))
    # viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(args.epochs):
        train(epoch,train_loader)

            # viz.line([loss.item()], [global_step], win='loss', update='append')
            # global_step += 1

        # if epoch % 1 == 0:
        val_acc = evaluate(valid_loader)
        test_acc = evaluate(test_loader)
        print("epoch:%d    val_acc:%f    \n test_acc:%f"%(epoch,val_acc,test_acc))
        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc

            torch.save(model.state_dict(),'best_transfer_resnet50.mdl')
            print("model saved!!best_epoch: %d,best_acc:%f   "%(best_epoch,best_acc))
            print('--'*50)
            # viz.line([val_acc], [global_step], win='val_acc', update='append')
    print('best acc:',best_acc,'best_epoch:',best_epoch)

    model.load_state_dict(torch.load('best_transfer_resnet50.mdl'))
    print('loaded from ckpt!')
    test_acc = evaluate(test_loader)
    print('test acc:',test_acc)







if __name__ == '__main__':
    # print(torch.cuda.device_count())
    main()
    # trained_model = resnet18(pretrained=True)
    # model = nn.Sequential(*list(trained_model.children())[:-1],  # [b,512,1,1]
    #                       Flatten(),  # [b,512,1,1]=>[b,512]
    #                       nn.Linear(512, args.num_class)
    #                       ).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # criteon = nn.CrossEntropyLoss()

    # model.load_state_dict(torch.load('best.mdl'))
    # print('loaded from ckpt!')
    # test_img()
