# -*- coding:utf-8 -*-
# @Time     : 2019-09-26 8:45
# @Author   : Richardo Mu
# @FILE     : train_transfer.PY
# @Software : PyCharm
import torch
# import torchvision
from torch import optim,nn
import visdom
from torch.utils.data import DataLoader
from hustdata import HustData
# from resnet import ResNet18
import os
from torchvision.models import resnet34
from utils import Flatten
import argparse
#
parser = argparse.ArgumentParser(description="Image Classification -ResNet18")
parser.add_argument('--epochs',type=int,default=30,
                    help='epochs limit (default 10)')
parser.add_argument('--lr',type=float,default=1e-3,
                    help='initial learning rate (default :1e-3)')
parser.add_argument('--seed',type=int,default=1234,
                    help='random seed (default: 1234)')
parser.add_argument('--batchsz',type=int,default=32,
                    help='data batch_size(default=32)')
parser.add_argument('--num_class',type=int,default=5,
                    help='number of classification u want to ')
# parser.add_argument("")
args = parser.parse_args()

torch.manual_seed(args.seed)

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda')

cwd = os.getcwd()
# print(cwd)
folder = os.path.join(cwd,'HustData')

train_db = HustData(folder,224,mode='train')
valid_db = HustData(folder,224,mode='valid')
test_db = HustData(folder,224,mode='test')
train_loader = DataLoader(train_db,batch_size=args.batchsz,shuffle=True,
                          num_workers=4)
valid_loader = DataLoader(valid_db,batch_size=args.batchsz,num_workers=2)
test_loader = DataLoader(test_db,batch_size=args.batchsz,num_workers=2)



# viz = visdom.Visdom()


def evaluate(model,loader):
    correct = 0
    total = len(loader.dataset)
    for x,y in loader:
        x,y = x.to(device),y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred,y).sum().float().item()

    return  correct/total



def main():

    # model = ResNet18(10).to(device)
    trained_model = resnet34(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1],  # [b,512,1,1]
                          Flatten(),  # [b,512,1,1]=>[b,512]
                          nn.Linear(512, args.num_class)
                          ).to(device)
    # x = torch.randn(2,3,224,224)
    # print(model(x).shape)

    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    criteon = nn.CrossEntropyLoss()

    best_acc,best_epoch = 0, 0

    # global_step = 0
    # viz.line([0],[-1],win='loss',opts=dict(title='loss'))
    # viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(args.epochs):
        for step, (x,y) in enumerate(train_loader):
            # x : [b,3,224,224], y:[b]
            x,y = x.to(device),y.to(device)

            logits = model(x)
            loss = criteon(logits,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # viz.line([loss.item()], [global_step], win='loss', update='append')
            # global_step += 1

        if epoch % 1 == 0:
            val_acc = evaluate(model,valid_loader)
            test_acc = evaluate(model,test_loader)
            print("epoch:%d    val_acc:%f    \n test_acc:%f"%(epoch,val_acc,test_acc))
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(),'best.mdl')
                print("model saved!!best_epoch: %d,best_acc:%f  "%(best_epoch,best_acc))
                print('--'*50)
                # viz.line([val_acc], [global_step], win='val_acc', update='append')
    print('best acc:',best_acc,'best_epoch:',best_epoch)

    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt!')
    test_acc = evaluate(model,test_loader)
    print('test acc:',test_acc)







if __name__ == '__main__':
    # print(torch.cuda.device_count())
    main()

