# utf-*- coding:gbk -*-
# @Time     : 2019-09-25 16:01
# @Author   : Richardo Mu
# @FILE     : train_scratch.PY
# @Software : PyCharm
import torch
from torch import optim,nn
import visdom
from torch.utils.data import DataLoader
from hustdata import HustData
from resnet import ResNet18
import os
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
parser.add_argument('--num_class',type=int,default=10,
                    help='number of classification u want to ')
args = parser.parse_args()

device = torch.device('cuda')
torch.manual_seed(args.seed)
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

#viz = visdom.Visdom()


def evaluate(model,loader):
    correct = 0
    total = len(loader.dataset)
    for x,y in loader:
        x,y = x.to(device),y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            # total as total dataset length ,for example validation dataset's total is 315,
            # which means 315 image in it 
            # print("total",total)
            # logits dimension is 32 because we choose 32 as a batch ,so is y and pred
            # print('torch.eq(pred,y):',torch.eq(pred,y))
            """torch.eq(pred,y): tensor([False,  True, False,  True, False, False,  True,  True, False,  True,
        False, False,  True,  True,  True, False, False,  True, False,  True,
        False,  True,  True, False,  True, False, False, False,  True, False,
        False, False], device='cuda:0')
            """
            # print('torch.eq(pred,y).sum():',torch.eq(pred,y).sum())
            """ sum the true """
            # print("torch.eq(pred,y).sum().float():",torch.eq(pred,y).sum().float())
            """turn from 19 to 19.0 in case /total is integer"""
            # print("torch.eq(pred,y).sum().float().item():",torch.eq(pred,y).sum().float().item())
        correct += torch.eq(pred,y).sum().float().item()

    return  correct/total



def main():

    model = ResNet18(args.num_class).to(device)
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    criteon = nn.CrossEntropyLoss()

    best_acc,best_epoch = 0, 0
   
    #global_step = 0
    #viz.line([0],[-1],win='loss',opts=dict(title='loss'))
    #viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(1,args.epochs+1):
        correct_train = 0
        for step, (x,y) in enumerate(train_loader):
            # x : [b,3,224,224], y:[b]
            x,y = x.to(device),y.to(device)

            logits = model(x)
            loss = criteon(logits,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #viz.line([loss.item()], [global_step], win='loss', update='append')
            #global_step += 1
            # 计算acc
            pred = logits.argmax(dim=1)
            correct_train += torch.eq(pred,y).sum().float().item()
        total = len(train_loader.dataset)
        print("epoch:%d    loss:%f  acc:%f"%(epoch,loss,correct_train/total))
        print('--'*50)
        if epoch % 1 == 0:
            val_acc = evaluate(model,valid_loader)
            test_acc = evaluate(model,test_loader)
            print("epoch:%d    val_acc:%f    \n test_acc:%f"%(epoch,val_acc,test_acc))
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(),'best.mdl')
                print("model saved!!best_epoch: %d,best_acc:%f   "%(best_epoch,best_acc))
                print('--'*50)
                #viz.line([val_acc], [global_step], win='val_acc', update='append')
    print('best acc:',best_acc,'best_epoch:',best_epoch)

    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt!')
    test_acc = evaluate(model,test_loader)
    print('test acc:',test_acc)







if __name__ == '__main__':
    main()
