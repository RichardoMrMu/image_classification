Image_Classification Demo
===============================


#####################环境
pytorch  1.3
python 3.7


#####################目录结构描述
├── .idea                        // pycharm setting
├── __pycache__                  // pycharm setting                  
├── model                        // model parameters
│   ├── 2019.10.09-11类-scratch               // classify 11 normal resnet18
│   ├── 2019.10.09-11类-transfer         // classify 11 fine-tuning resnet18
│   ├── 2019.10.10-11类-scratch+eval-train          // classify 11 normal resnet18 add model.eval() and model.train()
│   ├── 2019.10.10-11类-transfer-resnet18       // classify 11 use fine-tuning resnet18 
│   └── 2019.10.10-11类-transfer-resnet34        // classify 11 use fine-tuning resnet34
├── test_img                                    //test img 
├── HustData_LOAD.py                      // load test img          
├── README.md                             // help 
├── __init__.py                      // init 
├── best.mdl                         // best model parameters 
├── crop_image.py                    // crop image by code make things in imgs more apparent 
├── hustdata.py                      // load train\valid\test imgs and add some enhance
├── resnet.py           //          model resnet18 
├── train_scratch.py   // normal resnet18
├── train_transfer.py  // fine-tuning resnet18 
├── train_transfer_resnet18.py        // same 
├── train_transfer_resnet34.py                     // fine-tuning resnet34
├── train_transfer_resnet50.py        // fine-tuning resnet50 
└── utils.py             // flatten which pytorch does not have 

################################使用描述
选择你需要的模型文件，运行train_scratch or train_transfer_resnetx.py 
文件中将main注释并将剩下代码解除注释可以进行测试
将main解除注释可以进行训练，在arg中可以修改参数，
如epoch class等，训练好的模型都是class为11，即
总共有11类。epoch 和 learing-rate可以自行选择，
也可以按照默认。

###############################备注
1.需要使用GPU，如果没有或者不想用，请讲代码中to device 等语句注释掉
2.更多问题请邮件联系tianbw2019@qq.com