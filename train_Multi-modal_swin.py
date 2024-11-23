import torch
#import torch.cuda as cuda
import torch.optim as optim
import torch.nn as nn
from hadi_dataloader_multi_modal_train import YourDataset
from resnet_visible import model_visible
from resnet_mwir import model_mwir
from may_val_pair_unpair import test
from dyswin_V1 import dynamic_swin as net

import torch.nn as nn
import torch
#from backbones import get_model
#import net

import torch
from torch import nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
#from magface import iresnet100 as mag_iresnet100



# helpers



#root_dir = "/home/shoaibmerajsami/Desktop/UCHE/PycharmProjects/abc/"
root_dir = '/home/shoaibmerajsami/Desktop/atr hadi final/Visible_NVESD/train/'
train_data = YourDataset(root_dir)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=80, shuffle=True,  num_workers=4)
class Fusion_model(nn.Module):

    def __init__(self, mwir_data = None, visible_data = None, pair_unpair=None):
        super().__init__()
        """
        if torch.cuda.is_available():
            self.model_visible= nn.DataParallel(model_visible, device_ids=[0,1])
            self.model_mwir= nn.DataParallel(model_mwir, device_ids=[0,1])
        if torch.cuda.is_available():
            self.model_visible=self.model_visible.cuda()
            self.model_mwir=self.model_mwir.cuda()
        """
        self.model_visible=nn.DataParallel(model_visible, device_ids=[0,1]).cuda()
        self.model_mwir = nn.DataParallel(model_mwir, device_ids=[0,1]).cuda()
        self.model_mwir.load_state_dict(torch.load('/media/shoaibmerajsami/SMS/ATR_database_october_2021/cycle_gan_july_2022_final/Modified_CUT_sep12/MWIR_66_image_ResNet18_6_Epoch_Accuracy_99_62.pth')["model"],strict= False)
        self.model_visible.load_state_dict(torch.load('/media/shoaibmerajsami/SMS/ATR_database_october_2021/cycle_gan_july_2022_final/Modified_CUT_sep12/Visible_66_image_ResNet18_5_Epoch99_28.pth')["model"],strict= False)
        self.swin = nn.DataParallel(net, device_ids=[0,1]).cuda()



        
    def forward(self,mwir_data, visible_data):
        output_vis = self.model_visible(visible_data)

        output_mwir = self.model_mwir(mwir_data)
        out_fusion = torch.cat([output_vis, output_mwir],dim =2)

        out_fusion = self.swin(out_fusion)

        


        #output =torch.cat((output_vis, output_mwir), dim=1).cuda()

        """
        if pair_unpair == 'unpair':
            output_vis = model_visible(visible_data)
            output_vis2 = model_visible(mwir_data)
            output =torch.cat((output_vis, output_vis2), dim=1)
        else:
            output_vis = model_visible(visible_data)
            output_vis2 = model_visible(mwir_data)
            output =torch.cat((output_vis, output_vis2), dim=1)
            print("Strange")
        """

        #output = output.view(output.size(0), -1)
        #out_fusion = self.fusion(out_fusion).cuda()



        return out_fusion

f_model = Fusion_model().cuda()
#f_model = nn.DataParallel(f_model, device_ids=[0,1]).cuda()


# Loss and optimizer
#f_model.load_state_dict(torch.load('/home/shoaibmerajsami/Desktop/atr hadi final/Multi_domain_fusion/ResNet50_image72_6_Epoch.pth')["model"])
#model.load_state_dict(checkpoint["model"])
print("File Loaded into ResNet")
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0002)
optimizer = optim.Adam(f_model.parameters(), lr=0.0001, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)




for epoch in range(800):
    avg_loss=0
    # for batch_idx, (data, targets) in enumerate(tqdm(trainloader)):
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        visible_data, mwir_data, pair_unpair, targets = data
        # data=data.to(device=device)
        # targets=targets.to(device=device)
        #print(len(visible_data))
        vis=[]
        mwir =[]
        lb =[]
        #print(len(pair_unpair))
        for x in range(len(pair_unpair)):
            if pair_unpair[x] == 'unpair':
                pass
            else:
                vis.append(visible_data[x])
                mwir.append(mwir_data[x])
                lb.append(targets[x])

        vis1 =torch.stack(vis,dim=0)
        mwir1 =torch.stack(mwir,dim=0)
        lb1 =torch.stack(lb,dim=0)
        #print(vis1.size())
        #print(len(visible_data))
        vis1 = vis1.cuda()
        mwir1 = mwir1.cuda()
        lb1 = lb1.cuda()
        #pair_unpair = pair_unpair.cuda()
        optimizer.zero_grad()
        # print('a')
        # Get data to cuda if possible
        # data = data.to(device=device)
        # targets = targets.to(device=device)


        scores = f_model(vis1,mwir1)
        # scores2=torch.stack(list(scores), dim=0)
        loss = criterion(scores, lb1)
        avg_loss += loss.item()
        #print(avg_loss/((i+1)*60))
        #print(targets)

        # print(targets)

        # backward
        loss.backward()
        #print(i)
        if i % 1000 == 820:
            print("Iteration is %d", i)

            if (epoch ==0 or epoch == 1) and i>1:
                model_file = "ResNet50_Visible_%d_Epoch_1_2.pth" % (i+1+epoch)
                print(model_file)

                checkpoint1 = {}
                checkpoint1["model"] = f_model.state_dict()
                checkpoint1["optimizer"] = optimizer.state_dict()
                test(f_model, checkpoint1)

                torch.save(checkpoint1, model_file)


  



        # gradient descent or adam step
        optimizer.step()
    print("avg loss: {}".format(avg_loss))
    if epoch > 0 :
        scheduler.step()
    print("Epoch  {} | learning rate {}".format(epoch+ 1,
                                                scheduler.get_lr()))

    model_file = "ResNet50_image72_%d_Epoch.pth" % (epoch + 6)
    print(model_file)

    checkpoint1 = {}
    checkpoint1["model"] = f_model.state_dict()
    checkpoint1["optimizer"] = optimizer.state_dict()
    test(f_model, checkpoint1)

    torch.save(checkpoint1, model_file)
    print("saving model")

