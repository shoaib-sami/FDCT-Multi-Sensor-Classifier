import torch
#import torch.cuda as cuda
import torch.optim as optim
import torch.nn as nn
from hadi_dataloader_multi_modal_train import YourDataset
from resnet_visible import model_visible
from resnet_mwir import model_mwir
from may_val_pair_unpair import test
from dyswin_V2 import dynamic_swin_v2 as net

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
from mgca_module_decomp_concatenate import MGCA
import config
from sklearn.metrics import confusion_matrix
# helpers



#root_dir = "/home/shoaibmerajsami/Desktop/UCHE/PycharmProjects/abc/"
root_dir = '/media/shoaibmerajsami/new/VEDAI/dataset_preprocess/All_IR_68_68_class/train_multi_sensor/'
train_data = YourDataset(root_dir)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=6, shuffle=True,  num_workers=0)
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
        #self.model_mwir.load_state_dict(torch.load('/media/shoaibmerajsami/SMS/ATR_database_october_2021/cycle_gan_july_2022_final/Modified_CUT_sep12/MWIR_66_image_ResNet18_6_Epoch_Accuracy_99_62.pth')["model"],strict= False)
        #self.model_visible.load_state_dict(torch.load('/media/shoaibmerajsami/SMS/ATR_database_october_2021/cycle_gan_july_2022_final/Modified_CUT_sep12/Visible_66_image_ResNet18_5_Epoch99_28.pth')["model"],strict= False)
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

f_model = MGCA().to(config.device)#Fusion_model().cuda()
#f_model = nn.DataParallel(f_model, device_ids=[0,1]).cuda()


# Loss and optimizer

print("File Loaded into ResNet")
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0002)
optimizer = optim.AdamW(f_model.parameters(), lr=5e-6, weight_decay=0.05, betas=(0.9, 0.999)) #it was 2.5e-5
#optimizer = optim.AdamW(f_model.parameters(), lr=5e-4, weight_decay=1e-6, betas=(0.9, 0.999))
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=1e-8,last_epoch=-1)


check={}
check= torch.load('/home/shoaibmerajsami/Desktop/atr hadi final/DSIAC_multi_modal_MGCA/best_99_28_decomp_udt.pth')
a = check["model"]
#del a['fc.weight']
#del a['fc.bias']


f_model.load_state_dict(a, strict=True)
print("Model 99_20... lr...5e-6 is loaded 1000 Iteration")

#checkpoint1 = {}
#checkpoint1["model"] = f_model.state_dict()
#checkpoint1["optimizer"] = optimizer.state_dict()
#test(f_model, check)

print("I have done test")
acc=[]
num_correct = 0
num_samples = 0
accuracy = 0
predictions = 0
ls_sq_dist = []
ls_sq_dist2 = []
ls_labels = []
my_label =[]
my_prediction = []
for epoch in range(800):
    avg_loss=0
    num_correct=0
    num_samples=0
    # for batch_idx, (data, targets) in enumerate(tqdm(trainloader)):
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        visible_data, mwir_data, pair_unpair, targets = data
        #print(visible_data)
        #print(pair_unpair)
        # data=data.to(device=device)
        # targets=targets.to(device=device)
        #print(len(visible_data))
        #print(targets)
        vis=[]
        mwir =[]
        lb =[]
        #print(len(pair_unpair))
        for x in range(len(pair_unpair)):
            if pair_unpair[x] == 'unpair':
                vis.append(visible_data[x])
                mwir.append(mwir_data[x])
                lb.append(targets[x])
            else:
                vis.append(visible_data[x])
                mwir.append(mwir_data[x])
                lb.append(targets[x])

        if len(vis)!=0:
            vis1 =torch.stack(vis,dim=0)
            mwir1 =torch.stack(mwir,dim=0)
            lb1 =torch.stack(lb,dim=0)
        else:
            vis1 =torch.tensor([])
            mwir1 =torch.tensor([])
            lb1 =torch.tensor([])


        #print(vis1.size())
        #print(len(visible_data))
        vis1 = vis1.to(config.device)
        mwir1 = mwir1.to(config.device)
        lb1 = lb1.to(config.device)
        #print(lb1)
        #pair_unpair = pair_unpair.cuda()
        optimizer.zero_grad()
        # print('a')
        # Get data to cuda if possible
        # data = data.to(device=device)
        # targets = targets.to(device=device)
        if len(lb1) == 0:
            pass
        if len(lb1) == 1:
            pass
        else:
            #print("Train")
            #print(len(vis1))
            if len(vis1)==0:
                pass
            else:
                a,b,c,decomp,d,e,scores = f_model(vis1,mwir1)
                #acc.append(d)
                #print(sum(acc)/len(acc))
                #print(d)
                # scores2=torch.stack(list(scores), dim=0)
                scores = scores.to(config.device)
                #print(scores)
                #print("decompositon loss")
                #print(decomp)
                loss_scores = criterion(scores, lb1)
                loss = 1*a+1*b+1*c+ 1*decomp+loss_scores
                #print("Total Loss")
                #print(loss)
                #print(loss)
                #avg_loss += loss.item()
                #print(avg_loss/((i+1)*60))
                #print(targets)

                # print(targets)

                # backward
                _, predictions = scores.max(1)
                #print(predictions)
                # dis_my, bb= torch.max(scores,1)
                #print(lb1)
                label = lb1
                # out1, out2 = self.model(img1, img2)
                dist_sq = scores[:, 0]  # torch.sqrt() torch.pow(scores, 2)

                # print(label)

                ls_sq_dist.append(dist_sq.data)
                # ls_sq_dist.append(dis_my.data)
                #ls_labels.append((1 - label).data)
                num_correct += (predictions == label).sum()
                num_samples += predictions.size(0)
                abc=(predictions == label).sum()

                #done by SMS
                #print("Per iteration accuracy")
                #print(abc/3)
                #print(num_correct/num_samples)
                my_label.append(label.data)
                my_prediction.append(predictions.data)
                loss.backward()
                optimizer.step()
        #print(i)
        if i % 100 == 1:
            print("Iteration")
            print(i)
        if i % 1000 == 990:
            print("Iteration is %d", i)    
            model_file = "Third_ResNet50_Visible_%d_Epoch_1_2.pth" % (i+1+epoch)
            print(model_file)
            print("Bismillah")
            #abc = torch.load("/home/shoaibmerajsami/Desktop/atr hadi final/DSIAC_multi_modal_MGCA/Second_run_ResNet50_image72_6_Epoch.pth")
            checkpoint1 = {}
            checkpoint1["model"] = f_model.state_dict()
            checkpoint1["optimizer"] = optimizer.state_dict()
            test(f_model, checkpoint1)

            torch.save(checkpoint1, model_file)
            print("avg loss: {}".format(loss))
            
            scheduler.step()
            print("Epoch  {} | learning rate {}".format(epoch+ 1, scheduler.get_lr()))


  



        # gradient descent or adam step
        
    accuracy = num_correct / num_samples
    print('accuracy is')
    print(accuracy)
    pred_ls = torch.cat(my_label, 0)
    true_label = torch.cat(my_prediction, 0)
    pred_ls = pred_ls.cpu().detach().numpy()
    true_label = true_label.cpu().detach().numpy()
    #print("Confusion Matrix")
    #cm = confusion_matrix(true_label,pred_ls)
    #print(cm)  

    print("avg loss: {}".format(avg_loss))
    
    scheduler.step()
    print("Epoch  {} | learning rate {}".format(epoch+ 1,
                                                scheduler.get_lr()))

    model_file = "Third_run_ResNet50_image72_%d_Epoch.pth" % (epoch + 6)
    print(model_file)

    checkpoint1 = {}
    checkpoint1["model"] = f_model.state_dict()
    checkpoint1["optimizer"] = optimizer.state_dict()
    test(f_model, checkpoint1)

    torch.save(checkpoint1, model_file)
    print("saving model")

