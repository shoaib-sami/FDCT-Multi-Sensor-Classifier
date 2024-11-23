import torch
#import torch.cuda as cuda
import torch.optim as optim
import torch.nn as nn
from hadi_dataloader_multi_modal_test import YourDataset
from resnet_visible import model_visible
from resnet_mwir import model_mwir
#from may_val_pair_unpair import test
import config
#root_dir = "/home/shoaibmerajsami/Desktop/UCHE/PycharmProjects/abc/"
root_dir = '/media/shoaibmerajsami/new/VAIS_dataset/ALL_IR/test/'
train_data = YourDataset(root_dir)
#test_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True,  num_workers=4)
from mgca_module_decomp_concatenate_test import MGCA
from sklearn.metrics import confusion_matrix
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

        self.fusion = nn.Sequential(
            nn.Linear(200, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
            ).cuda()

    def forward(self,mwir_data, visible_data):
       
        output_vis = model_visible(visible_data)
        output_mwir = model_mwir(mwir_data)
        output =torch.cat((output_vis, output_mwir), dim=1)

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
        output = self.fusion(output)


        return output

model = MGCA().to(torch.device('cuda:1'))

def test(model1, checkpoint):
    root_dir = '/home/shoaibmerajsami/Desktop/atr hadi final/Multi_domain_fusion_dr_asif/dataset/test_llm/'
    test_data = YourDataset(root_dir)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=80, shuffle=True, num_workers=4)

    model.load_state_dict(checkpoint["model"])
    model.eval()
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
    my_label=[]
    my_prediction=[]
    # m = nn.LogSoftmax(dim=1)
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        visible_data, mwir_data, pair_unpair, targets = data
        # data=data.to(device=device)
        # targets=targets.to(device=device)
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
        vis1 = vis1.to(torch.device('cuda:1'))
        mwir1 = mwir1.to(torch.device('cuda:1'))
        lb1 = lb1.to(torch.device('cuda:1'))
        if len(vis1)==[0]:
            pass
        if len(vis1)==[1]:
            pass

        if len(vis1)==0:
            pass
        if len(vis1)==1:
            pass
        else:
            #print("Hello")
            #print(len(vis1))
            if len(vis1)==0:
                pass
            else:
                #print("Finished")
                with torch.no_grad() :
                    a,b,c,decom,d,e,scores = model(vis1,mwir1)
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
                ls_labels.append((1 - label).data)
                num_correct += (predictions == label).sum()
                num_samples += predictions.size(0)
                aaa=0
                aaa=(predictions == label).sum()

                #done by SMS
                #print("Per iteration accuracy")
                #print(aaa/2)
                #print(num_correct/num_samples)
                my_label.append(label.data)
                my_prediction.append(predictions.data)
                #print(a,b,c,d,e)
    accuracy = num_correct / num_samples
    print('accuracy is')
    print(accuracy)
    pred_ls = torch.cat(my_label, 0)
    true_label = torch.cat(my_prediction, 0)
    pred_ls = pred_ls.cpu().detach().numpy()
    true_label = true_label.cpu().detach().numpy()
    print("Confusion Matrix")
    cm = confusion_matrix(true_label,pred_ls)
    print(cm)  
        #pair_unpair = pair_unpair.cuda()
        
        # print('a')
        # Get data to cuda if possible
        # data = data.to(device=device)
        # targets = targets.to(device=device)


       
        




