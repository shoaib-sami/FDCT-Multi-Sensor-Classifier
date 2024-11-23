import torch
#import torch.cuda as cuda
import torch.optim as optim
import torch.nn as nn
from hadi_dataloader import YourDataset
from my_resnetA import model1 as model
import config
#root_dir = "/home/shoaibmerajsami/Desktop/UCHE/PycharmProjects/abc/"
root_dir = '/home/shoaibmerajsami/Desktop/atr hadi final/66_chip/visible/66_vis_ttv/val/'
train_data = YourDataset(root_dir)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=400, shuffle=True,  num_workers=4)

model = model
if torch.cuda.is_available():
    model= nn.DataParallel(model, device_ids=[0])



def test(model, checkpoint):
    root_dir = '/home/shoaibmerajsami/Desktop/atr hadi final/66_chip/visible/66_vis_ttv/val/'
    train_data = YourDataset(root_dir)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True, num_workers=4)

    model.load_state_dict(checkpoint["model"])
    num_correct = 0
    num_samples = 0
    accuracy = 0
    predictions = 0
    ls_sq_dist = []
    ls_sq_dist2 = []
    ls_labels = []
    # m = nn.LogSoftmax(dim=1)
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        img1, label = data
        # print(img1)
        img1 = img1.to(config.device)
        # label = label.type(torch.FloatTensor)

        label = label.to(config.device)
        scores = model(img1)

        scores = scores.to(config.device)
        _, predictions = scores.max(1)
        #print(predictions)
        # dis_my, bb= torch.max(scores,1)

        label = label.type(torch.FloatTensor).to(config.device)
        # out1, out2 = self.model(img1, img2)
        dist_sq = scores[:, 0]  # torch.sqrt() torch.pow(scores, 2)

        # print(label)

        ls_sq_dist.append(dist_sq.data)
        # ls_sq_dist.append(dis_my.data)
        ls_labels.append((1 - label).data)
        num_correct += (predictions == label).sum()
        num_samples += predictions.size(0)
        # my_utility_2_class_morph_real.calculate_scores(ls_labels, ls_sq_dist)

    #a, b, c, auc, eer = my_utility_apcer.calculate_scores(ls_labels, ls_sq_dist)
    accuracy = num_correct / num_samples
    print('accuracy is')
    print(accuracy)
    return accuracy




