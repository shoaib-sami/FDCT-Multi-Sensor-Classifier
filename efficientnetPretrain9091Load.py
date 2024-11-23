from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch
model1 = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5)


is_load=True
import os
import torch
import torch.optim as optim
import torch.nn as nn
from hadi_dataloader import YourDataset


root_dir = "/home/shoaibmerajsami/Desktop/roi_test/"
testa = YourDataset(root_dir)
testloader = torch.utils.data.DataLoader(testa, batch_size=50, shuffle=True,  num_workers=4)
"""
root_dir = "/home/shoaibmerajsami/Desktop/UCHE/PycharmProjects/abc/"
train_data = YourDataset(root_dir)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=200, shuffle=True,  num_workers=4)
"""
model =model1
#model = model.cuda()

    #model= nn.DataParallel(model, device_ids=[0,1])
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0002)
optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.92)



def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()
            y = y.cuda()

            scores = model(x)
            _, predictions = scores.max(1)
            print('label')
            #print(y)
            #print('prediction next')
            print(predictions)
            num_correct += (predictions == (y+0)).sum()
            num_samples += predictions.size(0)

    #model.train()

    return num_correct/num_samples


# model_file='siamese_for_VGG_165_Epoch_model.pth'
path='/home/shoaibmerajsami/Desktop/EfficientNet91.pth'
checkpoint = torch.load(path, map_location=torch.device('cpu'))

# create new OrderedDict that does not contain `module.`
from collections import OrderedDict

new_state_dict = OrderedDict()
for key, v in checkpoint.items():
    name = key.replace("module.", "") # remove `module.`
    new_state_dict[name] = v

# load params
model.load_state_dict(new_state_dict, strict=False)
print(model)
model=model.cuda()

d=check_accuracy(testloader, model)
print(d*100)