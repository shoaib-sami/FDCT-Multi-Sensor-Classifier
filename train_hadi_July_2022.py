import torch
#import torch.cuda as cuda
import torch.optim as optim
import torch.nn as nn
from hadi_dataloader_multi_modal import YourDataset
from my_resnet import model
from may_val import test
#root_dir = "/home/shoaibmerajsami/Desktop/UCHE/PycharmProjects/abc/"
root_dir = '/home/shoaibmerajsami/Desktop/atr hadi final/Visible_NVESD/train/'
train_data = YourDataset(root_dir)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=117, shuffle=True,  num_workers=4)

model = model
if torch.cuda.is_available():
    model= nn.DataParallel(model, device_ids=[0,1])
# Loss and optimizer
model.load_state_dict(torch.load('Accuracy_99.45_ResNet50_image72_11_Epoch.pth')["model"])
#model.load_state_dict(checkpoint["model"])
print("File Loaded into ResNet")
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0002)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


for epoch in range(800):
    avg_loss=0
    # for batch_idx, (data, targets) in enumerate(tqdm(trainloader)):
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        data1, data2, targets = data
        # data=data.to(device=device)
        # targets=targets.to(device=device)
        data = data2.cuda()
        targets = targets.cuda()
        optimizer.zero_grad()
        # print('a')
        # Get data to cuda if possible
        # data = data.to(device=device)
        # targets = targets.to(device=device)

        if torch.cuda.is_available():
            model.cuda()
        scores = model(data)
        # scores2=torch.stack(list(scores), dim=0)
        loss = criterion(scores, targets)
        avg_loss += loss.item()
        print(avg_loss/((i+1)*117))
        #print(targets)

        # print(targets)

        # backward
        loss.backward()
        #print(i)
        if i % 300 == 0:
            print("Iteration is %d", i)

            if (epoch ==0 or epoch == 1) and i>1:
                model_file = "ResNet50_Visible_%d_Epoch_1_2.pth" % (i+1+epoch)
                print(model_file)

                checkpoint1 = {}
                checkpoint1["model"] = model.state_dict()
                checkpoint1["optimizer"] = optimizer.state_dict()
                test(model, checkpoint1)

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
    checkpoint1["model"] = model.state_dict()
    checkpoint1["optimizer"] = optimizer.state_dict()
    test(model, checkpoint1)

    torch.save(checkpoint1, model_file)
    print("saving model")

