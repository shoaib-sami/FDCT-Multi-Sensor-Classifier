import torch
import torch.optim as optim
import torch.nn as nn
from hadi_dataloader import YourDataset
from my_model import model1
root_dir = "/home/shoaibmerajsami/Desktop/UCHE/PycharmProjects/abc/"
train_data = YourDataset(root_dir)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=200, shuffle=True,  num_workers=4)
model =model1
model = model.cuda()
if torch.cuda.is_available():
    model= nn.DataParallel(model, device_ids=[0,1])
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0002)
optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.92)


for epoch in range(800):
    avg_loss=0
    # for batch_idx, (data, targets) in enumerate(tqdm(trainloader)):
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        data, targets = data
        # data=data.to(device=device)
        # targets=targets.to(device=device)
        data = data.cuda()
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
        # print(targets)

        # backward

        loss.backward()

        # gradient descent or adam step
        optimizer.step()
    print("avg loss: {}".format(avg_loss))
    if epoch > 20 and epoch %10==2:
        scheduler.step()
    print("Epoch  {} | learning rate {}".format(epoch+ 1,
                                                scheduler.get_lr()))

    model_file = "Hadi_efficientNet_CBAM_%d_Epoch.pth" % (epoch + 1)
    print(model_file)
    if epoch % 10 == 0 and epoch >= 30:
        checkpoint = {}
        checkpoint["model"] = model.state_dict()
        checkpoint["optimizer"] = optimizer.state_dict()
        torch.save(checkpoint, model_file)
        print("saving model")

