import torch
#import torch.cuda as cuda
import torch.optim as optim
import torch.nn as nn
from hadi_dataloader import YourDataset
from my_resnetA import model1 as model
from may_val import test
#root_dir = "/home/shoaibmerajsami/Desktop/UCHE/PycharmProjects/abc/"
root_dir = '/media/shoaibmerajsami/SMS/ATR_database_october_2021/cycle_gan_july_2022_final/august_25th_2022/pytorch-CycleGAN-and-pix2pix-master/results/MtoV66_9_all_MWIR_train_Visible/test_latest/images/' #'/media/shoaibmerajsami/SMS/ATR_database_october_2021/cycle_gan_july_2022_final/MoNCE-main/CUT_MoNCE/results/MoNCE_M_V/test_latest/images/fake_B/'
YourDataset = YourDataset
train_data = YourDataset(root_dir)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=1200, shuffle=True,  num_workers=4)
#model = model
if torch.cuda.is_available():
    model= nn.DataParallel(model, device_ids=[0])
# Loss and optimizer
checkpoint = torch.load('/home/shoaibmerajsami/Desktop/atr hadi final/CycleGAN_classifier_Visible/Cycle_GAN_No_finetuining_October_6__5e_3_ResNet18_MWIR_weight_synthetic_image72_12_Epoch.pth') #('/media/shoaibmerajsami/SMS/ATR_database_october_2021/cycle_gan_july_2022_final/query-MoNCE/MWIR_66_image_ResNet18_6_Epoch_Accuracy_99_62.pth')
model.load_state_dict(checkpoint["model"])
print("NO NECESSARY File Loaded into ResNet")
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0002)
optimizer = optim.Adam(model.parameters(), lr=0.005, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

if torch.cuda.is_available():
    model.cuda()
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


        scores = model(data)
        # scores2=torch.stack(list(scores), dim=0)
        loss = criterion(scores, targets)
        avg_loss += loss.item()
        print(loss)
        #print(targets)

        # print(targets)

        # backward
        loss.backward()
        #print(i)
        if i % 200 == 90:
            print("Iteration is %d", i)

            if (epoch ==0 or epoch <= 5) and i>1:
                model_file = "Cycle_GAN_No_finetuneing_oct_8_1e_3_resNet18_Visible_%d_Epoch_1_2.pth" % (i+1+epoch)
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

    model_file = "Cycle_GAN_No_finetuining_October_8__1e_3_ResNet18_MWIR_weight_synthetic_image72_%d_Epoch.pth" % (epoch + 6)
    print(model_file)

    checkpoint1 = {}
    checkpoint1["model"] = model.state_dict()
    checkpoint1["optimizer"] = optimizer.state_dict()
    test(model, checkpoint1)

    torch.save(checkpoint1, model_file)
    print("saving model")

