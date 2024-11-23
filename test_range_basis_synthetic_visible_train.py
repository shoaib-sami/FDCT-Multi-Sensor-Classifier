import torch
#import torch.cuda as cuda
import torch.optim as optim
import torch.nn as nn
from hadi_dataloader import YourDataset
from my_resnetA import model1
import config
#root_dir = "/home/shoaibmerajsami/Desktop/UCHE/PycharmProjects/abc/"
from sklearn.metrics import confusion_matrix
import numpy
model = model1
if torch.cuda.is_available():
    model= nn.DataParallel(model, device_ids=[0,1])



def test(model):
    root_dir = '/home/shoaibmerajsami/Desktop/atr hadi final/66_chip/Synthetic_MWIR_sept19/' #'/home/shoaibmerajsami/Desktop/atr hadi final/66_chip/mwir/mwir_ttv_66_chip/test_distance_basis/5000/'#/home/shoaibmerajsami/Desktop/atr hadi final/cycle_gan_generated_image/August_10_genA_vis/'#"/home/shoaibmerajsami/Desktop/atr hadi final/cycle_gan_generated_image/July_31_gen_A_no_Cls/mwToGenA_vis_val_set_41k/GenA/"
    #/home/shoaibmerajsami/Desktop/atr hadi final/cycle_gan_generated_image/gen_B #'/home/shoaibmerajsami/Desktop/atr hadi final/MW_NVESD_ranged_basis/Test_range_5000/'
    print(root_dir)
    train_data = YourDataset(root_dir)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=250, shuffle=True, num_workers=4)
    #model.load_state_dict(checkpoint["model"])
    print("I am in test Function")
    num_correct = 0
    num_samples = 0
    accuracy = 0
    predictions = 0
    ls_sq_dist = []
    ls_sq_dist2 = []
    ls_labels = []
    my_label =[]
    my_prediction = []
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

        #done by SMS
        my_label.append(label.data)
        my_prediction.append(predictions.data)

        # my_utility_2_class_morph_real.calculate_scores(ls_labels, ls_sq_dist)

    #a, b, c, auc, eer = my_utility_apcer.calculate_scores(ls_labels, ls_sq_dist)
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

    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    target_names = ['2S3','BMP2','BRDM2','BTR70','D20','MTLB','Pickup','Sport_Vechicle','T72','ZSU23-4']
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
    plt.title('Synthetic Visible Images Classification')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('Synthetic_visible.png', bbox_inches='tight')
    plt.show(block=False)
    fig1, ax1 = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.3f', xticklabels=target_names, yticklabels=target_names)
    plt.title('Synthetic Visible Images Classification (Normalized Confusion Matrix)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('Normalized_Synthetic_visible.png', bbox_inches='tight')
    plt.show(block=False)
    #f, axarr = plt.subplots(1,2)
    return accuracy


#model_file = "Accuracy_99.45_ResNet50_image72_11_Epoch.pth"
#file = '/home/shoaibmerajsami/Desktop/atr hadi final/CycleGan_classifier_mw/Accuracy_99.45_ResNet50_image72_11_Epoch.pth'

# Loss and optimizer
model.load_state_dict(torch.load('/media/shoaibmerajsami/SMS/ATR_database_october_2021/cycle_gan_july_2022_final/visible_mwir_cyclegan/cyclegan/models/MWIR_66_image_ResNet18_6_Epoch_Accuracy_99_62.pth')["model"])
model = model.cuda()

test(model)




