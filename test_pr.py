from may_val_pair_unpair_PR import test
import torch
import os
import config
#from resnet_my import resnet_abc
#f_model = resnet_abc.to(config.device)# MGCA().to(config.device)#Fusion_model().cuda()

def bismillah(abc):
	checkpoint1 = torch.load(abc)
	#checkpoint1["model"] = f_model.state_dict()
	#checkpoint1["optimizer"] = optimizer.state_dict()
	test(checkpoint1)




def list_files_in_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list



file_path = '/home/shoaibmerajsami/Desktop/atr hadi final/DSIAC_multi_modal_MGCA_final_april_2024/best_decomp_UDT_99_29.pth'
print(file_path)
bismillah(file_path)