import torch, torchvision

from torchvision.transforms import ToTensor, ToPILImage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import Image
import PIL
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import random
import torch.nn.functional as F
import tarfile
import io
import os
import pandas as pd
import cv2
import config
from torch.utils.data import Dataset
import torch
import re

class YourDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = '/home/shoaibmerajsami/Desktop/atr hadi final/Multi_domain_fusion/dataset/test_vis/'

        #self.img_dir2 = img_dir
        
        #self.transform = transform
        self.labels = sorted(os.listdir(self.img_dir), key=lambda x: int(x))

        lb = [int(l) - 1 for l in self.labels]
        self.labels_ohe = lb
        # self.labels_ohe = F.one_hot(torch.as_tensor(lb), num_classes=11) # I have changed from 11 to 5 sami

        self.img_lists = []
        self.all_class_dirs = [os.path.join(self.img_dir, label) for label in self.labels]

        for class_dir in self.all_class_dirs[:10]:  # For 5 classes
            self.img_lists += os.listdir(class_dir)

        """

        self.labels2 = sorted(os.listdir(self.img_dir2), key=lambda x: int(x))

        lb2 = [int(l) - 1 for l in self.labels2]
        self.labels_ohe2 = lb2
        # self.labels_ohe = F.one_hot(torch.as_tensor(lb), num_classes=11) # I have changed from 11 to 5 sami

        self.img_lists2 = []
        self.all_class_dirs2 = [os.path.join(self.img_dir2, label2) for label in self.labels2]

        for class_dir2 in self.all_class_dirs2[:10]:  # For 5 classes
            self.img_lists2 += os.listdir(class_dir2)
        """
        
        self.i = 0
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            #transforms.CenterCrop(64),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: x.repeat(3,1,1)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225] )
        ])

    def __len__(self):
        #return 1000
        return len(self.img_lists)

    def __getitem__(self, index):
        all_img_abs_dir = []
        for class_dir in self.all_class_dirs[:10]:  # for 5 classes
            all_img_abs_dir += [os.path.join(class_dir, img_name) for img_name in os.listdir(class_dir)]

        image_abs_dir = all_img_abs_dir[index]
        #img_train_mwir ='/home/shoaibmerajsami/Desktop/atr hadi final/MW_NVESD/train/'
        label = int(image_abs_dir.split("/")[-2])
        image_name_mwir = image_abs_dir.split("/")[-1]
        image_specific = image_name_mwir.split("_")[-3]
        
        image_abs_dir_mwir = image_abs_dir.replace('test_vis','test_mwir')
        if image_specific=='i1co02003':
            #re.sub(r'\bi1co02003\b', 'cegr01923', image_abs_dir_mwir)
            image_abs_dir_mwir=image_abs_dir_mwir.replace('i1co02003','cegr01923')   


        if image_specific=='i1co02005':
            #re.sub(r'\bi1co02005\b', 'cegr01925', image_abs_dir_mwir)
            image_abs_dir_mwir=image_abs_dir_mwir.replace('i1co02005','cegr01925')



        if image_specific=='i1co02007':
            #re.sub(r'\bi1co02007\b', 'cegr01927', image_abs_dir_mwir)
            image_abs_dir_mwir=image_abs_dir_mwir.replace('i1co02007','cegr01927')             
       
        if image_specific=='i1co02009':
           
            #re.sub(r'\bi1co02009\b', 'cegr01929', image_abs_dir_mwir)
            image_abs_dir_mwir=image_abs_dir_mwir.replace('i1co02009','cegr01929')


        if image_specific=='i1co02011':
            #re.sub(r'\bi1co02011\b', 'cegr01931', image_abs_dir_mwir)
            image_abs_dir_mwir=image_abs_dir_mwir.replace('i1co02011','cegr01931')              

           
        if image_specific=='i1co02013':
            #re.sub(r'\bi1co02013\b', 'cegr01933', image_abs_dir_mwir)
            image_abs_dir_mwir=image_abs_dir_mwir.replace('i1co02013','cegr01933')

            

        if image_specific=='i1co02015':
            image_abs_dir_mwir=image_abs_dir_mwir.replace('i1co02015','cegr01935')
            #re.sub(r'\bi1co02015\b', 'cegr01935', image_abs_dir_mwir)
  
                   
        if image_specific=='i1co02017':
            image_abs_dir_mwir=image_abs_dir_mwir.replace('i1co02017','cegr01937')
            #re.sub(r'\bi1co02017\b', 'cegr01937', image_abs_dir_mwir)              
          

        if image_specific=='i1co02019':
            image_abs_dir_mwir=image_abs_dir_mwir.replace('i1co02019','cegr01939')
            #re.sub(r'\bi1co02019\b', 'cegr01939', image_abs_dir_mwir)            

              
       
        if os.path.exists(image_abs_dir_mwir):
            pass
        else:
            aaa = image_abs_dir_mwir.split(".")[-2]
            aaa2 = aaa.split("_")[-1]
            aaa2_string = aaa2 + '.png'
            aaa3 =str(int(aaa2)+1)+'.png'
            image_abs_dir_mwir = image_abs_dir_mwir.replace(aaa2_string,aaa3)
            
            #print("Hello I have converted String")

            
            
            if os.path.exists(image_abs_dir_mwir):
                pass
            else:
                aaa = image_abs_dir_mwir.split(".")[-2]
                aaa2 = aaa.split("_")[-1]
                aaa2_string = aaa2 + '.png'
                aaa3 =str(int(aaa2)+1)+'.png'
                image_abs_dir_mwir = image_abs_dir_mwir.replace(aaa2_string,aaa3)
                
                #print("Hello I have converted String 2nd time``````````````````````'''''''")            
                if os.path.exists(image_abs_dir_mwir):
                    pass
                else:
                    aaa = image_abs_dir_mwir.split(".")[-2]
                    aaa2 = aaa.split("_")[-1]
                    aaa2_string = aaa2 + '.png'
                    aaa3 =str(int(aaa2)+1)+'.png'
                    image_abs_dir_mwir = image_abs_dir_mwir.replace(aaa2_string,aaa3)
                    
                    #print("Hello I have converted String 3rd time``````````````````````'''''''")            
                    if os.path.exists(image_abs_dir_mwir):
                        pass
                    else:
                        aaa = image_abs_dir_mwir.split(".")[-2]
                        aaa2 = aaa.split("_")[-1]
                        aaa2_string = aaa2 + '.png'
                        aaa3 =str(int(aaa2)+1)+'.png'
                        image_abs_dir_mwir = image_abs_dir_mwir.replace(aaa2_string,aaa3)
                        
                        #print("Hello I have converted String 4th time``````````````````````'''''''")            
                        if os.path.exists(image_abs_dir_mwir):
                            pass
                        else:
                            aaa = image_abs_dir_mwir.split(".")[-2]
                            aaa2 = aaa.split("_")[-1]
                            aaa2_string = aaa2 + '.png'
                            aaa3 =str(int(aaa2)+1)+'.png'
                            image_abs_dir_mwir = image_abs_dir_mwir.replace(aaa2_string,aaa3)
                            
                            #print("Hello I have converted String 5th time``````````````````````'''''''")


                            if os.path.exists(image_abs_dir_mwir):
                                pass
                            else:
                                aaa = image_abs_dir_mwir.split(".")[-2]
                                aaa2 = aaa.split("_")[-1]
                                aaa2_string = aaa2 + '.png'
                                aaa3 =str(int(aaa2)+1)+'.png'
                                image_abs_dir_mwir = image_abs_dir_mwir.replace(aaa2_string,aaa3)
                                
                                #print("Hello I have converted String 2nd time``````````````````````'''''''")            
                                if os.path.exists(image_abs_dir_mwir):
                                    pass
                                else:
                                    aaa = image_abs_dir_mwir.split(".")[-2]
                                    aaa2 = aaa.split("_")[-1]
                                    aaa2_string = aaa2 + '.png'
                                    aaa3 =str(int(aaa2)+1)+'.png'
                                    image_abs_dir_mwir = image_abs_dir_mwir.replace(aaa2_string,aaa3)
                                    
                                    #print("Hello I have converted String 3rd time``````````````````````'''''''")            
                                    if os.path.exists(image_abs_dir_mwir):
                                        pass
                                    else:
                                        aaa = image_abs_dir_mwir.split(".")[-2]
                                        aaa2 = aaa.split("_")[-1]
                                        aaa2_string = aaa2 + '.png'
                                        aaa3 =str(int(aaa2)+1)+'.png'
                                        image_abs_dir_mwir = image_abs_dir_mwir.replace(aaa2_string,aaa3)
                                        
                                        #print("Hello I have converted String 4th time``````````````````````'''''''")            
                                        if os.path.exists(image_abs_dir_mwir):
                                            pass
                                        else:
                                            aaa = image_abs_dir_mwir.split(".")[-2]
                                            aaa2 = aaa.split("_")[-1]
                                            aaa2_string = aaa2 + '.png'
                                            aaa3 =str(int(aaa2)+1)+'.png'
                                            image_abs_dir_mwir = image_abs_dir_mwir.replace(aaa2_string,aaa3)
                                            
                                            #print("Hello I have converted String 5th time``````````````````````'''''''")

                                            if os.path.exists(image_abs_dir_mwir):
                                                pass
                                            else:
                                                aaa = image_abs_dir_mwir.split(".")[-2]
                                                aaa2 = aaa.split("_")[-1]
                                                aaa2_string = aaa2 + '.png'
                                                aaa3 =str(int(aaa2)+1)+'.png'
                                                image_abs_dir_mwir = image_abs_dir_mwir.replace(aaa2_string,aaa3)
                                                
                                                #print("Hello I have converted String")

                                                
                                                
                                                if os.path.exists(image_abs_dir_mwir):
                                                    pass
                                                else:
                                                    aaa = image_abs_dir_mwir.split(".")[-2]
                                                    aaa2 = aaa.split("_")[-1]
                                                    aaa2_string = aaa2 + '.png'
                                                    aaa3 =str(int(aaa2)+1)+'.png'
                                                    image_abs_dir_mwir = image_abs_dir_mwir.replace(aaa2_string,aaa3)
                                                    
                                                    #print("Hello I have converted String 2nd time``````````````````````'''''''")            
                                                    if os.path.exists(image_abs_dir_mwir):
                                                        pass
                                                    else:
                                                        aaa = image_abs_dir_mwir.split(".")[-2]
                                                        aaa2 = aaa.split("_")[-1]
                                                        aaa2_string = aaa2 + '.png'
                                                        aaa3 =str(int(aaa2)+1)+'.png'
                                                        image_abs_dir_mwir = image_abs_dir_mwir.replace(aaa2_string,aaa3)
                                                        
                                                        #print("Hello I have converted String 3rd time``````````````````````'''''''")            
                                                        if os.path.exists(image_abs_dir_mwir):
                                                            pass
                                                        else:
                                                            aaa = image_abs_dir_mwir.split(".")[-2]
                                                            aaa2 = aaa.split("_")[-1]
                                                            aaa2_string = aaa2 + '.png'
                                                            aaa3 =str(int(aaa2)+1)+'.png'
                                                            image_abs_dir_mwir = image_abs_dir_mwir.replace(aaa2_string,aaa3)
                                                            
                                                            #print("Hello I have converted String 4th time``````````````````````'''''''")            
                                                            if os.path.exists(image_abs_dir_mwir):
                                                                pass
                                                            else:
                                                                aaa = image_abs_dir_mwir.split(".")[-2]
                                                                aaa2 = aaa.split("_")[-1]
                                                                aaa2_string = aaa2 + '.png'
                                                                aaa3 =str(int(aaa2)+1)+'.png'
                                                                image_abs_dir_mwir = image_abs_dir_mwir.replace(aaa2_string,aaa3)
                                                                
                                                                #print("Hello I have converted String 5th time``````````````````````'''''''")


                                                                if os.path.exists(image_abs_dir_mwir):
                                                                    pass
                                                                else:
                                                                    aaa = image_abs_dir_mwir.split(".")[-2]
                                                                    aaa2 = aaa.split("_")[-1]
                                                                    aaa2_string = aaa2 + '.png'
                                                                    aaa3 =str(int(aaa2)+1)+'.png'
                                                                    image_abs_dir_mwir = image_abs_dir_mwir.replace(aaa2_string,aaa3)
                                                                    
                                                                    #print("Hello I have converted String 2nd time``````````````````````'''''''")            
                                                                    if os.path.exists(image_abs_dir_mwir):
                                                                        pass
                                                                    else:
                                                                        aaa = image_abs_dir_mwir.split(".")[-2]
                                                                        aaa2 = aaa.split("_")[-1]
                                                                        aaa2_string = aaa2 + '.png'
                                                                        aaa3 =str(int(aaa2)+1)+'.png'
                                                                        image_abs_dir_mwir = image_abs_dir_mwir.replace(aaa2_string,aaa3)
                                                                        
                                                                        #print("Hello I have converted String 3rd time``````````````````````'''''''")            
                                                                        if os.path.exists(image_abs_dir_mwir):
                                                                            pass
                                                                        else:
                                                                            aaa = image_abs_dir_mwir.split(".")[-2]
                                                                            aaa2 = aaa.split("_")[-1]
                                                                            aaa2_string = aaa2 + '.png'
                                                                            aaa3 =str(int(aaa2)+1)+'.png'
                                                                            image_abs_dir_mwir = image_abs_dir_mwir.replace(aaa2_string,aaa3)
                                                                            
                                                                            #print("Hello I have converted String 4th time``````````````````````'''''''")            
                                                                            if os.path.exists(image_abs_dir_mwir):
                                                                                pass
                                                                            else:
                                                                                aaa = image_abs_dir_mwir.split(".")[-2]
                                                                                aaa2 = aaa.split("_")[-1]
                                                                                aaa2_string = aaa2 + '.png'
                                                                                aaa3 =str(int(aaa2)+1)+'.png'
                                                                                image_abs_dir_mwir = image_abs_dir_mwir.replace(aaa2_string,aaa3)
                                                                                
                                                                                #print("Hello I have converted String 5th time``````````````````````'''''''")
                                
                
                
                
                
                



   

        try:
            img = Image.open(image_abs_dir).convert("RGB") #.convert("L")
            #img = img
            img1 = self.transform(img)
            if os.path.exists(image_abs_dir_mwir):
                pair_unpair = 'pair'
            else:
                image_abs_dir_mwir = image_abs_dir
                pair_unpair = 'unpair'
                self.i+=1
                if self.i % 500 ==1:
                    print(" What happen")
                    print(self.i)

            img_mwir = Image.open(image_abs_dir_mwir).convert("RGB")
            img_mwir2 =self.transform(img_mwir)
           
            #x.unsqueeze_(0) # done august 16 2022 by Shoaib
            #img = img.repeat(1, 3, 1, 1)   # done august 16 2022
            #x = x.view(-1, 72, 72)
            # vis = np.concatenate((img, img, img), axis=1)
            return img1,img_mwir2,pair_unpair, self.labels_ohe[label - 1]

        except PIL.UnidentifiedImageError as ui:
            #print(image_abs_dir)
            return None, None