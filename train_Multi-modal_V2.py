import torch
#import torch.cuda as cuda
import torch.optim as optim
import torch.nn as nn
from hadi_dataloader_multi_modal_train import YourDataset
from resnet_visible import model_visible
from resnet_mwir import model_mwir
from may_val_pair_unpair import test



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



# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 3, dim_head = 64, dropout = 0.): #head was 8
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias=False):
        super(CrossAttention, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.kvq1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kvq2 = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.num_heads = num_heads
        dim = 512
        depth = 6
        heads = 16 #it was 16
        mlp_dim = 2048
        dropout = 0.1
        emb_dropout = 0.1
        num_patches = 81*2
        dim_head = 64
        num_classes =512
        batch = 768
        """
        self.patch_embedding_my = nn.Sequential(
            nn.Linear(512,2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048,512),
            nn.Dropout(dropout)
            
            ) 
        """
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        #self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x1, x2):

        x1 = x1.flatten(2).transpose(1, 2)
        
        x2 = x2.flatten(2).transpose(1, 2)

        x = torch.cat((x1,x2),dim=1)
        
        #print(x.size())
        #print(x.size())
        B, N, C = x.shape
        b = B
        d =512
        n = N
        cls_tokens1 = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        
        x = torch.cat((cls_tokens1, x), dim=1)
        
        self.norm = nn.LayerNorm((b, n+1,512)).cuda()
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        #x = self.patch_embedding_my(x)
        x = self.transformer(x)
        x = self.to_latent(x)
        #x = x[:, 0]
        x =self.norm(x)

        return x


#root_dir = "/home/shoaibmerajsami/Desktop/UCHE/PycharmProjects/abc/"
root_dir = '/home/shoaibmerajsami/Desktop/atr hadi final/Visible_NVESD/train/'
train_data = YourDataset(root_dir)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=80, shuffle=True,  num_workers=4)
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
        self.model_mwir.load_state_dict(torch.load('/media/shoaibmerajsami/SMS/ATR_database_october_2021/cycle_gan_july_2022_final/Modified_CUT_sep12/MWIR_66_image_ResNet18_6_Epoch_Accuracy_99_62.pth')["model"],strict= False)
        self.model_visible.load_state_dict(torch.load('/media/shoaibmerajsami/SMS/ATR_database_october_2021/cycle_gan_july_2022_final/Modified_CUT_sep12/Visible_66_image_ResNet18_5_Epoch99_28.pth')["model"],strict= False)
        self.cross_attention = CrossAttention(512, num_heads=4).requires_grad_(True)
        self.flatten_layer = torch.nn.Flatten()
        self.dropout_layer = torch.nn.Dropout()
        self.fusion = nn.Sequential(
            nn.Linear(512*(81*2+1), 512),
            nn.ReLU(),
            nn.Linear(512, 10)
            )

        self.linear_1 = torch.nn.Linear(in_features=512*(81*2+1), out_features=10).requires_grad_(True)
        self.batchnorm1d_layer = torch.nn.BatchNorm1d(512, momentum=0.1, affine=False,
                                                      track_running_stats=True).requires_grad_(True)



        self.cross_attention = CrossAttention(512, num_heads=8).requires_grad_(True)
    def forward(self,mwir_data, visible_data):
        output_vis = self.model_visible(visible_data)

        output_mwir = self.model_mwir(mwir_data)
        out_fusion = self.cross_attention(output_vis, output_mwir)

        out_fusion = self.dropout_layer(out_fusion)
        out_fusion = self.flatten_layer(out_fusion)
        out_fusion = self.fusion(out_fusion)

        


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

f_model = Fusion_model().cuda()
#f_model = nn.DataParallel(f_model, device_ids=[0,1]).cuda()


# Loss and optimizer
#f_model.load_state_dict(torch.load('/home/shoaibmerajsami/Desktop/atr hadi final/Multi_domain_fusion/ResNet50_image72_6_Epoch.pth')["model"])
#model.load_state_dict(checkpoint["model"])
print("File Loaded into ResNet")
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0002)
optimizer = optim.Adam(f_model.parameters(), lr=0.0001, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)




for epoch in range(800):
    avg_loss=0
    # for batch_idx, (data, targets) in enumerate(tqdm(trainloader)):
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        visible_data, mwir_data, pair_unpair, targets = data
        # data=data.to(device=device)
        # targets=targets.to(device=device)
        #print(len(visible_data))
        vis=[]
        mwir =[]
        lb =[]
        #print(len(pair_unpair))
        for x in range(len(pair_unpair)):
            if pair_unpair[x] == 'unpair':
                pass
            else:
                vis.append(visible_data[x])
                mwir.append(mwir_data[x])
                lb.append(targets[x])

        vis1 =torch.stack(vis,dim=0)
        mwir1 =torch.stack(mwir,dim=0)
        lb1 =torch.stack(lb,dim=0)
        #print(vis1.size())
        #print(len(visible_data))
        vis1 = vis1.cuda()
        mwir1 = mwir1.cuda()
        lb1 = lb1.cuda()
        #pair_unpair = pair_unpair.cuda()
        optimizer.zero_grad()
        # print('a')
        # Get data to cuda if possible
        # data = data.to(device=device)
        # targets = targets.to(device=device)


        scores = f_model(vis1,mwir1)
        # scores2=torch.stack(list(scores), dim=0)
        loss = criterion(scores, lb1)
        avg_loss += loss.item()
        #print(avg_loss/((i+1)*60))
        #print(targets)

        # print(targets)

        # backward
        loss.backward()
        #print(i)
        if i % 1000 == 820:
            print("Iteration is %d", i)

            if (epoch ==0 or epoch == 1) and i>1:
                model_file = "ResNet50_Visible_%d_Epoch_1_2.pth" % (i+1+epoch)
                print(model_file)

                checkpoint1 = {}
                checkpoint1["model"] = f_model.state_dict()
                checkpoint1["optimizer"] = optimizer.state_dict()
                test(f_model, checkpoint1)

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
    checkpoint1["model"] = f_model.state_dict()
    checkpoint1["optimizer"] = optimizer.state_dict()
    test(f_model, checkpoint1)

    torch.save(checkpoint1, model_file)
    print("saving model")

