import torch.nn as nn
import torch
from backbones import get_model
import net

import torch
from torch import nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from magface import iresnet100 as mag_iresnet100

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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
arcface = get_model(
    "r100", dropout=0.0, fp16=True, num_features=512).cuda()

arcface.load_state_dict(torch.load('/home/nasser/Downloads/insightface-002/insightface/recognition/arcface_torch/backbone.pth'))


def load_pretrained_model(model_name='ir50'):

    adaface_models = {
        'ir50': ["../pretrained/adaface_ir50_ms1mv2.ckpt", 'ir_50'],
        'ir101_ms1mv2': ["/home/nasser/Downloads/insightface-002/insightface/adaface_ir101_ms1mv2.ckpt", 'ir_101'],
        'ir101_ms1mv3': ["../pretrained/adaface_ir101_ms1mv3.ckpt", 'ir_101'],
        'ir101_webface4m': ["../pretrained/adaface_ir101_webface4m.ckpt", 'ir_101'],
        'ir101_webface12m': ["../pretrained/adaface_ir101_webface12m.ckpt", 'ir_101']}

    # load model and pretrained statedict
    ckpt_path = adaface_models[model_name][0]
    arch = adaface_models[model_name][1]

    model = net.build_model(arch)
    statedict = torch.load(ckpt_path)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

adaface_model = load_pretrained_model(model_name='ir101_ms1mv2').cuda()

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
        num_patches = 98*2
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



class FusionModel(torch.nn.Module):

    def __init__(self):
        super(FusionModel, self).__init__()

        arcface = get_model(
            "r100", dropout=0.0, fp16=True, num_features=512).cuda()

        arcface.load_state_dict(torch.load('/home/nasser/Downloads/insightface-002/insightface/recognition/arcface_torch/backbone.pth'))

        adaface_model = load_pretrained_model(model_name='ir101_ms1mv2').cuda()

        self.arcface = arcface
        #mag_iresnet100.load_state_dict(torch.load('/home/nasser/Downloads/magface_epoch_00025.pth'))
        #mag_iresnet100().load_state_dict(torch.load('/home/nasser/Downloads/magface_epoch_00025.pth'))
        print(mag_iresnet100())
        self.magface = mag_iresnet100
        for param in mag_iresnet100.parameters():
            param.requires_grad = False

        for param in arcface.parameters():
            param.requires_grad = False

        self.adaface_model = adaface_model

        for param in adaface_model.parameters():
            param.requires_grad = False

        self.arcface.dropout = torch.nn.Identity()
        self.arcface.fc = torch.nn.Identity()
        self.arcface.features = torch.nn.Identity()

        self.adaface_model.output_layer[-1] = torch.nn.Identity()
        self.adaface_model.output_layer[-2] = torch.nn.Identity()
        self.adaface_model.output_layer[-3] = torch.nn.Identity()

        self.permute = [2, 1, 0]

        self.flatten_layer = torch.nn.Flatten()
        self.dropout_layer = torch.nn.Dropout()

        self.linear_1 = torch.nn.Linear(in_features=512*(98*2+1), out_features=512).requires_grad_(True)

        self.batchnorm1d_layer = torch.nn.BatchNorm1d(512, momentum=0.1, affine=False,
                                                      track_running_stats=True).requires_grad_(True)

        self.cross_attention = CrossAttention(512, num_heads=8).requires_grad_(True)

    def forward(self, x):

        arc_out = self.arcface(x)
        print(self.magface())

        y = x[:, self.permute]

        ada_out = self.adaface_model(y)

        arc_out = torch.reshape(arc_out, (arc_out.shape[0], 512, 7, 7))

        out_fusion = self.cross_attention(ada_out, arc_out)

        #concat_features = torch.cat((feature1, feature2), dim = 2)

        out_fusion = self.dropout_layer(out_fusion)
        out_fusion = self.flatten_layer(out_fusion)
        out_fusion = self.linear_1(out_fusion)

        norm = torch.norm(out_fusion, 2, 1, True)
        output = torch.div(out_fusion, norm)

        return output





