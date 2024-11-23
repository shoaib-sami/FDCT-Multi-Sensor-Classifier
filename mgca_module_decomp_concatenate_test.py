import datetime
import os
from argparse import ArgumentParser
from UDT import ViT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from dateutil import tz
from einops import rearrange
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
#from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin
#from mgca.datasets.data_module import DataModule
#from mgca.datasets.pretrain_dataset import (MultimodalPretrainingDataset,multimodal_collate_fn)
#from mgca.datasets.transforms import DataTransforms

from encoder import  ImageEncoder
from torch import distributed as dist
from decomposition_loss import decomp
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#from sparsemax import Sparsemax




def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out
    
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)

# =============================================================================

# =============================================================================
import numbers
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads = heads[2])
        self.detailFeature = DetailFeatureExtraction()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_base = torch.nn.Conv2d(in_channels=64, out_channels=2,
            kernel_size=(1, 1))
        self.conv_detail=torch.nn.Conv2d(in_channels=64, out_channels=1,
            kernel_size=(1, 1))
             
    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature1 = self.baseFeature(out_enc_level1)
        base_feature = self.conv_base(base_feature1)

        detail_feature1 = self.detailFeature(out_enc_level1)
        detail_feature = self.conv_detail(detail_feature1)
        x=torch.cat((base_feature, detail_feature), dim=1)
        local_features = x
      
        return x.contiguous(),base_feature1,detail_feature1
        #base_feature, detail_feature, out_enc_level1
       

class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()              
    def forward(self, inp_img, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0
    
if __name__ == '__main__':
    height = 128
    width = 128
    window_size = 8
    modelE = Restormer_Encoder().to(torch.device('cuda:1'))
    modelD = Restormer_Decoder().to(torch.device('cuda:1'))



class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input
sparsemax = Sparsemax(dim=-1)
sparsemax2 = Sparsemax(dim=1)
from torch.nn import Conv2d
class MGCA(LightningModule):
    '''Pytorch lightning implementation of MGCA'''

    def __init__(self,
                 img_encoder: str = "vit_base",
                 freeze_bert: bool = False,
                 emb_dim: int = 768,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 2e-5,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 # TODO: tune this hyperparameter
                 local_temperature: float = 0.1,
                 proto_temperature: float = 0.2,
                 num_prototypes: int = 500,
                 bidirectional: bool = True,
                 use_local_atten: bool = False,
                 num_heads: int = 1,
                 lamb: float = 0.75,
                 lambda_1: float = 1,
                 lambda_2: float = 0.7,
                 lambda_3: float = 0.5,
                 freeze_prototypes_epochs: int = 1,
                 sinkhorn_iterations: int = 3,
                 epsilon: float = 0.05,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        # init encoders
        self.vit_cnn_feature1 = Restormer_Encoder()
        self.vit_cnn_feature2 = Restormer_Encoder()
        self.conv1 = torch.nn.Conv2d(in_channels=128, out_channels=3,
            kernel_size=(1, 1))
        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim)
        self.text_encoder_q = ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim)

        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)
        # sentence local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)

        self.prototype_layer = nn.Linear(emb_dim, num_prototypes, bias=False)

        self.get_assignments = self.sinkhorn
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(301056, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.decomp= decomp
        self.udt = ViT()
        self.relu = nn.ReLU()
    def forward(self, img1,img2):
        '''Forward step of our method'''

        # Forward of query image encoder
        img1,base1,detail1 = self.vit_cnn_feature1(img1)
        x1=torch.cat((base1, detail1), dim=1)
        #print("deatail")
        #print(detail1.shape)
        #print("base")
        #print(base1.shape)
        ##print(img1.shape)
        #img1=self.conv1(img1)
        img_feat_q, patch_feat_q = self.udt(x1)#self.img_encoder_q(img1)
        #print("Image Feature and Patch Feature")
        #print(img_feat_q.shape)
        #print(patch_feat_q.shape)

        #patch_emb_q = self.img_encoder_q.local_embed(patch_feat_q)
        patch_emb_q = F.normalize(patch_feat_q, dim=-1)
        #img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_feat_q, dim=-1)

        # Forward of query text encoder
        img2, base2, detail2 = self.vit_cnn_feature2(img2)
        x2=torch.cat((base2, detail2), dim=1)
        report_feat_q, word_feat_q = self.udt(x2)
        word_attn_q = word_feat_q
        #word_emb_q = self.img_encoder_q.local_embed(patch_feat_q)
        word_emb_q = F.normalize(word_feat_q, dim=-1)
        #report_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        report_emb_q = F.normalize(report_feat_q, dim=-1)
        patch_feat_q_flatten = self.flatten(patch_feat_q)
        report_emb_q_flatten = self.flatten(word_feat_q)
        my_feature_cat = torch.cat((patch_feat_q_flatten,report_emb_q_flatten),1)
        ##print(my_feature_cat.size())
        my_feature_cat_fc1 = self.relu(self.fc(my_feature_cat))
        my_feature_cat_fc = self.fc2(my_feature_cat_fc1)
        my_decompostion_loss = 1*self.decomp(base1,detail1,base2,detail2)
        bz = img_emb_q.size(0)
        labels = torch.arange(bz).type_as(report_emb_q).long()

        scores = img_emb_q.mm(report_emb_q.t())
        scores /= self.hparams.softmax_temperature
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, labels)
        loss1 = F.cross_entropy(scores1, labels)
        loss_ita = loss0 + loss1
        
        # compute retrieval accuracy
        i2t_acc1, i2t_acc5 = self.precision_at_k(
            scores, labels, top_k=(1, 1))
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            scores1, labels, top_k=(1, 1))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.
        
        ########### Token-level alignment ################
        # cross attention patch to sentences
        #acc1 = acc5 =0
   

        if self.hparams.use_local_atten:
            word_atten_output, _ = self.word_local_atten_layer(
                word_emb_q, patch_emb_q, patch_emb_q)
        else:
            atten_sim = torch.bmm(word_emb_q, patch_emb_q.permute(0, 2, 1))
            word_num = word_emb_q.size(1)
            # atten_sim[mask.unsqueeze(1).repeat(1, word_num, 1)] = float("-inf")
            atten_scores = sparsemax(
                atten_sim / self.hparams.local_temperature)  # bz, 196, 111
            word_atten_output = torch.bmm(atten_scores, patch_emb_q)

        word_atten_output = F.normalize(word_atten_output, dim=-1)

        word_sim = torch.bmm(
            word_emb_q, word_atten_output.permute(0, 2, 1)) / self.hparams.local_temperature



        loss_word = 0

        if self.hparams.bidirectional:
            # Try not use atten layer
            if self.hparams.use_local_atten:
                patch_atten_output, _ = self.patch_local_atten_layer(
                    patch_emb_q, word_emb_q, word_emb_q, key_padding_mask=mask)
            else:
                atten_sim = torch.bmm(patch_emb_q, word_emb_q.permute(0, 2, 1))
                patch_num = patch_emb_q.size(1)

                atten_scores = sparsemax(
                    atten_sim / self.hparams.local_temperature)  # bz, 196, 111
                patch_atten_output = torch.bmm(atten_scores, word_emb_q)

            # patch_atten_output: bz, 196, 128
            #if "vit" not in self.hparams.img_encoder:
            patch_atten_output = F.normalize(patch_atten_output, dim=-1)
            patch_num = patch_atten_output.size(1)
            patch_atten_weights = torch.ones(
                bz, patch_num) / patch_num
            patch_atten_weights=patch_atten_weights.to(torch.device('cuda:1'))

            # else:
            #     with torch.no_grad():
            #         img_attn_map = self.img_encoder_q.model.blocks[-1].attn.attention_map.detach(
            #         )
            #         atten_weights = img_attn_map[:, :, 0, 1:].mean(dim=1)
            #         patch_atten_weights = []
            #         for i in range(bz):
            #             atten_weight = atten_weights[i]
            #             atten_weight = atten_weight.clip(torch.quantile(
            #                 atten_weight, 0.1), torch.quantile(atten_weight, 0.9))
            #             patch_atten_weights.append(atten_weight.clone())
            #         patch_atten_weights = torch.stack(patch_atten_weights)

            #     patch_atten_weights /= patch_atten_weights.sum(
            #         dim=1, keepdims=True)

            patch_sim = torch.bmm(patch_emb_q, patch_atten_output.permute(
                0, 2, 1)) / self.hparams.local_temperature
            patch_num = patch_sim.size(1)
            patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
            targets = torch.arange(patch_num).type_as(
                patch_emb_q).long().repeat(bz)
            # loss_patch_1 = F.cross_entropy(patch_sim_1, targets)
            loss_patch_1 = torch.sum(F.cross_entropy(
                patch_sim_1, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

            patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
            loss_patch_2 = torch.sum(F.cross_entropy(
                patch_sim_2, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

            loss_patch = (loss_patch_1 + loss_patch_2) / 2.

            loss_local = loss_patch + loss_word

        else:

            loss_local = loss_word

        # normalize prototype layer
        with torch.no_grad():
            w = self.prototype_layer.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototype_layer.weight.copy_(w)

        # Compute assign code of images
        img_proto_out = self.prototype_layer(img_emb_q)
        report_proto_out = self.prototype_layer(report_emb_q)

        # TODO: define this to hparams
        with torch.no_grad():
            img_code = torch.exp(
                img_proto_out / self.hparams.epsilon).t()
            img_code = self.get_assignments(
                img_code, self.hparams.sinkhorn_iterations).to(torch.device('cuda:1'))         # bz, 500
            report_code = torch.exp(
                report_proto_out / self.hparams.epsilon).t()
            report_code = self.get_assignments(
                report_code, self.hparams.sinkhorn_iterations)       # bz, 500

        img_proto_prob = F.softmax(
            img_proto_out / self.hparams.proto_temperature,dim=1)
        report_proto_prob = F.softmax(
            report_proto_out / self.hparams.proto_temperature,dim=1)

        loss_i2t_proto = - \
            torch.mean(
                torch.sum(img_code * torch.log(report_proto_prob), dim=1))
        loss_t2i_proto = - \
            torch.mean(torch.sum(report_code *
                       torch.log(img_proto_prob), dim=1))

        loss_proto = (loss_i2t_proto + loss_t2i_proto) / 2.
        ##print(loss_proto)
        ##print(self.softmax(scores))

        return loss_ita, loss_local, loss_proto, my_decompostion_loss, acc1, acc5, my_feature_cat_fc

    def sinkhorn(self, Q, nmb_iters):
        ''' 
            :param Q: (num_prototypes, batch size)

        '''
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape
            K = K
            B=B


            u = torch.zeros(K).to(torch.device('cuda:1'))
            r = torch.ones(K).to(torch.device('cuda:1')) / K
            c = torch.ones(B).to(torch.device('cuda:1')) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            if self.hparams.gpus > 0:
                u = torch.zeros(Q.shape[0]).to(torch.device('cuda:1'))(non_blocking=True)
                r = torch.ones(Q.shape[0]).to(torch.device('cuda:1'))(non_blocking=True) / Q.shape[0]
                c = torch.ones(Q.shape[1]).to(torch.device('cuda:1'))(
                    non_blocking=True) / (self.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def training_step(self, batch, batch_idx):
        loss_ita, loss_local, loss_proto, acc1, acc5 = self(
            batch, batch_idx, "train")
        loss = self.hparams.lambda_1 * loss_ita + self.hparams.lambda_2 * \
            loss_local + self.hparams.lambda_3 * loss_proto

        log = {
            "train_loss": loss,
            "train_loss_ita": self.hparams.lambda_1 * loss_ita,
            "train_loss_local": self.hparams.lambda_2 * loss_local,
            "train_loss_proto": self.hparams.lambda_3 * loss_proto,
            "train_acc1": acc1,
            "train_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)

        return loss

    # freeze prototype layer
    def on_after_backward(self):
        if self.current_epoch < self.hparams.freeze_prototypes_epochs:
            for param in self.prototype_layer.parameters():
                param.grad = None

    def validation_step(self, batch, batch_idx):
        loss_ita, loss_local, loss_proto, acc1, acc5 = self(
            batch, batch_idx, "valid")

        loss = self.hparams.lambda_1 * loss_ita + self.hparams.lambda_2 * \
            loss_local + self.hparams.lambda_3 * loss_proto

        log = {
            "val_loss": loss,
            "val_loss_ita": self.hparams.lambda_1 * loss_ita,
            "val_loss_local": self.hparams.lambda_2 * loss_local,
            "val_loss_proto": self.hparams.lambda_3 * loss_proto,
            "val_acc1": acc1,
            "val_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss

    # def on_train_epoch_end(self):
    #     ''' Save img_queue and report_queue for visualization '''
    #     if self.local_rank == 0:
    #         img_queue_path = f"{self.trainer.callbacks[-1].dirpath}/img_queue.pth"
    #         torch.save(self.img_queue, img_queue_path)
    #         report_queue_path = f"{self.trainer.callbacks[-1].dirpath}/report_queue.pth"
    #         torch.save(self.report_queue, report_queue_path)

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=1e-8,
            warmup_steps=int(self.training_steps * 0.4)
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--img_encoder", type=str, default="vit_base")
        parser.add_argument("--freeze_bert", action="store_true")
        parser.add_argument("--emb_dim", type=int,
                            default=128, help="128, 256")
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=72)
        parser.add_argument("--num_prototypes", type=int, default=500)
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--lambda_1", type=float, default=1.)
        parser.add_argument("--lambda_2", type=float, default=1.)
        parser.add_argument("--lambda_3", type=float, default=1.)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--bidirectional", action="store_false")
        parser.add_argument("--data_pct", type=float, default=1.)
        return parser

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, (DDPPlugin, DDP2Plugin))
        else:
            return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs


@torch.no_grad()
def concat_all_gather(tensor):
    '''
    Performs all_gather operation on the provided tensors
    '''
    tensors_gather = [torch.ones_like(tensor) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def cli_main():
    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)
    # model args
    parser = MGCA.add_model_specific_args(parser)
    args = parser.parse_args()

    args.deterministic = True
    args.max_epochs = 50

    # seed
    seed_everything(args.seed)

    datamodule = DataModule(MultimodalPretrainingDataset, multimodal_collate_fn,
                            DataTransforms, args.data_pct,
                            args.batch_size, args.num_workers)

    # Add load from checkpoint
    model = MGCA(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/MGCA/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=5),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=5, verbose=False, mode="min")
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"../../../data")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="MGCA", save_dir=logger_dir, name=extension)
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=wandb_logger)

    model.training_steps = model.num_training_steps(trainer, datamodule)
    ##print(model.training_steps)
    trainer.fit(model, datamodule=datamodule)

    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)


if __name__ == "__main__":
    a1= torch.rand(5,3,224,224)
    a2= torch.rand(5,3,224,224)
    a1 = a1.to(torch.device('cuda:1'))
    a2 = a2.to(torch.device('cuda:1'))
    mg = MGCA().to(torch.device('cuda:1'))
    a,b,c,cc,d,e,f =mg(a1,a2)
    #print(a,b,c,cc,d,e,f)
