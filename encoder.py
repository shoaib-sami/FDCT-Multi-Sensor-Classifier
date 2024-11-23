import os

import torch
import torch.nn as nn
from einops import rearrange
#from mgca.models.backbones import cnn_backbones
#from mgca.models.backbones.med import BertModel
from vits import create_vit
from transformers import AutoTokenizer, BertConfig, BertTokenizer, logging

logging.set_verbosity_error()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class GlobalEmbedding(nn.Module):
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 2048,
                 output_dim: int = 512) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        return self.head(x)


class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)

        return x.permute(0, 2, 1)


class ImageEncoder(nn.Module):
    def __init__(self,
                 model_name: str = "resnet_50",
                 text_feat_dim: int = 768,
                 output_dim: int = 768,
                 hidden_dim: int = 2048,
                 pretrained: bool = True
                 ):
        super(ImageEncoder, self).__init__()

        self.model_name = model_name
        self.output_dim = output_dim
        self.text_feat_dim = text_feat_dim

        vit_grad_ckpt = False
        vit_ckpt_layer = 0
        image_size = 224

        vit_name = model_name[4:]
        self.model, vision_width = create_vit(
            vit_name, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

        self.feature_dim = vision_width

        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True)
        state_dict = checkpoint["model"]
        msg = self.model.load_state_dict(state_dict, strict=False)

        self.global_embed = GlobalEmbedding(
            vision_width, hidden_dim, output_dim
        )

        self.local_embed = LocalEmbedding(
            vision_width, hidden_dim, output_dim
        )



    def vit_forward(self, x):
        return self.model(x, register_blk=11)

    def forward(self, x, get_local=False):

            img_feat = self.vit_forward(x)
            return img_feat[:, 0].contiguous(), img_feat[:, 1:].contiguous()






if __name__ == "__main__":
    from mgca.datasets.pretrain_dataset import MultimodalPretrainingDataset
    from mgca.datasets.transforms import DataTransforms
    transform = DataTransforms(is_train=True)
    dataset = MultimodalPretrainingDataset(split="train", transform=transform)

    for i, data in enumerate(dataset):
        imgs, caps, cap_len, key = data
        if caps["attention_mask"].sum() == 112:
            model = BertEncoder()
            report_feat, sent_feat, sent_mask, sents = model(
                caps["input_ids"],
                caps["attention_mask"],
                caps["token_type_ids"],
                get_local=True)