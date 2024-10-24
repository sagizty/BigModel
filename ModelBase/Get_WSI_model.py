"""
Build WSI level models      Script  ver： Oct 25th 00:30
"""
import os
import sys
from pathlib import Path

# For convenience, import all path to sys
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir))
sys.path.append(str(this_file_dir.parent))
sys.path.append(str(this_file_dir.parent.parent))
sys.path.append(str(this_file_dir.parent.parent.parent))  # Go up 3 levels

import torch
import torch.nn as nn
import huggingface_hub
from typing import Optional, List
import timm

# from WSI_models.path_rwkv.v6 import PathRWKVv6 as PathRWKV

from ModelBase.WSI_models.gigapath.slide_encoder import gigapath_slide_enc12l768d
from ModelBase.WSI_models.WSI_Transformer.WSI_Transformer_blocks import Slide_Transformer_blocks, Prompt_Slide_Transformer_blocks


def build_WSI_backbone_model(model_name='gigapath', local_weight_path=None,
                             ROI_feature_dim=None, MTL_token_num=0, **kwargs):
    # fixme internal token
    # Hugging Face API token
    os.environ["HF_TOKEN"] = "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"

    if model_name == 'SlideViT':
        default_ROI_feature_dim = 768  # the embedding input size for vit_base_patch16_224
        slide_embed_dim = 768  # the embedding output size

        slide_backbone = Slide_Transformer_blocks(default_ROI_feature_dim=default_ROI_feature_dim,
                                                  embed_dim=default_ROI_feature_dim,
                                                  MTL_token_num=MTL_token_num,
                                                  depth=12, num_heads=12, mlp_ratio=4.,
                                                  qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                                                  drop_path_rate=0., norm_layer=None, act_layer=None, **kwargs)
        # Transferlearning on Encoders
        ViT_backbone_weights = timm.create_model('vit_base_patch16_224', pretrained=True).state_dict()
        del ViT_backbone_weights['patch_embed.proj.weight']
        del ViT_backbone_weights['patch_embed.proj.bias']
        slide_backbone.load_state_dict(ViT_backbone_weights, False)
        return slide_backbone, default_ROI_feature_dim, slide_embed_dim

    elif model_name == 'SlideVPT':
        default_ROI_feature_dim = 768  # the embedding input size for vit_base_patch16_224
        slide_embed_dim = 768  # the embedding output size

        slide_backbone = Prompt_Slide_Transformer_blocks(Prompt_Token_num=20, VPT_type="Deep",
                                                         default_ROI_feature_dim=default_ROI_feature_dim,
                                                         embed_dim=default_ROI_feature_dim,
                                                         MTL_token_num=MTL_token_num,
                                                         depth=12, num_heads=12, mlp_ratio=4.,
                                                         qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                                                         drop_path_rate=0., norm_layer=None, act_layer=None, **kwargs)
        # Transferlearning on Encoders
        ViT_backbone_weights = timm.create_model('vit_base_patch16_224', pretrained=True).state_dict()
        del ViT_backbone_weights['patch_embed.proj.weight']
        del ViT_backbone_weights['patch_embed.proj.bias']
        slide_backbone.load_state_dict(ViT_backbone_weights, False)
        return slide_backbone, default_ROI_feature_dim, slide_embed_dim

    elif model_name == 'gigapath':
        default_ROI_feature_dim = 1536  # the embedding input size
        slide_embed_dim = 768  # the embedding output size

        slide_backbone = gigapath_slide_enc12l768d(in_chans=ROI_feature_dim or default_ROI_feature_dim,
                                                   global_pool=False, **kwargs)
        print("Slide encoder param #", sum(p.numel() for p in slide_backbone.parameters()))

        # download the weights
        if local_weight_path is None:
            hf_hub = "hf_hub:prov-gigapath/prov-gigapath"
            hub_name = hf_hub.split(":")[1]
            local_dir = os.path.join(os.path.expanduser("~"), ".cache/")

            huggingface_hub.hf_hub_download(hub_name, filename="slide_encoder.pth",
                                            local_dir=local_dir, force_download=True)
            local_weight_path = os.path.join(local_dir, "slide_encoder.pth")

        # build weight for slide_feature level
        if local_weight_path is False:
            print("Pretrained weights not required. Randomly initialized the model! ")

        elif os.path.exists(local_weight_path):
            state_dict = torch.load(local_weight_path, map_location="cpu")["model"]

            missing_keys, unexpected_keys = slide_backbone.load_state_dict(state_dict, strict=False)
            if len(missing_keys) > 0:
                for k in missing_keys:
                    print("Missing ", k)

            if len(unexpected_keys) > 0:
                for k in unexpected_keys:
                    print("Unexpected ", k)

            print("\033[92m Successfully Loaded Pretrained GigaPath model from {} \033[00m".format(local_weight_path))
        else:
            print("\033[93m Pretrained weights not found at {}. Randomly initialized the model! \033[00m".format(
                local_weight_path))

        return slide_backbone, default_ROI_feature_dim, slide_embed_dim

    elif model_name == "PathRWKV":
        default_ROI_feature_dim = 768  # the embedding input size  fixme csc pls check this,
        slide_embed_dim = 768  # the embedding output size  fixme csc pls check this,

        slide_backbone = PathRWKV(**kwargs)

        if local_weight_path:
            state_dict = torch.load(local_weight_path, map_location="cpu", weights_only=True)
            missing_keys, unexpected_keys = slide_backbone.load_state_dict(state_dict)
            if len(missing_keys) > 0:
                for k in missing_keys:
                    print("Missing ", k)

            if len(unexpected_keys) > 0:
                for k in unexpected_keys:
                    print("Unexpected ", k)

            print(f"Successfully loaded pretrained model from {local_weight_path}")

        # for test
        elif local_weight_path == False:
            pass

        else:
            slide_backbone.init_params()
            print("Pretrained weights not found. Initializing model by default method.")

        return slide_backbone, default_ROI_feature_dim, slide_embed_dim

    elif model_name[0:3] == 'UNI':
        # ABMIL
        pass

    elif model_name[0:7] == 'Virchow':
        pass

    else:
        raise NotImplementedError


class MTL_module_baseline(nn.Module):
    '''
    this one project [B,D] -> [B,N*D]
    '''
    def __init__(self, MTL_token_num, latent_feature_dim):
        super().__init__()
        self.MTL_token_num, self.latent_feature_dim = MTL_token_num, latent_feature_dim
        self.layer = nn.Linear(latent_feature_dim, MTL_token_num * latent_feature_dim)

    def forward(self, latent_features):
        MTL_tokens = self.layer(latent_features).view(-1, self.MTL_token_num, self.latent_feature_dim)
        return MTL_tokens


class MTL_Model_builder(nn.Module):
    def __init__(self, backbone: nn.Module,
                 MTL_module: Optional[nn.Module] = None, MTL_token_design=None,
                 MTL_heads: List[nn.Module] = [nn.Identity(), nn.Identity(), nn.Identity()],
                 ROI_feature_dim: int = 1536, default_ROI_feature_dim: int = 1536,
                 slide_embed_dim: int = 768, MTL_feature_dim: int = 128, Froze_backbone=False):
        """
        :param backbone: slide_feature-level modeling model

        :param MTL_module: MTL model projecting the features to multiple task heads
        :param MTL_token_design: default 'Through_out' for putting the tokens in each transformer layer,
                                else 'latent' convert from the slide model output
        :param MTL_heads: the list of multiple WSI-level task heads for each task

        :param ROI_feature_dim: feature dim of embedded tile images
        :param default_ROI_feature_dim: feature dim of default(pre-trained) slide model design

        :param slide_embed_dim: feature dim for WSI level modeling model's output
        :param MTL_feature_dim: feature dim for MTL modeling

        :param Froze_backbone:
        """
        super().__init__()

        self.backbone = backbone  # the output feature is [B, slide_embed_dim]
        self.MTL_token_design = MTL_token_design

        if ROI_feature_dim == default_ROI_feature_dim:
            self.ROI_embedding_converter = nn.Identity()
        else:
            self.ROI_embedding_converter = nn.Linear(ROI_feature_dim, default_ROI_feature_dim)

        if MTL_feature_dim == slide_embed_dim:
            self.slide_embedding_converter = nn.Identity()
        else:
            self.slide_embedding_converter = nn.Linear(slide_embed_dim, MTL_feature_dim)

        self.MTL_token_num = len(MTL_heads)
        assert self.MTL_token_num > 0, "MTL_heads cannot be empty."

        # MTL_token_design: default 'Through_out' for putting the tokens in the first layer like cls token
        # and then pass through all transformer layer,
        # else None, convert from the slide model output to MTL token
        if self.MTL_token_design == 'Through_out':
            # Embedding the MTL tokens
            self.MTL_tokens = nn.Parameter(torch.zeros(1, self.MTL_token_num, default_ROI_feature_dim))
            self.MTL_module = nn.Identity()  # TODO enable future design of MTL modules
        else:
            self.MTL_tokens = None
            if MTL_module is None:
                # this one project [B,D] -> [B,N*D]
                self.MTL_module = MTL_module_baseline(self.MTL_token_num, MTL_feature_dim)
                # MTL_feature_dim is for MTL_heads
            else:
                self.MTL_module = MTL_module

        self.MTL_heads = nn.ModuleList(MTL_heads)

        if Froze_backbone:
            self.Freeze_backbone()

    def Freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def UnFreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, image_features, coords=None):
        """
        Forward pass for the MTL Transformer.

        :param image_features: Tensor of shape [B, N, feature_dim],
                               where B is batch size, N is the number of patches/features per slide_feature.
        :return: List of task predictions, where each element in the list corresponds to a task and has
                 shape [B, output_dim] (output_dim may vary depending on the task).
        """
        # [token_num, ROI_feature_dim] -> [token_num, default_ROI_feature_dim]
        image_features = self.ROI_embedding_converter(image_features)

        if self.MTL_token_design == 'Through_out':
            # [batch, MTL_token_num + tile_num, default_ROI_feature_dim] ->
            # [batch, MTL_token_num + tile_num, slide_embed_dim]
            slide_latent = self.backbone(image_features, coords=coords, MTL_tokens=self.MTL_tokens)

            # [B, MTL_token_num, slide_embed_dim]
            MTL_tokens = slide_latent[:, :self.MTL_token_num]

        else:  # this design generate MTL tokens from latent
            # [B, tile_num, default_ROI_feature_dim] -> [B, tile_num, slide_embed_dim] or [B, slide_embed_dim]
            if coords is None:
                slide_latent = self.backbone(image_features)
            else:
                slide_latent = self.backbone(image_features, coords=coords)
            # [B, slide_embed_dim]
            # fixme temp: global average pooling
            MTL_tokens = slide_latent.mean(dim=1) if len(slide_latent.shape) == 3 else slide_latent  # shape of [B,D]
        # MTL module
        # [..., slide_embed_dim] -> [..., MTL_feature_dim], MTL_feature_dim is for MTL_heads
        # -> [B, MTL_token_num, MTL_feature_dim]
        MTL_tokens = self.MTL_module(self.slide_embedding_converter(MTL_tokens))

        # Predicate the Tasks with MTL task heads
        WSI_tasks_pred = []
        for idx, head in enumerate(self.MTL_heads):
            task_pred = head(MTL_tokens[:, idx])
            WSI_tasks_pred.append(task_pred)

        return WSI_tasks_pred


# ------------------- WSI VQA Image Encoder (ViT) -------------------
# Pre-processed image tensor is passed through WSI model to obtain image embedding (ViT CLS token)
class ImageEncoder(nn.Module):
    def __init__(self, WSI_Encoder, embed_size=768):
        super(ImageEncoder, self).__init__()

        self.Image_Encoder = WSI_Encoder
        self.embed_convert = nn.Linear(self.Image_Encoder.embed_dim, embed_size) \
            if self.Text_Encoder.embed_dim != embed_size else nn.Identity()

    def forward(self, images):
        # Process image through Image_Encoder to get the embeddings
        Image_cls_embedding = self.Image_Encoder(images)  # CLS token output from ViT [B,D]
        return self.embed_convert(Image_cls_embedding)


class slide_embedding_model_builder(nn.Module):
    def __init__(self, backbone: nn.Module):
        """
        :param backbone: slide_feature-level modeling model
        """
        super().__init__()

        self.backbone = backbone  # the output feature is [B, slide_embed_dim]

    def forward(self, image_features, img_coords):
        """
        Forward pass for the MTL Transformer.

        :param image_features: Tensor of shape [B, N, feature_dim],
                               where B is batch size, N is the number of patches/features per slide_feature.
        :return: slide_latent has
                 shape [B, output_dim] (output_dim may vary depending on the task).
        """
        slide_latent = self.backbone(image_features, img_coords)

        return slide_latent


def build_WSI_task_model(MTL_token_design ='Through_out',
                         model_name='gigapath', local_weight_path=None, ROI_feature_dim=None,
                         MTL_heads: List[nn.Module] = None, latent_feature_dim=128, Froze_backbone=False, **kwargs):
    """
    :param MTL_token_design: default 'Through_out' for putting the tokens in each transformer layer,
                                else 'latent' convert from the slide model output
    """
    assert MTL_heads is not None
    MTL_token_num = len(MTL_heads) if MTL_token_design =='Through_out' else 0

    slide_backbone, default_ROI_feature_dim, slide_embed_dim = \
        build_WSI_backbone_model(model_name, local_weight_path, ROI_feature_dim, MTL_token_num=MTL_token_num,
                                 **kwargs)
    ROI_feature_dim = ROI_feature_dim or default_ROI_feature_dim

    if model_name == 'gigapath':  # fixme this should also include others models
        MTL_token_design=None

    MTL_Model = MTL_Model_builder(slide_backbone, MTL_module=None, MTL_token_design=MTL_token_design,
                                  MTL_heads=MTL_heads,
                                  ROI_feature_dim=ROI_feature_dim, default_ROI_feature_dim=default_ROI_feature_dim,
                                  slide_embed_dim=slide_embed_dim, MTL_feature_dim=latent_feature_dim,
                                  Froze_backbone=Froze_backbone)

    return MTL_Model


def build_WSI_prob_embedding_model(model_name='gigapath', local_weight_path=None, ROI_feature_dim=None, **kwargs):
    slide_backbone, default_ROI_feature_dim, slide_embed_dim = \
        build_WSI_backbone_model(model_name, local_weight_path, ROI_feature_dim, **kwargs)

    slide_embedding_model = slide_embedding_model_builder(slide_backbone)
    return slide_embedding_model, default_ROI_feature_dim


def build_WSI_MLLM_model(model_name='gigapath', local_weight_path=None, ROI_feature_dim=None, **kwargs):
    pass


if __name__ == '__main__':
    # cuda issue
    print('cuda avaliablity:', torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    slide_task_model = build_WSI_task_model(MTL_token_design=None,
                                            model_name='SlideVPT', local_weight_path=None, ROI_feature_dim=None,
                                            latent_feature_dim=128, MTL_heads=[nn.Linear(128,3)])

    # MTL_token_design='Through_out'

    slide_task_model = slide_task_model.to(dev).half()

    # play data (to see that Transformer can do multiple length, therefore can do MTL easily)
    coords = torch.randn(2, 528, 2).to(dev).half()
    x = torch.randn(2, 528, 768).to(dev).half()
    y = slide_task_model(x,coords=coords)
    print(y)