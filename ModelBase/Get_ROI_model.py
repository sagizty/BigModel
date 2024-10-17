"""
Build ROI level models    Script  ver： Oct 17th 22:00
"""
import timm
from pprint import pprint
import os
import sys
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

try:
    from ROI_models.GetPromptModel import *
except:
    from ModelBase.ROI_models.GetPromptModel import *

import torch
import torch.nn as nn
from torchvision import models


# get model
def get_model(num_classes=0, edge_size=224, model_idx=None, pretrained_backbone=True):
    """
    :param num_classes: classification required number of your dataset,
    0 for taking the feature, -1 for taking the original feature map

    :param edge_size: the input edge size of the dataloder
    :param model_idx: the model we are going to use. by the format of Model_size_other_info

    :param pretrained_backbone: The backbone CNN is initiate randomly or by its official Pretrained models

    :return: prepared model
    """
    # fixme internal token
    # Hugging Face API token
    os.environ["HF_TOKEN"] = "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"

    # default transforms
    transforms = None

    if pretrained_backbone == True or pretrained_backbone == False:
        pretrained_backbone_weight = None
        load_weight_online = pretrained_backbone

    elif pretrained_backbone == None:
        pretrained_backbone_weight = None
        load_weight_online = True

    else:  # str path
        if os.path.exists(pretrained_backbone):
            pretrained_backbone_weight = pretrained_backbone
        else:
            pretrained_backbone_weight = None
            raise  # the manual pretrained_backbone_weight is missing, check your file there
        load_weight_online = False

    if model_idx[0:5] == 'ViT_h':
        # Transfer learning for ViT
        model_names = timm.list_models('*vit*')
        pprint(model_names)
        if edge_size == 224:
            model = timm.create_model('vit_huge_patch14_224_in21k', pretrained=load_weight_online,
                                      num_classes=num_classes)
        else:
            print('not a avaliable image size with', model_idx)

    elif model_idx[0:5] == 'ViT_l':
        # Transfer learning for ViT
        model_names = timm.list_models('*vit*')
        pprint(model_names)
        if edge_size == 224:
            model = timm.create_model('vit_large_patch16_224', pretrained=load_weight_online,
                                      num_classes=num_classes)
        elif edge_size == 384:
            model = timm.create_model('vit_large_patch16_384', pretrained=load_weight_online,
                                      num_classes=num_classes)
        else:
            print('not a avaliable image size with', model_idx)

    elif model_idx[0:5] == 'ViT_s':
        # Transfer learning for ViT
        model_names = timm.list_models('*vit*')
        pprint(model_names)
        if edge_size == 224:
            model = timm.create_model('vit_small_patch16_224', pretrained=load_weight_online,
                                      num_classes=num_classes)
        elif edge_size == 384:
            model = timm.create_model('vit_small_patch16_384', pretrained=load_weight_online,
                                      num_classes=num_classes)
        else:
            print('not a avaliable image size with', model_idx)

    elif model_idx[0:5] == 'ViT_t':
        # Transfer learning for ViT
        model_names = timm.list_models('*vit*')
        pprint(model_names)
        if edge_size == 224:
            model = timm.create_model('vit_tiny_patch16_224', pretrained=load_weight_online,
                                      num_classes=num_classes)
        elif edge_size == 384:
            model = timm.create_model('vit_tiny_patch16_384', pretrained=load_weight_online,
                                      num_classes=num_classes)
        else:
            print('not a avaliable image size with', model_idx)

    elif model_idx[0:5] == 'ViT_b' or model_idx[0:3] == 'ViT':  # vit_base
        # ->  torch.Size([1, 768])
        # Transfer learning for ViT
        model_names = timm.list_models('*vit*')
        pprint(model_names)
        if edge_size == 224:
            model = timm.create_model('vit_base_patch16_224', pretrained=load_weight_online,
                                      num_classes=num_classes)
        elif edge_size == 384:
            model = timm.create_model('vit_base_patch16_384', pretrained=load_weight_online,
                                      num_classes=num_classes)
        else:
            print('not a avaliable image size with', model_idx)

    elif model_idx[0:3] == 'vgg':
        # Transfer learning for vgg16_bn
        model_names = timm.list_models('*vgg*')
        pprint(model_names)
        if model_idx[0:8] == 'vgg16_bn':
            model = timm.create_model('vgg16_bn', pretrained=load_weight_online, num_classes=num_classes)
        elif model_idx[0:5] == 'vgg16':
            model = timm.create_model('vgg16', pretrained=load_weight_online, num_classes=num_classes)
        elif model_idx[0:8] == 'vgg19_bn':
            model = timm.create_model('vgg19_bn', pretrained=load_weight_online, num_classes=num_classes)
        elif model_idx[0:5] == 'vgg19':
            model = timm.create_model('vgg19', pretrained=load_weight_online, num_classes=num_classes)

    elif model_idx[0:3] == 'uni' or model_idx[0:3] == "UNI":
        #  ->  torch.Size([1, 1024])
        if load_weight_online:
            # fixme this somehow failed, we build UNI with manual config and huggingface
            '''
            model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True,
                                      init_values=1e-5, dynamic_img_size=True)
            '''
            from huggingface_hub import hf_hub_download
            # Download the model weights
            model_weights_path = hf_hub_download(repo_id="MahmoodLab/UNI", filename="pytorch_model.bin",
                                                 use_auth_token=os.environ["HF_TOKEN"])

            # Load the model using timm and the downloaded weights
            # Create the ViT model using the configuration read from Hugging Face
            model = timm.create_model(
                "vit_large_patch16_224",  # Model architecture
                pretrained=True,  # Load pretrained weights
                img_size=224,  # Image size is 224x224
                num_classes=0,  # No classification head (feature extractor mode)
                init_values=1.0,  # Layer scale initialization value
                global_pool="token",  # Use token pooling (default for ViT)
                dynamic_img_size=True  # Allow dynamic image size
            )
            model.load_state_dict(torch.load(model_weights_path))
        else:
            model = timm.create_model(
                "vit_large_patch16_224",  # Model architecture
                pretrained=True,  # Load pretrained weights
                img_size=224,  # Image size is 224x224
                num_classes=0,  # No classification head (feature extractor mode)
                init_values=1.0,  # Layer scale initialization value
                global_pool="token",  # Use token pooling (default for ViT)
                dynamic_img_size=True  # Allow dynamic image size
            )

    # VPT feature embedding
    elif model_idx[0:3] == 'VPT' or model_idx[0:3] == 'vpt':
        # ->  torch.Size([1, 768])
        if load_weight_online:
            from huggingface_hub import hf_hub_download
            # Define the repo ID
            repo_id = "Tianyinus/PuzzleTuning_VPT"

            # Download the base state dictionary file
            base_state_dict_path = hf_hub_download(repo_id=repo_id,
                                                   filename="PuzzleTuning/Archive/ViT_b16_224_Imagenet.pth")

            # Download the prompt state dictionary file
            prompt_state_dict_path = hf_hub_download(repo_id=repo_id,
                                                     filename="PuzzleTuning/Archive/ViT_b16_224_timm_PuzzleTuning_SAE_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth")

            # Load these weights into your model
            base_state_dict = torch.load(base_state_dict_path)
            prompt_state_dict = torch.load(prompt_state_dict_path)

        else:
            prompt_state_dict = None
            base_state_dict = 'timm'

        # Build your model using the loaded state dictionaries
        model = build_promptmodel(prompt_state_dict=prompt_state_dict,
                                  base_state_dict=base_state_dict,
                                  num_classes=0)

    elif model_idx[0:4] == 'deit':  # Transfer learning for DeiT
        model_names = timm.list_models('*deit*')
        pprint(model_names)
        if edge_size == 384:
            model = timm.create_model('deit_base_patch16_384',
                                      pretrained=load_weight_online, num_classes=num_classes)
        elif edge_size == 224:
            model = timm.create_model('deit_base_patch16_224',
                                      pretrained=load_weight_online, num_classes=num_classes)
        else:
            pass

    elif model_idx[0:5] == 'twins':  # Transfer learning for twins

        model_names = timm.list_models('*twins*')
        pprint(model_names)
        model = timm.create_model('twins_pcpvt_base', pretrained=load_weight_online, num_classes=num_classes)

    elif model_idx[0:5] == 'pit_b' and edge_size == 224:  # Transfer learning for PiT
        model_names = timm.list_models('*pit*')
        pprint(model_names)
        model = timm.create_model('pit_b_224', pretrained=load_weight_online, num_classes=num_classes)

    elif model_idx[0:5] == 'gcvit' and edge_size == 224:  # Transfer learning for gcvit
        model_names = timm.list_models('*gcvit*')
        pprint(model_names)
        model = timm.create_model('gcvit_base', pretrained=load_weight_online, num_classes=num_classes)

    elif model_idx[0:6] == 'xcit_s':  # Transfer learning for XCiT
        model_names = timm.list_models('*xcit*')
        pprint(model_names)
        if edge_size == 384:
            model = timm.create_model('xcit_small_12_p16_384_dist', pretrained=load_weight_online,
                                      num_classes=num_classes)
        elif edge_size == 224:
            model = timm.create_model('xcit_small_12_p16_224_dist', pretrained=load_weight_online,
                                      num_classes=num_classes)
        else:
            pass

    elif model_idx[0:6] == 'xcit_m':  # Transfer learning for XCiT
        model_names = timm.list_models('*xcit*')
        pprint(model_names)
        if edge_size == 384:
            model = timm.create_model('xcit_medium_24_p16_384_dist', pretrained=load_weight_online,
                                      num_classes=num_classes)
        elif edge_size == 224:
            model = timm.create_model('xcit_medium_24_p16_224_dist', pretrained=load_weight_online,
                                      num_classes=num_classes)
        else:
            pass

    elif model_idx[0:6] == 'mvitv2':  # Transfer learning for MViT v2 small  fixme bug in model!
        model_names = timm.list_models('*mvitv2*')
        pprint(model_names)
        model = timm.create_model('mvitv2_small_cls', pretrained=load_weight_online, num_classes=num_classes)

    elif model_idx[0:6] == 'convit' and edge_size == 224:  # Transfer learning for ConViT fixme bug in model!
        model_names = timm.list_models('*convit*')
        pprint(model_names)
        model = timm.create_model('convit_base', pretrained=load_weight_online, num_classes=num_classes)

    elif model_idx[0:6] == 'ResNet':  # Transfer learning for the ResNets
        if model_idx[0:8] == 'ResNet18':
            model = models.resnet18(pretrained=load_weight_online)
        elif model_idx[0:8] == 'ResNet34':
            model = models.resnet34(pretrained=load_weight_online)
        elif model_idx[0:8] == 'ResNet50':
            model = models.resnet50(pretrained=load_weight_online)
        elif model_idx[0:9] == 'ResNet101':
            model = models.resnet101(pretrained=load_weight_online)
        else:
            print('this model is not defined in get model')
            return -1

        # Custom Flatten module
        class Feature_Flatten(nn.Module):
            def forward(self, x):
                # Flatten the tensor while keeping the batch dimension
                return x.view(x.size(0), -1)

        num_ftrs = model.fc.in_features
        if num_classes > 0:
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif num_classes == 0:
            model.fc = Feature_Flatten()
            # ResNet18 ->  torch.Size([1, 512])
            # ResNet34 ->  torch.Size([1, 512])
            # ResNet50 ->  torch.Size([1, 2048])
            # ResNet101 ->  torch.Size([1, 2048])
        elif num_classes == -1:  # call for original feature shape
            model.fc = nn.Identity()
        else:
            raise NotImplementedError

    elif model_idx[0:7] == 'bot_256' and edge_size == 256:  # Model: BoT
        model_names = timm.list_models('*bot*')
        pprint(model_names)
        # NOTICE: we find no weight for BoT in timm
        # ['botnet26t_256', 'botnet50ts_256', 'eca_botnext26ts_256']
        model = timm.create_model('botnet26t_256', pretrained=load_weight_online, num_classes=num_classes)

    elif model_idx[0:7] == 'Virchow':
        # -> [1, 2560]
        class VirchowTaskModel(nn.Module):
            """
            A model that integrates the Virchow backbone and extracts a class token or a full tile embedding based on the version.

            Args:
                backbone_model: The pre-trained Virchow backbone model.
                Virchow_version: The version of Virchow model being used ('1' or '2').
                num_classes: If non-zero, returns only the class token (used for classification tasks).
                             If zero, concatenates the class token and mean patch token for feature extraction.
            """

            def __init__(self, backbone_model, Virchow_version='1', num_classes=0):
                super(VirchowTaskModel, self).__init__()
                assert Virchow_version in ['1', '2'], "Virchow_version must be either '1' or '2'"

                self.backbone_model = backbone_model
                self.Virchow_version = Virchow_version
                self.num_classes = num_classes

            def forward(self, x):
                """
                Memo from original authors:
                    output = model(image)  # size: 1 x 257 (or 261 for Virchow2) x 1280

                    class_token = output[:, 0]    # size: 1 x 1280
                    patch_tokens = output[:, 1:]  # size: 1 x 256 x 1280
                    # for Virchow 2 its:
                    # patch_tokens = output[:, 5:]  # size: 1 x 260 x 1280, tokens 1-4 are register tokens so we ignore those

                    # concatenate class token and average pool of patch tokens
                    embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
                    # We concatenate the class token and the mean patch token to create the final tile embedding.

                    Notice:
                    In more resource constrained settings, one can experiment with using just class token or the mean patch
                    token. For downstream tasks with dense outputs (i.e. segmentation), the 256 x 1280 tensor of patch
                    tokens can be used.
                """
                # Pass image through the backbone model
                output = self.backbone_model(x)  # size: 1 x 257 (or 261 for Virchow2) x 1280

                # Extract class token
                class_token = output[:, 0]  # size: 1 x 1280 or 1 x num_classes (if used for classification task)

                # If num_classes is non-zero, return only the class token (for classification tasks)
                if self.num_classes != 0:
                    return class_token

                # For Virchow2, ignore the first 4 tokens (register tokens)
                if self.Virchow_version == '2':
                    patch_tokens = output[:, 5:]  # size: 1 x 260 x 1280
                else:
                    patch_tokens = output[:, 1:]  # size: 1 x 256 x 1280

                # Compute final embedding by concatenating class token and mean of patch tokens
                embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560

                return embedding

        # ref: https://huggingface.co/paige-ai/Virchow
        if load_weight_online:
            from timm.layers import SwiGLUPacked

            if model_idx[0:8] == 'Virchow2':
                Virchow_version = '2'
                backbone_model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                                                   num_classes=0 if num_classes < 0 else num_classes,
                                                   mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
            else:  # Virchow1
                Virchow_version = '1'
                backbone_model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True,
                                                   num_classes=0 if num_classes < 0 else num_classes,
                                                   mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

            transforms = create_transform(**resolve_data_config(backbone_model.pretrained_cfg, model=backbone_model))
            if num_classes == -1:
                model = backbone_model
            else:
                model = VirchowTaskModel(backbone_model=backbone_model,
                                         Virchow_version=Virchow_version,
                                         num_classes=num_classes)

        else:
            from timm.layers import SwiGLUPacked

            if model_idx[0:8] == 'Virchow2':
                Virchow_version = '2'
                # Model configuration from JSON
                config = {
                    "architecture": "vit_huge_patch14_224",
                    "model_args": {
                        "img_size": 224,
                        "init_values": 1e-5,
                        "num_classes": 0,
                        "reg_tokens": 4,
                        "mlp_ratio": 5.3375,
                        "global_pool": "",
                        "dynamic_img_size": True
                    },
                    "pretrained_cfg": {
                        "tag": "virchow_v2",
                        "custom_load": False,
                        "input_size": [3, 224, 224],
                        "fixed_input_size": False,
                        "interpolation": "bicubic",
                        "crop_pct": 1.0,
                        "crop_mode": "center",
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                        "num_classes": 0,
                        "pool_size": None,
                        "first_conv": "patch_embed.proj",
                        "classifier": "head",
                        "license": "CC-BY-NC-ND-4.0"
                    }
                }
            else:  # Virchow1
                Virchow_version = '1'
                # Model configuration from JSON
                config = {
                    "architecture": "vit_huge_patch14_224",
                    "model_args": {
                        "img_size": 224,
                        "init_values": 1e-5,
                        "num_classes": 0,
                        "mlp_ratio": 5.3375,
                        "global_pool": "",
                        "dynamic_img_size": True
                    },
                    "pretrained_cfg": {
                        "tag": "virchow_v1",
                        "custom_load": False,
                        "input_size": [3, 224, 224],
                        "fixed_input_size": False,
                        "interpolation": "bicubic",
                        "crop_pct": 1.0,
                        "crop_mode": "center",
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                        "num_classes": 0,
                        "pool_size": None,
                        "first_conv": "patch_embed.proj",
                        "classifier": "head",
                        "license": "Apache 2.0"
                    }
                }

            # Create the model using timm
            backbone_model = timm.create_model(
                config['architecture'],
                pretrained=False,
                img_size=config['model_args']['img_size'],
                init_values=config['model_args']['init_values'],
                mlp_ratio=config['model_args']['mlp_ratio'],
                num_classes=num_classes,
                global_pool=config['model_args']['global_pool'],
                dynamic_img_size=config['model_args']['dynamic_img_size'],
                mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

            # Virchow 1 and 2 are special as we are loading the backbone and stack the model here,
            # if loading their weights, we should load the weights to the backbone
            # other model just load weightsin the end of the model building
            backbone_model.load_state_dict(torch.load(pretrained_backbone_weight), False)

            if num_classes == -1:
                model = backbone_model
            else:
                model = VirchowTaskModel(backbone_model=backbone_model,
                                         Virchow_version=Virchow_version,
                                         num_classes=num_classes)

    # prov-gigapath feature embedding ViT
    elif model_idx[0:8] == 'gigapath':
        # ->  torch.Size([1, 1536])
        # ref: https://www.nature.com/articles/s41586-024-07441-w
        if load_weight_online:
            # fixme if "HF_TOKEN" failed, use your own hugging face token and register for the project gigapath
            model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        else:
            # Model configuration from your JSON
            config = {
                "architecture": "vit_giant_patch14_dinov2",
                "num_classes": num_classes,  # original is 0
                "num_features": 1536,
                "global_pool": "token",
                "model_args": {
                    "img_size": 224,
                    "in_chans": 3,
                    "patch_size": 16,
                    "embed_dim": 1536,
                    "depth": 40,
                    "num_heads": 24,
                    "init_values": 1e-05,
                    "mlp_ratio": 5.33334,
                    "num_classes": 0
                }
            }

            # Create the model using timm
            model = timm.create_model(
                config['architecture'],
                pretrained=False,  # Set to True if you want to load pretrained weights
                img_size=config['model_args']['img_size'],
                in_chans=config['model_args']['in_chans'],
                patch_size=config['model_args']['patch_size'],
                embed_dim=config['model_args']['embed_dim'],
                depth=config['model_args']['depth'],
                num_heads=config['model_args']['num_heads'],
                init_values=config['model_args']['init_values'],
                mlp_ratio=config['model_args']['mlp_ratio'],
                num_classes=config['model_args']['num_classes']
            )

    elif model_idx[0:8] == 'densenet':  # Transfer learning for densenet
        model_names = timm.list_models('*densenet*')
        pprint(model_names)
        model = timm.create_model('densenet121', pretrained=load_weight_online, num_classes=num_classes)

    elif model_idx[0:8] == 'xception':  # Transfer learning for Xception
        model_names = timm.list_models('*xception*')
        pprint(model_names)
        model = timm.create_model('xception', pretrained=load_weight_online, num_classes=num_classes)

    elif model_idx[0:9] == 'pvt_v2_b0':  # Transfer learning for PVT v2 (todo not applicable with torch summary)
        model_names = timm.list_models('*pvt_v2*')
        pprint(model_names)
        model = timm.create_model('pvt_v2_b0', pretrained=load_weight_online, num_classes=num_classes)

    elif model_idx[0:9] == 'visformer' and edge_size == 224:  # Transfer learning for Visformer
        model_names = timm.list_models('*visformer*')
        pprint(model_names)
        model = timm.create_model('visformer_small', pretrained=load_weight_online, num_classes=num_classes)

    elif model_idx[0:9] == 'coat_mini' and edge_size == 224:  # Transfer learning for coat_mini
        model_names = timm.list_models('*coat*')
        pprint(model_names)
        model = timm.create_model('coat_mini', pretrained=load_weight_online, num_classes=num_classes)

    elif model_idx[0:10] == 'swin_b_384' and edge_size == 384:  # Transfer learning for Swin Transformer (swin_b_384)
        model_names = timm.list_models('*swin*')
        pprint(model_names)  # swin_base_patch4_window12_384  swin_base_patch4_window12_384_in22k
        model = timm.create_model('swin_base_patch4_window12_384', pretrained=load_weight_online,
                                  num_classes=num_classes)

    elif model_idx[0:10] == 'swin_b_224' and edge_size == 224:  # Transfer learning for Swin Transformer (swin_b_384)
        model_names = timm.list_models('*swin*')
        pprint(model_names)  # swin_base_patch4_window7_224  swin_base_patch4_window7_224_in22k
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=load_weight_online,
                                  num_classes=num_classes)

    elif model_idx[0:11] == 'mobilenetv3':  # Transfer learning for mobilenetv3
        model_names = timm.list_models('*mobilenet*')
        pprint(model_names)
        model = timm.create_model('mobilenetv3_large_100', pretrained=load_weight_online,
                                  num_classes=num_classes)

    elif model_idx[0:11] == 'mobilevit_s':  # Transfer learning for mobilevit_s
        model_names = timm.list_models('*mobilevit*')
        pprint(model_names)
        model = timm.create_model('mobilevit_s', pretrained=load_weight_online, num_classes=num_classes)

    elif model_idx[0:11] == 'inceptionv3':  # Transfer learning for Inception v3
        model_names = timm.list_models('*inception*')
        pprint(model_names)
        model = timm.create_model('inception_v3', pretrained=load_weight_online, num_classes=num_classes)

    elif model_idx[0:13] == 'crossvit_base':  # Transfer learning for crossvit_base  (todo not okey with torch summary)
        model_names = timm.list_models('*crossvit_base*')
        pprint(model_names)
        model = timm.create_model('crossvit_base_240', pretrained=load_weight_online, num_classes=num_classes)

    elif model_idx[0:14] == 'efficientnet_b':  # Transfer learning for efficientnet_b3,4
        model_names = timm.list_models('*efficientnet*')
        pprint(model_names)
        model = timm.create_model(model_idx[0:15], pretrained=load_weight_online, num_classes=num_classes)

    elif model_idx[0:14] == 'ResN50_ViT_384':  # ResNet+ViT融合模型384
        model_names = timm.list_models('*vit_base_resnet*')
        pprint(model_names)
        model = timm.create_model('vit_base_resnet50_384', pretrained=load_weight_online, num_classes=num_classes)

    elif model_idx[0:15] == 'coat_lite_small' and edge_size == 224:  # Transfer learning for coat_lite_small
        model_names = timm.list_models('*coat*')
        pprint(model_names)
        model = timm.create_model('coat_lite_small', pretrained=load_weight_online, num_classes=num_classes)

    elif model_idx[0:17] == 'efficientformer_l' and edge_size == 224:  # Transfer learning for efficientnet_b3,4
        model_names = timm.list_models('*efficientformer*')
        pprint(model_names)
        model = timm.create_model(model_idx[0:18], pretrained=load_weight_online, num_classes=num_classes)

    else:
        print('\nThe model', model_idx, 'with the edge size of', edge_size)
        print("is not defined in the script！！", '\n')
        raise NotImplementedError

    # load state
    try:
        if os.path.exists(pretrained_backbone_weight):
            missing_keys, unexpected_keys = model.load_state_dict(pretrained_backbone_weight, False)
            if len(missing_keys) > 0:
                for k in missing_keys:
                    print("Missing ", k)

            if len(unexpected_keys) > 0:
                for k in unexpected_keys:
                    print("Unexpected ", k)
    except:
        pass

    try:
        # Print the model to verify
        print(model)

        img = torch.randn(1, 3, edge_size, edge_size)
        preds = model(img)  # (1, class_number)
        # print('test model output:', preds)
        print('Build ROI model with in/out shape: ', img.shape, ' -> ', preds.shape)

        print("ROI model param size #", sum(p.numel() for p in model.parameters()))
    except:
        print("Problem exist in the model defining process！！")
        raise  # Problem exist in the model defining process
    else:
        print('model is ready now!')
        return model
        # todo for future maybe we return model, transforms

            


# ------------------- ROI VQA Image Encoder (ViT) -------------------
# Pre-processed image tensor is passed through the Vision Transformer (ViT), to obtain image embedding (ViT CLS token)
class ImageEncoder(nn.Module):
    def __init__(self, model_idx='uni', embed_size=768):
        super(ImageEncoder, self).__init__()

        from timm.layers import SwiGLUPacked
        self.Image_Encoder = get_model(model_idx=model_idx, num_classes=0)

        self.embed_convert = nn.Linear(self.Image_Encoder.embed_dim, embed_size) \
            if self.Text_Encoder.embed_dim != embed_size else nn.Identity()

    def forward(self, images):
        # Process image through Image_Encoder to get the embeddings
        Image_cls_embedding = self.Image_Encoder(images)  # CLS token output from ViT [B,D]
        return self.embed_convert(Image_cls_embedding)

