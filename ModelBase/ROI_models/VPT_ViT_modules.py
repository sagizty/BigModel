"""
VPT     Script  ver： Apr 18th 17:30

based on
timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
"""

import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer, PatchEmbed


class VPT_ViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None, Prompt_Token_num=1,
                 VPT_type="Shallow", basic_state_dict=None):

        # Firstly, build ViT backbone
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         norm_layer=norm_layer, act_layer=act_layer)

        # load basic state_dict
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

        self.VPT_type = VPT_type  # "Deep" "Shallow" or None
        if VPT_type == "Deep":
            self.Prompt_Tokens = nn.Parameter(torch.zeros(depth, Prompt_Token_num, embed_dim))
            print('building VPT-'+str(VPT_type))
        elif VPT_type == "Shallow":
            self.Prompt_Tokens = nn.Parameter(torch.zeros(1, Prompt_Token_num, embed_dim))
            print('building VPT-' + str(VPT_type))
        else:
            self.Prompt_Tokens = None
            self.VPT_type = None
            print('building ViT instead of VPT')

    def New_CLS_head(self, new_classes=0):
        if new_classes==0:
            self.head = nn.Identity()
        else:
            self.head = nn.Linear(self.embed_dim, new_classes)

    def Freeze(self):
        # freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        if self.Prompt_Tokens is not None:
            self.Prompt_Tokens.requires_grad = True
            try:
                for param in self.head.parameters():
                    param.requires_grad = True
            except:
                pass

    def UnFreeze(self):
        # unfreeze all parameters
        for param in self.parameters():
            param.requires_grad = True

    def obtain_prompt(self):
        prompt_state_dict = {'head': self.head.state_dict(),
                             'Prompt_Tokens': self.Prompt_Tokens}
        # print(prompt_state_dict)
        return prompt_state_dict

    def load_prompt(self, prompt_state_dict):
        assert self.Prompt_Tokens is not None

        try:
            self.head.load_state_dict(prompt_state_dict['head'], False)
        except:
            print('head not match, so skip head')
        else:
            print('prompt head match')

        if self.Prompt_Tokens.shape == prompt_state_dict['Prompt_Tokens'].shape:

            # device check
            Prompt_Tokens = nn.Parameter(prompt_state_dict['Prompt_Tokens'].cpu())
            Prompt_Tokens.to(torch.device(self.Prompt_Tokens.device))

            self.Prompt_Tokens = Prompt_Tokens

        else:
            print('\n !!! cannot load prompt')
            print('shape of model req prompt', self.Prompt_Tokens.shape)
            print('shape of model given prompt', prompt_state_dict['Prompt_Tokens'].shape)
            print('')

    def forward_features(self, x):
        x = self.patch_embed(x)
        # print(x.shape, self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.VPT_type == "Deep":

            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            for i in range(len(self.blocks)):
                # concatenate Prompt_Tokens
                Prompt_Tokens = self.Prompt_Tokens[i].unsqueeze(0)
                # firstly concatenate
                x = torch.cat((x, Prompt_Tokens.expand(x.shape[0], -1, -1)), dim=1)
                num_tokens = x.shape[1]
                # lastly remove, a genius trick
                x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num]

        elif self.VPT_type == "Shallow":
            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            # concatenate Prompt_Tokens
            Prompt_Tokens = self.Prompt_Tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((x, Prompt_Tokens), dim=1)
            num_tokens = x.shape[1]
            # Sequntially procees
            x = self.blocks(x)[:, :num_tokens - Prompt_Token_num]

        else:  # self.VPT_type is None
            x = self.blocks(x)

        # normalize all output tokens (sequence without prompt tokens)
        x = self.norm(x)
        return x

    def forward(self, x):

        x = self.forward_features(x)

        # use cls token for cls head
        try:
            x = self.pre_logits(x[:, 0, :])
        except:
            x = self.fc_norm(x[:, 0, :])
        else:
            pass
        x = self.head(x)

        return x


def build_ViT_or_VPT(num_classes=0, edge_size=224, model_idx='ViT', patch_size=16,
                     Prompt_Token_num=20, VPT_type="Deep", prompt_state_dict=None,
                     base_state_dict='timm'):
    # VPT_type = "Deep" / "Shallow"

    if model_idx[0:3] == 'ViT':

        if base_state_dict is None:
            basic_state_dict = None

        elif type(base_state_dict) == str:
            if base_state_dict == 'timm':
                # ViT_Prompt
                import timm
                # from pprint import pprint
                # model_names = timm.list_models('*vit*')
                # pprint(model_names)

                basic_model = timm.create_model('vit_base_patch' + str(patch_size) + '_' + str(edge_size),
                                                pretrained=True)
                basic_state_dict = basic_model.state_dict()
                print('in prompt model building, timm ViT loaded for pretrained_weight')

            else:
                basic_state_dict = None
                print('in prompt model building, no vaild str for pretrained_weight')

        else:  # state dict: collections.OrderedDict
            basic_state_dict = base_state_dict
            print('in prompt model building, a .pth pretrained_weight loaded')

        model = VPT_ViT(img_size=edge_size, patch_size=patch_size, Prompt_Token_num=Prompt_Token_num,
                        VPT_type=VPT_type, basic_state_dict=basic_state_dict)
        # add new head, set num_classes=0 to use the latent embedding only
        model.New_CLS_head(num_classes)

        if prompt_state_dict is not None:
            try:
                model.load_prompt(prompt_state_dict)
            except:
                print('erro in .pth prompt_state_dict')
            else:
                print('in prompt model building, a .pth prompt_state_dict loaded')

        if VPT_type == "Deep" or VPT_type == "Shallow":
            model.Freeze()
    else:
        print("The model is not difined in the Prompt script！！")
        return -1

    try:
        img = torch.randn(1, 3, edge_size, edge_size)
        preds = model(img)  # (1, class_number)
        print('test model output：', preds.shape)
        print('model forward checked')

    except:
        print("Problem exist in the model defining process！！")
        return -1
    else:
        print('model is ready now!')
        return model
