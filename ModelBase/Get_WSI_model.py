"""
Build WSI level models     Script  ver： Aug 17th 12:00
"""
from ModelBase.gigapath.slide_encoder import *
import torch
import huggingface_hub

def build_WSI_backbone_model(model_name='gigapath',local_weight_path=None,ROI_feature_dim=1536, **kwargs):

    if model_name == 'gigapath':
        slide_backbone = gigapath_slide_enc12l768d(in_chans=ROI_feature_dim, global_pool=False, **kwargs)
        print("Slide encoder param #", sum(p.numel() for p in slide_backbone.parameters()))

        if local_weight_path is None:
            # Hugging Face API token
            os.environ["HF_TOKEN"] = "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"
            hf_hub= "hf_hub:prov-gigapath/prov-gigapath"
            hub_name = hf_hub.split(":")[1]
            local_dir= os.path.join(os.path.expanduser("~"), ".cache/")

            huggingface_hub.hf_hub_download(hub_name, filename="slide_encoder.pth",
                                            local_dir=local_dir, force_download=True)
            local_weight_path = os.path.join(local_dir, "slide_encoder.pth")

        # build weight for slide level
        if os.path.exists(local_weight_path):
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

        return slide_backbone


def build_WSI_task_model(model_name='gigapath',local_weight_path=None,ROI_feature_dim=1536):
    slide_backbone = build_WSI_backbone_model(model_name,local_weight_path,ROI_feature_dim)
    # todo build task heads
    return slide_backbone