from PIL import Image
from torchvision import transforms
import torch
import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from ModelBase.WSI_models.gigapath import load_tile_slide_encoder, run_inference_with_tile_encoder
except:
    from PuzzleAI.ModelBase.gigapath.Inference_pipeline import load_tile_slide_encoder, run_inference_with_tile_encoder

# Please set your Hugging Face API token
os.environ["HF_TOKEN"] = "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"

assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

def check_tile_embedding_model(tile_encoder):
    print("param #", sum(p.numel() for p in tile_encoder.parameters()))

    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # load a specific sample tile to see if the embedding of the pretrained model is correct
    img_path = "../Archive/images/prov_normal_000_1.png"
    sample_input = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        output = tile_encoder(sample_input).squeeze()
        print("Model output:", output.shape)
        print(output)

    expected_output = torch.load("../Archive/images/prov_normal_000_1.pt")
    print("Expected output:", expected_output.shape)
    print(expected_output)

    assert torch.allclose(output, expected_output, atol=1e-2)
    print('the embedding of the pretrained model is correct as expected')


# Load the tile and slide_feature encoder models (fixme change to manual later)
# NOTE: The CLS token is not trained during the slide_feature-level pretraining.
# Here, we enable the use of global pooling for the output embeddings.
tile_encoder, slide_encoder_model = load_tile_slide_encoder(global_pool=True)

check_tile_embedding_model(tile_encoder)

# Load the tile images
local_dir = os.path.join('..', "demo/")
slide_name = 'PROV-000-000001.ndpi'
slide_dir = os.path.join(local_dir, 'outputs', 'preprocessing', 'output', slide_name)  # todo fix this sb path design
image_paths = [os.path.join(slide_dir, img) for img in os.listdir(slide_dir) if img.endswith('.png')]

print(f"Found {len(image_paths)} image tiles")

# inference
tile_encoder_outputs = run_inference_with_tile_encoder(image_paths, tile_encoder)

for k in tile_encoder_outputs.keys():
    print(f"tile_encoder_outputs[{k}].shape: {tile_encoder_outputs[k].shape}")