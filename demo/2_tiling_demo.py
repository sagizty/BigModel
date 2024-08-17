# Crop patch ROIs  Aug 8th 17:00
import sys
import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import huggingface_hub
import os
import time

try:
    from DataPipe.Tiles_dataset import prepare_tiles_dataset_for_single_slide
except:
    from PuzzleAI.DataPipe.Tiles_dataset import prepare_tiles_dataset_for_single_slide

# Please set your Hugging Face API token
os.environ["HF_TOKEN"] = "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"

assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"
local_dir = os.path.join('../DownStream', "demo/")
huggingface_hub.hf_hub_download("prov-gigapath/prov-gigapath", filename="sample_data/PROV-000-000001.ndpi",
                                local_dir=local_dir, force_download=True)
slide_path = os.path.join(local_dir, "sample_data/PROV-000-000001.ndpi")

save_dir = os.path.join(local_dir, 'outputs/')

since = time.time()
print("NOTE: Prov-GigaPath is trained with 0.5 mpp preprocessed slides. "
      "Please make sure to use the appropriate level for the 0.5 MPP")
prepare_tiles_dataset_for_single_slide(slide_path, save_dir=save_dir)

time_elapsed = time.time() - since
print(f'cropping completed in {time_elapsed:.2f} seconds')

print("NOTE: tiling dependency libraries can be tricky to set up. Please double check the generated tile images.")
