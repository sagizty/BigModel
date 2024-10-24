import torch
from PIL import Image
from transformers import AutoTokenizer
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig

def main():
    # Select device
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    model_name = "microsoft/llava-med-v1.5-mistral-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = LlavaMistralConfig.from_pretrained(model_name)
    language_model = LlavaMistralForCausalLM.from_pretrained(
        model_name, config=config)

    # Move the entire model to the device
    language_model.to(device)

    # Access the vision tower and load it
    vision_tower = language_model.get_model().get_vision_tower()
    vision_tower.load_model()

    # Move the vision tower to the same device
    vision_tower.to(device)

    # Get the image processor from the vision tower
    image_processor = vision_tower.image_processor

    # Check devices of model components
    print(f"language_model device: {next(language_model.parameters()).device}")
    print(f"vision_tower device: {next(vision_tower.parameters()).device}")
    print(f"mm_projector device: {next(language_model.get_model().mm_projector.parameters()).device}")


    # Load and preprocess the image using the model's image processor
    image_path = '/mnt/data1/changhan/Hist_VQA/Path_VQA/Images/image_00001.png'
    image = Image.open(image_path).convert("RGB")
    processed_image = image_processor(images=image, return_tensors="pt")['pixel_values'].to(device)

    # # Print the size of the processed image tensor
    # print(f"Processed image shape: {processed_image.shape}")
    # # Check devices of data tensors
    # print(f"processed_image device: {processed_image.device}")
    


    # Example text input
    text_input = "What is this picture about?"

    # Tokenize the text input to get token IDs
    tokenized_input = tokenizer(text_input, return_tensors="pt").to(device)
    input_ids = tokenized_input["input_ids"]

    # Generate text output from the language model with multimodal input
    text_output = language_model.generate(inputs=input_ids, images=processed_image)

    # Decode the generated output back into text
    generated_text = tokenizer.decode(text_output[0], skip_special_tokens=True)
    print(image_path)
    print(text_input)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()
