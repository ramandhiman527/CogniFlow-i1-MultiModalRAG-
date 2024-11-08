import torch
import clip
from PIL import Image

# Load the CLIP model and the preprocessor
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)  # You can choose a different model here


def get_image_embedding(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features

def get_text_embedding(text):
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features

    
def get_embedding_function(data_type):
    if data_type == "text":
        return get_text_embedding  
    elif data_type == "image":
        return get_image_embedding
    

    #this is the advanced embedding function to generate image and text embeddings from different mdoels to produce better embeddings