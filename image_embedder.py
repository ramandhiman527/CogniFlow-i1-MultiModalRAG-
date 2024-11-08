from transformers import ViTFeatureExtractor, ViTModel
import torch
from PIL import Image

class ImageEmbedder:
    def __init__(self):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    
    def __call__(self, image_path):
        """Make the class callable - handles single image path"""
        image = Image.open(image_path).convert("RGB")
        return self.embed_page(image)
    
    def embed_documents(self, texts):
        """Required by Chroma - will handle image paths as texts"""
        embeddings = []
        for text in texts:
            image = Image.open(text).convert("RGB")
            embedding = self.embed_page(image)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text):
        """Required by Chroma for similarity search"""
        return self.embed_documents([text])[0]
    
    def embed_page(self, page_image):
        """Generate embeddings for a given page image."""
        # Prepare image features
        inputs = self.feature_extractor(images=page_image, return_tensors="pt")
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Convert to numpy, then to list, and flatten the embedding
            embeddings = outputs.last_hidden_state.numpy()
            # Take mean across the sequence length dimension to get a single vector
            embedding = embeddings[0].mean(axis=0).tolist()  # Convert to Python list
            
        return embedding

def get_embedding_function():
    """Returns an instance of the embedding function class"""
    return ImageEmbedder()
