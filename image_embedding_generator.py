from typing import List, Optional, Dict
import torch
from PIL import Image
import logging

class ImageEmbeddingGenerator:
    def __init__(self, model_name: str = "vit-base-patch16-224"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model(model_name)
        self.batch_size = 32
        self._setup_logging()

    def _initialize_model(self, model_name: str):
        try:
            model = ViTModel.from_pretrained(model_name).to(self.device)
            return model
        except Exception as e:
            logging.error(f"Error initializing model {model_name}: {str(e)}")
            raise e

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

    def generate_embeddings(self, image_paths: List[str]) -> Dict[str, Optional[torch.Tensor]]:
        """Batch process images with better error handling and memory management"""
        results = {}
        for batch in self._batch_generator(image_paths):
            try:
                with torch.no_grad():
                    embeddings = self._process_batch(batch)
                    results.update(embeddings)
            except Exception as e:
                logging.error(f"Batch processing error: {str(e)}")
        return results

    def _process_batch(self, image_paths: List[str]) -> Dict[str, torch.Tensor]:
        images = []
        valid_paths = []
        
        for img_path in image_paths:
            try:
                img = self._load_and_preprocess_image(img_path)
                images.append(img)
                valid_paths.append(img_path)
            except Exception as e:
                logging.warning(f"Failed to process {img_path}: {str(e)}")
                
        if not images:
            return {}
            
        batch_tensor = torch.stack(images).to(self.device)
        outputs = self.model(batch_tensor)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

        return {path: emb.cpu() for path, emb in zip(valid_paths, embeddings)}

    def _load_and_preprocess_image(self, img_path: str) -> torch.Tensor:
        image = Image.open(img_path).convert("RGB")
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        inputs = feature_extractor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)  # Remove batch dimension

    def _batch_generator(self, items: List[str], batch_size: int = 32):
        """Yield successive batches from the list of items."""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size] 