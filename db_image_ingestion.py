import os
import shutil
import uuid
from PIL import Image
from langchain.schema import Document
from langchain_chroma import Chroma
from image_embedder import get_embedding_function
from config import DATA_PATH, CHROMA_PATH

class DatabaseOperations:
    def __init__(self):
        self.data_path = DATA_PATH
        self.chroma_path = CHROMA_PATH

    def clear_database(self):
        """Clear the existing Chroma database if it exists."""
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)
            print("✅ Database cleared successfully.")

    def process_images(self):
        """Process images and generate their embeddings for storage in Chroma."""
        image_files = [f for f in os.listdir(self.data_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        ids = []
        documents = []
        
        for img_file in image_files:
            img_path = os.path.join(self.data_path, img_file)
            try:
                doc_id = str(uuid.uuid4())
                metadata = {
                    "name": img_file,
                    "tag": "image",
                    "source": img_path,
                    "id": doc_id
                }
                ids.append(doc_id)
                documents.append(Document(page_content=img_path, metadata=metadata))
                    
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
        
        return ids, documents

    def add_to_chroma(self, ids, documents):
        """Add processed image data to Chroma database."""
        try:
            db = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=get_embedding_function()
            )
            
            db.add_documents(
                documents=documents,
                ids=ids,
            )
            
            print(f"✅ Successfully added {len(documents)} images to Chroma database.")
            return True
        except Exception as e:
            print(f"Error adding to database: {str(e)}")
            return False

# Create a global instance
db_ops = DatabaseOperations()

# Export functions for external use
clear_database = db_ops.clear_database
add_to_chroma = db_ops.add_to_chroma
process_images = db_ops.process_images
