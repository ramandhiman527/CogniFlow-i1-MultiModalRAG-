import argparse
import os
import gradio as gr
import shutil
import uuid
from db_image_ingestion import add_to_chroma, clear_database, process_images
from document_preprocessor import load_documents
from text_chunker import split_documents
from query_processor import query_rag
from config import DATA_PATH, CHROMA_PATH
from langchain.schema import Document

class WebInterface:
    def __init__(self):
        self.data_path = DATA_PATH
        self.chroma_path = CHROMA_PATH

    def reset_database(self):
        print("✨ Clearing Database")
        clear_database()

    def train_model(self, file_paths):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        
        try:
            # Process documents
            documents = load_documents(self.data_path)
            # Generate unique IDs for each document before splitting
            for doc in documents:
                if "id" not in doc.metadata:
                    doc.metadata["id"] = str(uuid.uuid4())
            
            # Split documents into chunks while preserving metadata
            chunks = split_documents(documents)
            
            # Add to database
            if add_to_chroma(chunks):
                return "Training completed successfully."
            else:
                return "Error occurred during database addition."
        except Exception as e:
            return f"An error occurred during training: {e}"

    def test_model(self, query):
        try:
            result = query_rag(query)
            response_text = result.get("Response", "No response generated.")
            sources = result.get("sources", [])
            formatted_response = f"Response: {response_text}\n\nSources: {sources}"
            return formatted_response
        except Exception as e:
            return f"An error occurred during testing: {e}"

    def create_interface(self):
        # Define the train interface
        with gr.Blocks() as train_interface:
            gr.Markdown("# Train Model")
            with gr.Row():
                with gr.Column():
                    file_upload = gr.File(
                        label="Upload Training Data",
                        file_count="multiple",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                        type="filepath"
                    )
                    train_button = gr.Button("Train")
                with gr.Column():
                    train_response = gr.Textbox(label="Training Response", lines=5)
            
            train_button.click(self.train_model, inputs=file_upload, outputs=train_response)

        # Define the test interface
        with gr.Blocks() as test_interface:
            gr.Markdown("# Test Model")
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(label="Enter Query", lines=2)
                    test_button = gr.Button("Test")
                with gr.Column():
                    test_response = gr.Textbox(label="Test Response", lines=5)
            
            test_button.click(self.test_model, inputs=query_input, outputs=test_response)

        # Combine the interfaces into a single app
        return gr.TabbedInterface([train_interface, test_interface], ["Train", "Test"])

def main():
    interface = WebInterface()
    
    parser = argparse.ArgumentParser(description="RAG System Interface")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        interface.reset_database()

    app = interface.create_interface()
    app.launch()

if __name__ == "__main__":
    main()