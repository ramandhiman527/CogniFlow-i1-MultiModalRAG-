import fitz  # PyMuPDF
from PIL import Image
import io
import os
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.schema import Document

def preprocess(file_path):
    text_data = []
    img_data = []
    with fitz.open(file_path) as pdf_file:
        # Create a directory to store the images
        if not os.path.exists("extracted_images"):
            os.makedirs("extracted_images")

        # Loop through every page in the PDF
        for page_number in range(len(pdf_file)):
            page = pdf_file[page_number]
            
            # Get the text on page
            text = page.get_text().strip()
            
            text_data.append({
                "page_content": text, 
                "metadata": {
                    "source": file_path,
                    "page": page_number+1
                }
            })
            
            images = page.get_images(full=True)

            # Loop through all images found on the page
            for image_index, img in enumerate(images, start=0):
                xref = img[0]  
                base_image = pdf_file.extract_image(xref)  
                image_bytes = base_image["image"] 
                image_ext = base_image["ext"]
                
                image = Image.open(io.BytesIO(image_bytes))
                image_path = f"extracted_images/image_{page_number+1}_{image_index+1}.{image_ext}"
                image.save(image_path)

                from prompt import image_description_prompt
                prompt_template = ChatPromptTemplate.from_template(image_description_prompt)
                prompt = prompt_template.format(image=image_path)
                model = OllamaLLM(model="llava")
                response = model.invoke(prompt)
                
                img_data.append({
                    "page_content": response, 
                    "metadata": {
                        "source": file_path,
                        "image_path": image_path,
                        "page": page_number+1
                    }
                })

    return text_data, img_data

def load_documents(data_path):
    documents = []
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        text_data, img_data = preprocess(file_path)
        
        # Convert text_data and img_data to Document objects
        for item in text_data + img_data:
            documents.append(Document(
                page_content=item["page_content"],
                metadata=item["metadata"]
            ))
    
    return documents

