import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from text_embeddings import get_embedding_function
from langchain_chroma import Chroma
from document_preprocessor import load_documents
from text_chunker import split_documents
from config import DATA_PATH, CHROMA_PATH
import uuid


def main(data_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents(data_path)
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def preprocess_pdf(file_path):
    # Extract text from the PDF file
    text = ""
    with fitz.open(file_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text += page.get_text()

    # Clean and structure the extracted text
    cleaned_text = clean_text(text)
    structured_data = structure_data(cleaned_text, file_path)

    return Document(page_content=structured_data, metadata={"source": file_path})


def clean_text(text):
    cleaned_text = " ".join(text.split())
    return cleaned_text


def structure_data(text, source):
    # Assuming the structure is predictable, we can split the text into sections.
    # Here's a basic example of how to structure the data
    sections = text.split("Defect Analysis")
    if len(sections) < 2:
        return text  # If the structure is unexpected, return raw text.

    basic_info = sections[0].strip()
    defect_analysis = sections[1].strip().split("Defect Resolution")[0].strip()
    defect_resolution = sections[1].split("Defect Resolution")[1].strip() if "Defect Resolution" in sections[1] else ""

    # Structure it into a format
    structured = f"Basic Information:\n{basic_info}\n\nDefect Analysis:\n{defect_analysis}\n\nDefect Resolution:\n{defect_resolution}"
    return structured


def add_to_chroma(chunks):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Ensure each chunk has an ID
    chunks_with_ids = []
    for chunk in chunks:
        if "id" not in chunk.metadata:
            chunk.metadata["id"] = f"{chunk.metadata.get('source', 'unknown')}:{uuid.uuid4()}"
        chunks_with_ids.append(chunk)

    # Get IDs for new chunks
    chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
    
    try:
        db.add_documents(chunks_with_ids, ids=chunk_ids)
        print(f"✅ Successfully added {len(chunks_with_ids)} documents to database")
        return True
    except Exception as e:
        print(f"Error adding documents to database: {e}")
        return False


def calculate_chunk_ids(chunks):
    last_source = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")

        if source == last_source:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{source}:{current_chunk_index}"
        last_source = source
        chunk.metadata["id"] = chunk_id  # Ensure 'id' is set

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("✅ Database cleared successfully.")


if __name__ == "__main__":
    main(DATA_PATH)
