from langchain_ollama import OllamaLLM

# Initialize the LLM with the updated import
re_write_llm = OllamaLLM(model="mistral")

# Create a prompt template for query rewriting
query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 

                            Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

                            Original query: {original_query}

                            Rewritten query:"""

def rewrite_query(original_query):
    response = re_write_llm.invoke(query_rewrite_template.format(original_query=original_query))
    return response.strip()


