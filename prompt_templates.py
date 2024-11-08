# Updated Few-shot Prompt Template with examples
FEW_SHOT_PROMPT_TEMPLATE = """
                            You are a helpful assistant that answers questions based on the given context.

                            Here's an example of how to answer a question based on a context:

                            Context: The ship encountered an electrical failure on 01/01/2024 and the issue was resolved on 05/01/2024 by replacing the Magnetron.
                            Question: When did the defect occur?
                            Answer: The defect occurred on 01/01/2024.

                            Now, I'll provide you with a new context and question. Please answer the question based on the context provided.

                            Context: The repair of System 1 was successfully completed by Ram Vilas from Repair Unit 1.
                            Question: Who resolved the defect?
                            Answer: The defect was resolved by Ram Vilas from Repair Unit 1.

                            Now, here's your new context and question:

                            Context: {context}
                            Question: {question}

                            Answer based on the context:

                            The answer will be generated by the language model based on the context and the question. It might look something like this:

                            Answer: Based on the context, the defect was resolved by [name] from [repair unit].
                            """


image_description_prompt =  """You are an assistant tasked with summarizing tables, images and text for retrieval. \
                                These summaries will be embedded and used to retrieve the raw text or table elements \
                                Give a concise summary of the table or text that is well optimized for retrieval. Table or text or image: {image}"""