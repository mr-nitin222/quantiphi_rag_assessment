'''
Bio Rag : An application to answer questions from
biology textbook using a rag pipeline.

This code uses saved embeddings of chunks taken from textbook

created by : Nitin Mishra
created date: 15th October 2024
'''

# Importing required libraries
import os
import pickle
from tqdm.auto import tqdm
import random
from time import perf_counter as timer

import fitz
import pandas as pd
import numpy as np
from spacy.lang.en import English
import re

from sentence_transformers import SentenceTransformer, util
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# Function to load presaved embeddings of textbook chunks
def load_embedding_chunks(path: str):
    '''
    This function loads the saved text embeddings of the textbook chunks

    args:
        path -> path of saved embeddings

    returns:
        pages_and_chunks: a list of dictionaries containing the text chunk, it's metadata and embeddings
    '''
    with open(path, 'rb') as handle:
        pages_and_chunks = pickle.load(handle)

    return pages_and_chunks

    
# Checking for GPU, if not available run through CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

    
# Function for loading embedding model
def load_embedding_model():
    '''
    Function to load embedding model in memory and allocate to available device
    Returns: embedding model object
    '''
    # Loading embedding model
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device = device)

    return embedding_model

# Load embedding model
embedding_model = load_embedding_model()


# Function to retrieve top k chunks similar to user query
def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                n_resources_to_return: int=5,
                                print_time: bool=True):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query, 
                                   convert_to_tensor=True) 

    # Get dot product scores on embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

    scores, indices = torch.topk(input=dot_scores, 
                                 k=n_resources_to_return)

    return scores, indices


# Function for loading llm model and tokenizer
def load_llm(model_id:str):
    '''
    Function which takes model id and returns it's tokenizer and model object
    '''
    # Defining tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
    
    # 4. Instantiate the model
    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, 
                                                     torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=False, # use full memory 
                                                     ).to(device)

    return tokenizer, llm_model

# Load llm model and it's tokenizer
tokenizer, llm_model = load_llm(model_id = "google/gemma-2b-it")


# Function to format user query into a prompt which contains context. Ready to be fed to llm model
def prompt_formatter(query: str, context_items: list[dict]) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: Where do plants get each of the raw materials required for photosynthesis?
Answer: Plants require the following raw material for photosynthesis:\n1. CO2 is obtained from the atmosphere through stomata\n2. Water is absorbed by plant roots from the soil.\n3. Sunlight is an essential raw material for photosynthesis\n4. Nutrients are obtained by soil by plant roots
\nExample 2:
Query: Why are some substances biodegradable and some non-biodegradable?
Answer: The reason why some substances are biodegradable and some are non-biodegradable is because the microorganisms, like bacteria, and decomposers, like saprophytes, have a specific role to play. They can break down only natural products like paper, wood, etc., but they cannot break down human-made products like plastics. Based on this, some substances are biodegradable and some are non-biodegradable.
\nExample 3:
Query: Why is DNA copying an essential part of the process of reproduction?
Answer: DNA copying is an essential part of the process of reproduction because it carries the genetic information from the parents to offspring. A copy of DNA is produced through some chemical reactions resulting in two copies of DNA. Along with the additional cellular structure, DNA copying also takes place, which is then followed by cell division into two cells.
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

    # Update base prompt with context items and query   
    prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [{"role": "user",
                          "content": prompt}]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt


# Load saved chunks from textbook and their embeddings
pages_and_chunks = load_embedding_chunks(r"embeddings/text_chunks_and_embeddings.pkl")

# Convert dictionary to dataframe
text_chunks_and_embedding_df = pd.DataFrame(pages_and_chunks)

# Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()),  dtype=torch.float32).to(device)


# Function which takes a query, retrieve relevent context, generates prompt and gives llm response
def ask(query, 
        temperature=0.7,
        max_new_tokens=512,
        format_answer_text=True, 
        return_answer_only=True):
    """
    Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
    """
    
    # Get just the scores and indices of top related results
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  print_time=False)
    
    # Create a list of context items
    context_items = [pages_and_chunks[i] for i in indices]

    # Add score to context item
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu() # return score back to CPU 
        
    # Format the prompt with context items
    prompt = prompt_formatter(query=query,
                              context_items=context_items)
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate an output of tokens
    outputs = llm_model.generate(**input_ids,
                                 temperature=temperature,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens)
    
    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])

    if format_answer_text:
        # Replace special tokens and unnecessary help message
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here's the answer to the user's query:\n\n", "")

    # Only return the answer without the context items
    if return_answer_only:
        return output_text
    
    return output_text, context_items


# Main function for the whole app
def main(query):
    '''
    Main function for the whole app
    '''
    answer = ask(query=query, 
                 temperature=0.7,
                 max_new_tokens=512,
                 return_answer_only=True)


    return answer


if __name__ == '__main__':
    main()