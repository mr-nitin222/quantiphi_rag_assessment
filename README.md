# BIO RAG 

All designed to run locally on a NVIDIA GPU.

All the way from PDF ingestion to "chat with PDF" style features.

All using open-source tools.

In our specific example, we'll build Bio_RAG, a RAG workflow that allows a person to query a PDF version of a Concepts of Biology Textbook and have an LLM generate responses back to the query based on passages of text from the textbook.

PDF source: https://openstax.org/details/books/concepts-biology

You can also run notebook `bio_rag.ipynb` directly in your local machine after creating a new environment and installing all modules listed in requirements.txt. 

## Getting Started

Two main options:
1. If you have a local NVIDIA GPU with 5GB+ VRAM, follow the steps below to have this pipeline run locally on your machine. 
2. If you don’t have a local NVIDIA GPU, you can follow along in Google Colab and have it run on a NVIDIA GPU there. 

## Setup

Note: Tested in Python 3.12, running on Windows 11 with a NVIDIA RTX 3060 (6 GB VRAM) with CUDA 12.4.

### Create environment

```
python -m venv venv
```

### Activate environment

Windows: 
```
.\venv\Scripts\activate
```

### Install requirements

```
pip install -r requirements.txt
```

**Note:** I found I had to install `torch` manually see: https://pytorch.org/get-started/locally/

On Windows I used:

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

or

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Launch notebook

Launch jupyter notebook or jupyter lab and set working directory to the bio_rag.ipynb directory.
Then launch the bio_rag.ipynb.

**Note:** 
* Before running the streamlit frontend "app.py" or "bio_rag.py" for the first time, run the "bio_rag.ipynb" notebook till the first part that is saving the textbook embeddings. 
* Download and save the pdf in data directory before running the bio_rag.ipynb.

```
jupyter notebook
```

**Setup notes:** 
* If you run into any install/setup troubles, please leave an issue.
* To get access to the Gemma LLM models, you will have to [agree to the terms & conditions](https://huggingface.co/google/gemma-7b-it) on the Gemma model page on Hugging Face. You will then have to authorize your local machine via the [Hugging Face CLI/Hugging Face Hub `login()` function](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication). Once you've done this, you'll be able to download the models. If you're using Google Colab, you can add a [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens) to the "Secrets" tab.



**Possible Improvements**
1. In this pipline, different embedding models and llms can be used. Due to time and resources constraint I did not reseach much over selecting the best model.
2. For faster processing, only one unit of the textbook is processed. Given the time the same pipeline can be made more robust.
3. Given the time further improvements may be done in preprocessing part to include unneccessarry pages from the text, which impart noise in the vector database.
4. Given the time hyper parameter tuning of the llm and embedding models can be done to improve results.


