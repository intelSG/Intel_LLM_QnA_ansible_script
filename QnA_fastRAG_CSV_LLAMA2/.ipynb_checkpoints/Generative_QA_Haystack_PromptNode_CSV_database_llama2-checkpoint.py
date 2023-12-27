#Generative QA with CSV database with LLAMA2-7B

from haystack.telemetry import tutorial_running
import os
import logging
import pandas as pd
from haystack.utils import fetch_archive_from_http
import time

print("Running Generative QA with CSV database with LLAMA2 in progess......")
print()

# get the start time
st = time.time()

print("Let's remove the old faiss_document_store.db......")
print()

os.remove("faiss_document_store.db")

print("Let's download the new CSV File")
print()

# Download sample
doc_dir = "Test_Data"
#document_location = "/home/smgailab/Intel/fastRAG/examples/excel_data"
# https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/small_generator_dataset.csv.zip

sharelink_url = input ("Please input the URL of your .CSV File:")
print()
print ("The .CSV file that we will download is from this: ", sharelink_url)
print()
fetch_archive_from_http(url=sharelink_url, output_dir=doc_dir)

print ("Fetching file from URL completed.....")
print()
#Test_Template_Allen.csv
#filename = input ("Input your csv filename here: ")
# Create dataframe with columns "title" and "text"
print("Using Pandas function to read the csv file in progress.....")
print()
df = pd.read_csv(f"{doc_dir}/small_generator_dataset.csv", sep=",")
# Minimal cleaning
df.fillna(value="", inplace=True)
print("Do a print out the first 5 rows of the content......")
print()
print(df.head(n=5))
print()

from haystack import Document

# Use data to initialize Document objects
titles = list(df["title"].values)
texts = list(df["text"].values)
documents = []
for title, text in zip(titles, texts):
    documents.append(Document(content=text, meta={"name": title or ""}))

from haystack.document_stores import FAISSDocumentStore

print()
print("Creating faiss_document_store.db.....")

document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import bfloat16

#hf_token="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"

model_path = "/home/smgailab/Intel/fastRAG/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(model_path) #, use_auth_token=hf_token)

tokenizer = AutoTokenizer.from_pretrained(model_path) #, use_auth_token=hf_token)

from haystack.nodes import PromptNode,PromptModel

import torch

from haystack.nodes import BM25Retriever, SentenceTransformersRanker, EmbeddingRetriever
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes import PromptNode, PromptTemplate
from haystack.nodes import BM25Retriever, SentenceTransformersRanker
from haystack.nodes.prompt.invocation_layer import HFLocalInvocationLayer

print("Define Pipeline = Retriever + Reranker + Prompt Node")
print()

retriever = EmbeddingRetriever(document_store = document_store,
                               embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
#retriever = BM25Retriever(document_store=document_store, top_k=100)
reranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2", top_k=1)



lfqa_prompt = PromptTemplate(name="lfqa",
                             prompt_text="Answer the question using the provided context. Your answer should be in your own words and be no longer than 50 words. \n\n Context: {join(documents)} \n\n Question: {query} \n\n Answer:",
                             output_parser={"type": "AnswerParser"}) 

# exotic configuration based on model_kwargs
# inspiration: https://docs.haystack.deepset.ai/docs/prompt_node
# using-models-not-supported-in-hugging-face-transformers
local_model=PromptModel(
    model_name_or_path="/home/smgailab/Intel/fastRAG/Llama-2-7b-chat-hf",
    invocation_layer_class=HFLocalInvocationLayer,
    model_kwargs={'task_name':'text-generation'}
    )

prompt = PromptNode(
                max_length=1000,
                model_name_or_path=local_model,
                default_prompt_template=lfqa_prompt,
                model_kwargs={#"model":model,
                              "tokenizer":tokenizer,
                              #'task_name':"text2text-generation",
                              'device':None, # placeholder needed to make the underlying HF Pipeline work,
                              'model_max_length': 2048, 
                              "torch_dtype": torch.bfloat16,
                              'stream':True})



# Delete existing documents in documents store
document_store.delete_documents()

# Write documents to document store
document_store.write_documents(documents)

# Add documents embeddings to index
document_store.update_embeddings(retriever=retriever)

from haystack import Pipeline
p = Pipeline()
p.add_node(component=retriever, name="Retriever", inputs=["Query"])
p.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
p.add_node(component=prompt, name="prompt_node", inputs=["Reranker"])

a = p.run(query="who got the first nobel prize in physics", debug=True)
a['answers'][0].answer

# get the end time
et = time.time()
print()
print()

# get the execution time
elapsed_time = et - st
print('Total execution time to setup the Question-Answer AI Demo with CSV Database:', elapsed_time, 'seconds')
print()
print()
print("Setting up Gradio UI Interface .................. It's ready to use now!")
print()
print()
print("This demo is powered by Intel(R) Xeon(R) Platinum 8480+")
print()
print()



import gradio as gr

title = "Generative QA with fastRAG with csv file powered Intel Xeon 4th Gen Processor"

#question = "who got the first nobel prize in physics"

#results = pipe.run(query=question, params={"Generator": {"top_k": 1}, "Retriever": {"top_k": 1}})
#results = pipe.run(query=question)
#print_answers(results, details="minimum")


QUESTIONS = [
    "who got the first nobel prize in physics",
    "When iphone XS was launched?",
    "when is the next deadpool movie being released",
    "which mode is used for short wave broadcast service",
    "who is the owner of reading football club",
    "when is the next scandal episode coming out",
    "when is the last time the philadelphia won the superbowl",
    "what is the most current adobe flash player version",
    "how many episodes are there in dragon ball z",
    "what is the first step in the evolution of the eye",
    "where is gall bladder situated in human body",
    "what is the main mineral in lithium batteries",
    "who is the president of usa right now",
    "where do the greasers live in the outsiders",
    "panda is a national animal of which country",
    "what is the name of manchester united stadium",
]
    


def predict (question):
    results = p.run(query=question,debug=True)
    output = results['answers'][0].answer
    #answers = print_answers (result)
    return output

   
gr.Interface (fn=predict,
            inputs = gr.inputs.Textbox(label="Question"),
            outputs = gr.outputs.Textbox(label="Answer"),
            examples=QUESTIONS,
            description = "Retriever Model: sentence-transformers/multi-qa-mpnet-base-dot-v1, Reranker Model: cross-encoder/ms-marco-MiniLM-L-12-v2, Prompt Model:Llama-2-7b-chat-hf",
            #outputs = "text",
            title=title
                  
            ).launch(share=True)





