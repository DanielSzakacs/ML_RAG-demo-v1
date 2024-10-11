import os
import requests
import torch
import fitz
from tqdm.auto import tqdm
import pandas as pd
import re
import random
import numpy as np
import textwrap
import matplotlib.pyplot as plt
import gradio as gr
import torch
import spacy
from spacy.lang.en import English
import transformers
import sentence_transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import util, SentenceTransformer
import huggingface_hub
from huggingface_hub import login
from transformers import pipeline



# Set up the device
def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Set device to: {device}")
  return device
device = set_device()


def download_pdf(filename:str,
                 dir_path: str,
                 pdf_url:str):
  if not os.path.exists(dir_path):
    # Create the directory for the PDF
    os.makedirs(dir_path)
    print(f"[INFO] {dir_path} directory created")

  if not os.path.exists(dir_path + filename):
    # Send GET request to the url
    response = requests.get(pdf_url)
    if response.status_code == 200:
      # Open and save the pdf
      with open(dir_path + filename, "wb") as file:
        file.write(response.content)
      print(f"[INFO] {filename} downloaded")
    else:
      print(f"[ERROR] {filename} could not be downloaded")
  else:
    print("[INFO] PDF file exist")


filename = "businessAnalysis.pdf"
dir_path="./pdf_source/"
pdf_url = "https://raw.githubusercontent.com/DanielSzakacs/RAG-demo-v1/main/source/businessAnalysis.pdf"
full_pdf_path = dir_path + filename

download_pdf(filename=filename,
             dir_path=dir_path,
             pdf_url=pdf_url)


# Create NPL to split the text into sentences
nlp = English()
# Add module to recognice sentences
nlp.add_pipe("sentencizer")


def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

def open_and_read_pdf(pdf_path:str):
  doc = fitz.open(pdf_path)
  pages_and_text = []

  for page_number, page in tqdm(enumerate(doc)):
    text = text_formatter(page.get_text())
    pages_and_text.append({"page_number": page_number + 1,
                           "page_char_number": len(text),
                           "page_word_number": len(text.split(" ")),
                           "page_sentence_number": len(text.split(". ")),
                           "page_token_number": len(text) / 4, # Most of the time 4 chart is 1 token
                           "text" : text})

  return pages_and_text

pages_and_text = open_and_read_pdf(pdf_path=full_pdf_path)

for item in tqdm(pages_and_text):
  item["sentences"] = list(nlp(item["text"]).sents)
  # Make sure all sentences are strings (deault tyoe is spaCy datatype)
  item["sentences"] = [str(sentence) for sentence in item["sentences"]]
  # Count the sentences
  item["page_sentence_count_spacy"] = len(item["sentences"])

# Turn it into dataframe

df = pd.DataFrame(pages_and_text)

NUMBER_OF_CHUNK = 5

def split_list(input_list: list,
               slice_size: int) -> list[list[str]]:
                """
                Splits the input_list into sublists of size slice_size (or as close as possible).

                For example, a list of 17 sentences would be split into two lists of [[10], [7]]
                """
                return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

# Split sentences into chunks
for item in tqdm(pages_and_text):
  item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                       slice_size=NUMBER_OF_CHUNK)
  item["number_of_chunks"] = len(item["sentence_chunks"])



# Split each chunk into its own item
pages_and_chunks = []
for item in tqdm(pages_and_text):
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]

        # Join the sentences together into a paragraph-like structure (so they are a single string)
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo
        chunk_dict["sentence_chunk"] = joined_sentence_chunk

        # Get stats about the chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters

        pages_and_chunks.append(chunk_dict)

df = pd.DataFrame(pages_and_chunks)

# Create embedding model
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device=device) #all-mpnet-base-v2

for item in tqdm(pages_and_chunks):
  item["embedding"] = embedding_model.encode(item["sentence_chunk"])

# Save the embedding with all the usefull information
text_chunks_and_embedding_df = pd.DataFrame(pages_and_chunks)
text_chunks_and_embedding_df.to_csv("./pages_and_chunks.csv", index=False)

# Convert texts and embedding df to list of dicts
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

# Convert embedding to torch tensor and send to device
embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)

def get_most_relavent_resources(query:str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                n_resources_to_return: int=5,
                                print_time: bool=True):
    # Embed the query
    query_embedding = model.encode(query,
                                   convert_to_tensor=True)

    # Get dot product scores on embeddings
    dot_scores = util.dot_score(query_embedding, embeddings)[0]

    scores, indices = torch.topk(input=dot_scores,
                                 k=n_resources_to_return)
    return scores, indices


huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
model_name = "google/gemma-7b-it"
print(f"Morel name : {model_name}")

login(token=huggingface_token)
# Set up a text generation pipeline using a hosted model
generator = pipeline(
    "text-generation",
    model=model_name,
    device=device
)


# Create a tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
def promt_format(query: str,
                 context_items: list[dict]) -> str:
                  context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

                  base_promt = """
                  Based on the following context items, please answer the query.
                  Don't return the thinking, only return the answer.
                  Make sure your answers are as explanatory as possible.
                  \nNow use the following context items to answer the user query:
                  {context}
                  \nRelevant passages: <extract relevant passages 
                  User query: {query}
                  Answer:
                  """

                  base_promt = base_promt.format(context=context, query=query)

                  template = [
                      {"role": "user",
                      "content": base_promt}
                  ]

                  prompt = tokenizer.apply_chat_template(conversation=template,
                                          tokenize=False,
                                          add_generation_prompt=True)
                  return prompt

def extract_answer_from_generated_text(generated_text:str)->str:
  if "Answer:" in generated_text:
      answer_part = generated_text.split("Answer:")[1]
  elif "<start_of_turn>model" in generated_text:
      # This case is specific to your generated format
      answer_part = generated_text.split("<start_of_turn>model")[1]
  else:
      # If the specific markers aren't found, return the entire generated text
      answer_part = generated_text

  
  answer_part = answer_part.replace("<end_of_turn>", "").replace("<start_of_turn>model\nSure, here's the answer to the user's query:", "").replace("<start_of_turn>model", "").strip()
  return answer_part

def ask(query, n_resources_to_return=5, return_answer_only=True):
    # Get the relevant context items (as in your previous implementation)
    scores, indices = get_most_relavent_resources(query=query, embeddings=embeddings)
    context_items = [pages_and_chunks[i] for i in indices]

    # Format the prompt with context items
    prompt = promt_format(query=query, context_items=context_items)

    # Generate the response using the hosted model
    result = generator(prompt, max_new_tokens=712, num_return_sequences=1, temperature=0.7)

    # Extract the generated text from the result
    output_text = extract_answer_from_generated_text(result[0]["generated_text"])

    if return_answer_only:
        return output_text.strip()
    return output_text, context_items


def show_relavent_page(context):
  page_number = context[0]["page_number"]
  # Open the pdf with fitz
  doc = fitz.open(full_pdf_path)
  page = doc.load_page(page_number - 1)

  # Get the image of the page
  img = page.get_pixmap(dpi=300)

  # Convert the Pixmap to a numpy array
  img_array = np.frombuffer(img.samples_mv,
                            dtype=np.uint8).reshape((img.h, img.w, img.n))

  return img_array


# Create a function to be called by the Gradio interface
def gradio_ask(query):
    # Call the ask function with the query and return the result
    answer, context_items = ask(query=query,
                                return_answer_only=False)
    image_array = show_relavent_page(context_items)
    return answer, image_array


demo = gr.Interface(
    fn=gradio_ask,
    inputs="text",
    outputs=["text", "image"],
    title="RAG-demo-v1",
    description="""Before asking Please check the source folder. Topic : Business Analysis<br><br>
    For source: <a href='https://github.com/DanielSzakacs/RAG-demo-v1/blob/main/source/businessAnalysis.pdf'>click here</a><br>
    Download source: <a href='https://raw.githubusercontent.com/DanielSzakacs/RAG-demo-v1/main/source/businessAnalysis.pdf'>click here</a><br>
    Jupyter notebook: <a href='https://github.com/DanielSzakacs/RAG-demo-v1/blob/main/rag_pet_project.ipynb'>click here</a>""",
    examples=["What is the Porter’s value chain?", "What is the range of business analysis ?"]
)

# Launch the Gradio interface
demo.launch()
