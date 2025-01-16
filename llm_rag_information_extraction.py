from langchain_community.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
#from langchain.docstore.document import Document
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import torch
import os
from pathlib import Path
import re
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig, pipeline
import logging
import pandas as pd
import numpy as np
import nltk
import spacy


# Configure logging
logging.basicConfig(
    filename='faiss_debug_dev.log',  # Log file name
    filemode='a',                # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG           # Log level
)

# Load models and tokenizer
MODEL_PATH = "ISTA-DASLab/Meta-Llama-3.1-70B-AQLM-PV-2Bit-1x16"
#MODEL_PATH = "/home/oemerfar/models/Meta-Llama-3-8B-Instruct"
#MODEL_PATH = "/home/oemerfar/models/Meta-Llama-3.1-8B-Instruct"

# Load tokenizer and model
#tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
'''model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={'': 0},  # Adjust for device
    torch_dtype=torch.float16,  # Use float16 for optimization
    low_cpu_mem_usage=True,
)'''

# Quantization configurtaion
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Use 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={'': 0}, 
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
#tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up the embeddings model all-MiniLM-L6-v2
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_answer_part(output_text):
    # Look for the "ANSWER:" marker
    answer_start = output_text.find("ANSWER:")
    if answer_start == -1:
        return output_text.strip()
    # Start after the "ANSWER:" and get the remaining part
    answer_part = output_text[answer_start + len("ANSWER:"):].strip()
    # Remove any additional sections
    cutoff_markers = ["TASK:", "RULES:"]
    for marker in cutoff_markers:
        marker_pos = answer_part.find(marker)
        if marker_pos != -1:
            answer_part = answer_part[:marker_pos].strip()
    return answer_part
    
# Semantic chunking functions
def semantic_chunking_nltk(documents, similarity_threshold=0.75):
    nltk.download('punkt_tab')
    nltk.download('punkt')
    semantically_chunked_docs = []
    for document in documents:
        sentences = nltk.sent_tokenize(document.page_content) 
        embeddings = semantic_embedding_model.encode(sentences)

        current_chunk = []
        for i, sentence in enumerate(sentences):
            if len(current_chunk) == 0:
                current_chunk.append(sentence)
                continue

            # cosine similarity as similarity metric
            similarity = np.dot(embeddings[i], embeddings[i-1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1]))
            if similarity < similarity_threshold:
                semantically_chunked_docs.append(Document(page_content=". ".join(current_chunk)))
                current_chunk = [sentence]
            else:
                current_chunk.append(sentence)

        if current_chunk:
            semantically_chunked_docs.append(Document(page_content=". ".join(current_chunk)))
    
    return semantically_chunked_docs

def semantic_chunking_spacy(documents, similarity_threshold=0.75):
    nlp = spacy.load('en_core_web_sm')
    semantically_chunked_docs = []
    for document in documents:
        doc = nlp(document.page_content)
        sentences = [sent.text for sent in doc.sents]
        embeddings = semantic_embedding_model.encode(sentences)

        current_chunk = []
        for i, sentence in enumerate(sentences):
            if len(current_chunk) == 0:
                current_chunk.append(sentence)
                continue

            # cosine similarity as similarity metric
            similarity = np.dot(embeddings[i], embeddings[i-1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1]))
            if similarity < similarity_threshold:
                semantically_chunked_docs.append(Document(page_content=". ".join(current_chunk)))
                current_chunk = [sentence]
            else:
                current_chunk.append(sentence)

        if current_chunk:
            semantically_chunked_docs.append(Document(page_content=". ".join(current_chunk)))
    
    return semantically_chunked_docs

# Function to remove bibliography sections
def remove_bibliography_from_documents(documents):
    bibliography_headings = ["References", "Bibliography", "Reference", "BIBLIOGRAPHY", "REFERENCES"]
    pattern = re.compile(r'\b(' + '|'.join(bibliography_headings) + r')\b', re.IGNORECASE)
    new_documents = []
    for doc in documents:
        match = pattern.search(doc.page_content)
        if match:
            content_before_heading = doc.page_content[:match.start()]
            if content_before_heading.strip():
                doc.page_content = content_before_heading
                new_documents.append(doc)
            # Stop processing further documents
            break  # Assuming references are at the end
        else:
            new_documents.append(doc)
    return new_documents

# Step 1: Generate a summary of the observed effects
def generate_summary(llm, context):
    summary_prompt = PromptTemplate(
        input_variables=["context"],
        template='''
    CONTEXT:
    {context}

    TASK:
    Provide a concise summary of the observed functional grouping in results of the research.

    ANSWER:
    '''
    )
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    summary_output = summary_chain.run(context=context)

    answer_part_summary = extract_answer_part(summary_output)
    print("SUMMARY:", answer_part_summary)
    return answer_part_summary

# Step 2: Generate relevant frequency bands using the summary
def generate_bands(llm, context, question):
    relevant_bands_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template='''
    RULES:
    - Do NOT change the question or provide any additional explanations.
    - Context refers to the location of the brain regions
    - Only provide the output in the format specified below.
    - If specific frequency bands are mentioned in the context, only include these bands for relevant location.
    - Do NOT include any locations or frequency bands that are not explicitly mentioned in the context.
    - If no frequency band is mentioned in the context for any location, pair all frequency bands (Delta, Theta, Alpha, Beta, Gamma) with locations that are mentioned in the context.
    - If no location is mentioned in the context, return "NO LOCATION". 
    - Strictly follow the format: one line per Location; Frequency Band.
    
    CONTEXT:
    {context}

    QUESTION:
    {question}
    
    INSTRUCTIONS:
    1. Identify all locations mentioned in the context.
    2. Identify all frequency bands mentioned in the context.
    3. For each location, list the associated frequency bands.
    4. Output the results in the format "Location; Frequency Band" per line.
    
    OUTPUT FORMAT:
    - Location; Frequency Band

    IMPORTANT:
    - Strictly apply the output format, and each line should have one Location; Frequency Band pair.

    ANSWER:
    '''
    )
    relevant_bands_chain = LLMChain(llm=llm, prompt=relevant_bands_prompt)
    relevant_bands_output = relevant_bands_chain.run(context=context, question=question)
    relevant_bands = extract_answer_part(relevant_bands_output)
    print("RELEVANT FREQUENCY BANDS:", relevant_bands_output)
    return relevant_bands

# Chain of thought prompting 
def generate_bands_with_cot(llm, context, question):
    cot_prompt_template = '''
    RULES:
    - Do NOT change the question or provide any additional explanations.
    - Follow a step-by-step reasoning to identify brain regions and frequency bands, and then map them together.
    - Context refers to the location of the brain regions.
    - If specific frequency bands are mentioned in the context, only include these bands for relevant location.
    - Do NOT include any locations or frequency bands that are not explicitly mentioned in the context.
    - If no frequency band is mentioned in the context for any location, pair all frequency bands (Delta, Theta, Alpha, Beta, Gamma) with locations that are mentioned in the context.
    - If no location is mentioned in the context, return "NO LOCATION".

    CONTEXT:
    {context}

    QUESTION:
    {question}

    TASK:
    Step 1: Identify all brain regions mentioned in the context.
    Step 2: Identify all frequency bands mentioned in the context.
    Step 3: For each brain region identified, map the corresponding frequency bands.
    Step 4: Return the results in the format "Location; Frequency Band" per line.

    OUTPUT FORMAT:
    - One line per pair of "Location; Frequency Band".

    ANSWER:
    '''
    
    cot_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=cot_prompt_template
    )
    
    relevant_bands_chain = LLMChain(llm=llm, prompt=cot_prompt)
    relevant_bands_output = relevant_bands_chain.run(context=context, question=question)
    relevant_bands = extract_answer_part(relevant_bands_output)
    print("RELEVANT FREQUENCY BANDS (CoT):", relevant_bands_output)
    return relevant_bands

# Knowledge-augmentation prompting
def generate_bands_with_knowledge(llm, context, question, brain_region_knowledge, frequency_bands):
    knowledge_augmented_prompt_template = '''
    RULES:
    - You are provided with a list of brain regions and their associated electrode locations.
    - You are also provided with a list of frequency bands.
    - Use this knowledge to map each brain region to its corresponding frequency band based on the given CONTEXT.
    - If specific frequency bands are mentioned in the CONTEXT, only include these bands for relevant location.
    - Do NOT include any locations or frequency bands that are not explicitly mentioned in the CONTEXT.
    - If no frequency band is mentioned in the CONTEXT for any location, pair all frequency bands with brain regions that are mentioned in the CONTEXT.
    - If no location is mentioned in the CONTEXT, return "NO LOCATION".

    KNOWLEDGE:
    Brain Regions and Electrodes:
    {brain_region_knowledge}

    Frequency Bands:
    {frequency_bands}

    CONTEXT:
    {context}

    QUESTION:
    {question}

    TASK:
    Step 1: Identify all brain regions mentioned in the context.
    Step 2: Identify all frequency bands mentioned in the context.
    Step 3: For each brain region identified, map the corresponding frequency bands based on the knowledge provided.
    Step 4: Return the results in the format "Location; Frequency Band" per line.

    OUTPUT FORMAT:
    - One line per pair of "Location; Frequency Band".

    ANSWER:
    '''
    
    # Formatting the brain_region_knowledge and frequency_bands into the prompt
    knowledge = {
        "brain_region_knowledge": "\n".join([f"{region}: {', '.join(electrodes)}" for region, electrodes in brain_region_knowledge.items()]),
        "frequency_bands": ", ".join(frequency_bands)
    }
    
    # Defining the prompt
    knowledge_augmented_prompt = PromptTemplate(
        input_variables=["context", "question", "brain_region_knowledge", "frequency_bands"],
        template=knowledge_augmented_prompt_template
    )
    
    # Generating the LLM response with the injected knowledge
    relevant_bands_chain = LLMChain(llm=llm, prompt=knowledge_augmented_prompt)
    relevant_bands_output = relevant_bands_chain.run(
        context=context,
        question=question,
        brain_region_knowledge=knowledge["brain_region_knowledge"],
        frequency_bands=knowledge["frequency_bands"]
    )
    
    relevant_bands = extract_answer_part(relevant_bands_output)
    print("RELEVANT FREQUENCY BANDS (Knowledge Augmented):", relevant_bands_output)
    return relevant_bands


# Step 3: Formating the output
def format_bands(llm, context):
    formatted_bands_prompt = PromptTemplate(
        input_variables=["context"],
        template='''
    CONTEXT:
    {context}

    INSTRUCTIONS:
    - For each line in the CONTEXT, map any specific electrode locations to the corresponding brain region using the mapping below:
    * "Frontal": ['Fz', 'F1', 'F2', 'F3', 'F4', 'F7', 'F8', 'Fpz', 'Fp1', 'Fp2', 'AFz', 'AF3', 'AF4', 'AF7', 'AF8', 'FT9', 'FT10'],
    * "Central": ['Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CPz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6'],
    * "Motor": ['FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CPz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6'],
    * "Parietal": ['Pz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'TP7', 'TP8', 'TP9', 'TP10', 'POz', 'PO3', 'PO4', 'PO7', 'PO8'],
    * "Temporal": ['T7', 'T8', 'FT7', 'FT8', 'TP7', 'TP8'],
    * "Occipital": ['Oz', 'O1', 'O2', 'POz', 'PO3', 'PO4', 'PO7', 'PO8']

    - Ensure that each line follows the format: "Location; Frequency Band".
    - If a location is NOT as expected FORMAT, research on the internet and find which brain location (Frontal, Central, Motor, Parietal, Temporal, Occipital) does it belong. 
    - Output should contain only the following locations: Frontal, Central, Motor, Parietal, Temporal, Occipital.

    ANSWER:
    '''
    )
    formatted_bands_chain = LLMChain(llm=llm, prompt=formatted_bands_prompt)
    formatted_output = formatted_bands_chain.run(context=context)
    
    answer_part_formatted_output = extract_answer_part(formatted_output)
    print("FORMATTED OUTPUT:", answer_part_formatted_output)
    return answer_part_formatted_output

# Additional processing functions
def map_locations_to_regions(llm_output):
    location_groups = {
        "Frontal": ['Fz', 'F1', 'F2', 'F3', 'F4', 'F7', 'F8', 'Fpz', 'Fp1', 'Fp2', 'AFz', 'AF3', 'AF4', 'AF7', 'AF8', 'FT9', 'FT10'],
        "Central": ['Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CPz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6'],
        "Motor": ['FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CPz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6'],
        "Parietal": ['Pz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'TP7', 'TP8', 'TP9', 'TP10', 'POz', 'PO3', 'PO4', 'PO7', 'PO8'],
        "Temporal": ['T7', 'T8', 'FT7', 'FT8', 'TP7', 'TP8'],
        "Occipital": ['Oz', 'O1', 'O2', 'POz', 'PO3', 'PO4', 'PO7', 'PO8']
    }

    # Split the LLM output into lines
    lines = llm_output.split('\n')
    frequency_region_pairs = []
    for line in lines:
        if line.count(';') == 1:
            location_part, frequency_band = line.split(';')
            frequency_band = frequency_band.strip()
            locations = location_part.strip().split(',')

            for loc in locations:
                loc = loc.strip()
                region = None
                # Map the location to its highest-priority region
                for region_name, electrodes in location_groups.items():
                    if loc in electrodes:
                        region = region_name
                        break
                if region:
                    frequency_region_pairs.append((region, frequency_band))
    return frequency_region_pairs

def standardize_frequencies(frequency_region_pairs):
    standardized_pairs = []
    for region, band in frequency_region_pairs:
        if "theta" in band.lower():
            band = "Theta"
        elif "beta" in band.lower():
            band = "Beta"
        elif "alpha" or "mu" in band.lower():
            band = "Alpha"
        elif "delta" in band.lower():
            band = "Delta"
        elif "gamma" in band.lower():
            band = "Gamma"
        standardized_pairs.append((region, band))
    return standardized_pairs

def extract_location_frequency_pairs(text):
    """
    Extract Location; Frequency Band pairs from the LLM output.
    """
    # Split the output into lines and filter the relevant ones
    lines = text.split("\n")
    pairs = []
    for line in lines:
        if ";" in line:
            location, band = line.split(";")
            pairs.append((location.strip(), band.strip()))
    return pairs

# Domain based chunking
BRAIN_REGIONS = ['Frontal', 'Motor', 'Central', 'Parietal', 'Temporal', 'Occipital']
FREQUENCY_BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

def domain_based_chunking(documents):
    domain_chunks = []
    
    for document in documents:
        # Search for brain regions and frequency bands within the document
        content = document.page_content
        relevant_sentences = []

        # Split document into sentences
        sentences = nltk.sent_tokenize(content)
        
        for sentence in sentences:
            # Check if any brain region or frequency band is mentioned in the sentence
            if any(term in sentence for term in BRAIN_REGIONS) or any(band in sentence for band in FREQUENCY_BANDS):
                relevant_sentences.append(sentence)
        
        # If relevant sentences are found, create a new chunk
        if relevant_sentences:
            domain_chunks.append(Document(page_content=" ".join(relevant_sentences)))

    return domain_chunks

#-------------------------

# Recursive chunking
def process_directories(directories, llm, question, rules):
    # List to collect standardized pairs from all documents
    all_standardized_pairs = []
    # Iterate over each directory
    for dir_path in directories:
        # Get all PDF files in the directory
        pdf_files = list(Path(dir_path).glob('*.pdf'))
        for pdf_file in pdf_files:
            print(f"\nProcessing document: {pdf_file}")
            try:
                vector_store_name = pdf_file.stem
                vector_store_path = os.path.join(VECTOR_STORE_DIR, f"{vector_store_name}_faiss_index")

                # Step 1: Check if the vector store already exists
                if os.path.exists(vector_store_path):
                    print(f"Vector store exists for {pdf_file}, loading...")
                    try:
                        vectorstore = FAISS.load_local(
                            vector_store_path, embedding_model, allow_dangerous_deserialization=True
                        )
                        print(f"Vector store loaded successfully for {pdf_file}.")
                    except Exception as e:
                        logging.error(f"Failed to load FAISS vector store for {pdf_file}: {e}")
                        raise e
                else:
                    print(f"Vector store does not exist for {pdf_file}, processing...")
                    
                    # Load and process the text
                    loader = PyPDFLoader(str(pdf_file))
                    documents = loader.load()
                    
                    # Remove bibliography/references sections
                    documents = remove_bibliography_from_documents(documents)
                    
                    # Step 2: Apply recursive chunking
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=512,
                        chunk_overlap=20,
                    )
                    docs = text_splitter.split_documents(documents)

                    # Step 3: Generate embeddings and create a vector store
                    vectorstore = FAISS.from_documents(docs, embedding_model)
                    vectorstore.save_local(vector_store_path)
                    print(f"Vector store created and saved for {pdf_file}.")
                    
                # Step 4: Retrieve the most relevant chunks
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                relevant_docs = retriever.get_relevant_documents(rules)
                context = " ".join([doc.page_content for doc in relevant_docs])
                
                # Step 5: Generate a summary
                summary = generate_summary(llm, context)
                context_for_relevant_bands = context + "\n" + summary
                
                # Step 6: Generate relevant bands
                relevant_bands = generate_bands(llm, context_for_relevant_bands, question)
                
                # Step 7: Format the bands
                formatted_bands = format_bands(llm, relevant_bands)
                
                # Step 8: Extract location-frequency pairs
                pairs = extract_location_frequency_pairs(formatted_bands)
                standardized_pairs = standardize_frequencies(pairs)
                
                # Add document name to each pair
                standardized_pairs_with_doc = [(pdf_file.name, region, band) for region, band in standardized_pairs]
                all_standardized_pairs.extend(standardized_pairs_with_doc)
            
            except Exception as e:
                print(f"An error occurred while processing {pdf_file}: {e}")
                logging.error(f"Error processing {pdf_file}: {e}")
                
    return all_standardized_pairs

# Semantic chunking
def process_directories_semantic_chunking(directories, llm, question, rules):
    all_standardized_pairs = []
    
    for dir_path in directories:
        pdf_files = list(Path(dir_path).glob('*.pdf'))
        for pdf_file in pdf_files:
            print(f"\nProcessing document: {pdf_file}")
            try:
                vector_store_name = pdf_file.stem
                vector_store_path = os.path.join(VECTOR_STORE_DIR, f"{vector_store_name}_faiss_index")

                # Step 1: Check if the vector store already exists
                if os.path.exists(vector_store_path):
                    print(f"Vector store exists for {pdf_file}, loading...")
                    try:
                        vectorstore = FAISS.load_local(
                            vector_store_path, embedding_model, allow_dangerous_deserialization=True
                        )
                        print(f"Vector store loaded successfully for {pdf_file}.")
                    except Exception as e:
                        logging.error(f"Failed to load FAISS vector store for {pdf_file}: {e}")
                        raise e
                else:
                    print(f"Vector store does not exist for {pdf_file}, processing...")
                    loader = PyPDFLoader(str(pdf_file))
                    documents = loader.load()

                    # Remove bibliography/references sections
                    documents = remove_bibliography_from_documents(documents)

                    # Try Semantic Chunking with NLTK
                    try:
                        # Step 2: Apply Domain-specific Chunking (optional based on your domain knowledge)
                        domain_chunks = domain_based_chunking(documents)

                        # Step 3: Apply Semantic Chunking (with NLTK)
                        semantically_chunked_docs = semantic_chunking_nltk(domain_chunks, similarity_threshold=0.75)
                        if not semantically_chunked_docs:
                            raise ValueError("No chunks produced during NLTK chunking.")

                    except Exception as nltk_error:
                        logging.error(f"Error during NLTK semantic chunking for {pdf_file}: {nltk_error}")
                        print(f"Falling back to RecursiveCharacterTextSplitter for {pdf_file}")

                        # Fallback to RecursiveCharacterTextSplitter chunking
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=512,
                            chunk_overlap=20,
                        )
                        semantically_chunked_docs = text_splitter.split_documents(documents)
                
                    # Step 4: Generate embeddings and create a vector store
                    vectorstore = FAISS.from_documents(semantically_chunked_docs, embedding_model)
                    vectorstore.save_local(vector_store_path)
                    print(f"Vector store created and saved for {pdf_file}.")

                # Step 5: Retrieve the most relevant chunks
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                relevant_docs = retriever.get_relevant_documents(rules)
                context = " ".join([doc.page_content for doc in relevant_docs])

                # Step 6: Generate a summary
                summary = generate_summary(llm, context)
                context_for_relevant_bands = context + "\n" + summary

                # Step 7: Generate relevant bands
                relevant_bands = generate_bands(llm, context_for_relevant_bands, question)

                # Step 8: Format the bands
                formatted_bands = format_bands(llm, relevant_bands)

                # Step 9: Extract location-frequency pairs
                pairs = extract_location_frequency_pairs(formatted_bands)
                standardized_pairs = standardize_frequencies(pairs)

                # Add document name to each pair
                standardized_pairs_with_doc = [(pdf_file.name, region, band) for region, band in standardized_pairs]
                all_standardized_pairs.extend(standardized_pairs_with_doc)

            except Exception as e:
                print(f"An error occurred while processing {pdf_file}: {e}")
                logging.error(f"Error processing {pdf_file}: {e}")
    
    return all_standardized_pairs

# Standard combination: domain_based-semantic-recursive chunking
def process_directories_combined_chunking(directories, llm, question, rules):
    all_standardized_pairs = []
    
    for dir_path in directories:
        pdf_files = list(Path(dir_path).glob('*.pdf'))
        for pdf_file in pdf_files:
            print(f"\nProcessing document: {pdf_file}")
            try:
                vector_store_name = pdf_file.stem
                vector_store_path = os.path.join(VECTOR_STORE_DIR, f"{vector_store_name}_faiss_index")

                # Step 1: Check if the vector store already exists
                if os.path.exists(vector_store_path):
                    print(f"Vector store exists for {pdf_file}, loading...")
                    try:
                        vectorstore = FAISS.load_local(
                            vector_store_path, embedding_model, allow_dangerous_deserialization=True
                        )
                        print(f"Vector store loaded successfully for {pdf_file}.")
                    except Exception as e:
                        logging.error(f"Failed to load FAISS vector store for {pdf_file}: {e}")
                        raise e
                else:
                    print(f"Vector store does not exist for {pdf_file}, processing...")
                    loader = PyPDFLoader(str(pdf_file))
                    documents = loader.load()

                    # Remove bibliography/references sections
                    documents = remove_bibliography_from_documents(documents)

                    # Try Semantic Chunking with NLTK
                    try:
                        # Step 2: Apply Domain-specific Chunking (optional based on your domain knowledge)
                        domain_chunks = domain_based_chunking(documents)

                        # Step 3: Apply Semantic Chunking (with NLTK)
                        semantically_chunked_docs = semantic_chunking_nltk(domain_chunks, similarity_threshold=0.75)
                        if not semantically_chunked_docs:
                            raise ValueError("No chunks produced during NLTK chunking.")

                    except Exception as nltk_error:
                        logging.error(f"Error during NLTK semantic chunking for {pdf_file}: {nltk_error}")
                        print(f"Falling back to RecursiveCharacterTextSplitter for {pdf_file}")

                        # Fallback to recursive chunking
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=512,
                            chunk_overlap=20,
                        )
                        semantically_chunked_docs = text_splitter.split_documents(documents)
                    
                    # Step 4: Always apply recursive chunkung after semantic chunking
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=512,
                        chunk_overlap=20,
                    )
                    recursively_chunked_docs = text_splitter.split_documents(semantically_chunked_docs)
                    
                    # Step 5: Generate embeddings and create a vector store
                    vectorstore = FAISS.from_documents(recursively_chunked_docs, embedding_model)
                    vectorstore.save_local(vector_store_path)
                    print(f"Vector store created and saved for {pdf_file}.")

                # Step 6: Retrieve the most relevant chunks
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                relevant_docs = retriever.get_relevant_documents(rules)
                context = " ".join([doc.page_content for doc in relevant_docs])

                # Step 7: Generate a summary
                summary = generate_summary(llm, context)
                context_for_relevant_bands = context + "\n" + summary

                # Step 8: Generate relevant bands
                relevant_bands = generate_bands(llm, context_for_relevant_bands, question)

                # Step 9: Format the bands
                formatted_bands = format_bands(llm, relevant_bands)

                # Step 10: Extract location-frequency pairs
                pairs = extract_location_frequency_pairs(formatted_bands)
                standardized_pairs = standardize_frequencies(pairs)

                # Add document name to each pair
                standardized_pairs_with_doc = [(pdf_file.name, region, band) for region, band in standardized_pairs]
                all_standardized_pairs.extend(standardized_pairs_with_doc)

            except Exception as e:
                print(f"An error occurred while processing {pdf_file}: {e}")
                logging.error(f"Error processing {pdf_file}: {e}")
    
    return all_standardized_pairs

# Different combination: recursive-domain_based-semantic chunking
def process_directories_combined_chunking_dif(directories, llm, question, rules):
    all_standardized_pairs = []
    
    for dir_path in directories:
        pdf_files = list(Path(dir_path).glob('*.pdf'))
        for pdf_file in pdf_files:
            print(f"\nProcessing document: {pdf_file}")
            try:
                vector_store_name = pdf_file.stem
                vector_store_path = os.path.join(VECTOR_STORE_DIR, f"{vector_store_name}_faiss_index")

                # Step 1: Check if the vector store already exists
                if os.path.exists(vector_store_path):
                    print(f"Vector store exists for {pdf_file}, loading...")
                    try:
                        vectorstore = FAISS.load_local(
                            vector_store_path, embedding_model, allow_dangerous_deserialization=True
                        )
                        print(f"Vector store loaded successfully for {pdf_file}.")
                    except Exception as e:
                        logging.error(f"Failed to load FAISS vector store for {pdf_file}: {e}")
                        raise e
                else:
                    print(f"Vector store does not exist for {pdf_file}, processing...")
                    loader = PyPDFLoader(str(pdf_file))
                    documents = loader.load()

                    # Remove bibliography/references sections
                    documents = remove_bibliography_from_documents(documents)

                    # Step 2: Apply recursive chunking
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=512,
                        chunk_overlap=20,
                    )
                    recursively_chunked_docs = text_splitter.split_documents(documents)
                    
                    # Try Semantic Chunking with NLTK
                    try:
                        # Step 3: Apply Domain-specific Chunking
                        domain_chunks = domain_based_chunking(recursively_chunked_docs)

                        # Step 4: Apply Semantic Chunking (with NLTK)
                        semantically_chunked_docs = semantic_chunking_nltk(domain_chunks, similarity_threshold=0.75)
                        if not semantically_chunked_docs:
                            raise ValueError("No chunks produced during NLTK chunking.")

                    except Exception as nltk_error:
                        logging.error(f"Error during NLTK semantic chunking for {pdf_file}: {nltk_error}")
                        print(f"Falling back to RecursiveCharacterTextSplitter for {pdf_file}")

                        # Fallback to recursive chunking
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=512,
                            chunk_overlap=20,
                        )
                        semantically_chunked_docs = text_splitter.split_documents(recursively_chunked_docs)
                    
                    # Step 5: Generate embeddings and create a vector store
                    vectorstore = FAISS.from_documents(semantically_chunked_docs, embedding_model)
                    vectorstore.save_local(vector_store_path)
                    print(f"Vector store created and saved for {pdf_file}.")

                # Step 6: Retrieve the most relevant chunks
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                relevant_docs = retriever.get_relevant_documents(rules)
                context = " ".join([doc.page_content for doc in relevant_docs])

                # Step 7: Generate a summary
                summary = generate_summary(llm, context)
                context_for_relevant_bands = context + "\n" + summary

                # Step 8: Generate relevant bands
                relevant_bands = generate_bands(llm, context_for_relevant_bands, question)

                # Step 9: Format the bands
                formatted_bands = format_bands(llm, relevant_bands)

                # Step 10: Extract location-frequency pairs
                pairs = extract_location_frequency_pairs(formatted_bands)
                standardized_pairs = standardize_frequencies(pairs)

                # Add document name to each pair
                standardized_pairs_with_doc = [(pdf_file.name, region, band) for region, band in standardized_pairs]
                all_standardized_pairs.extend(standardized_pairs_with_doc)

            except Exception as e:
                print(f"An error occurred while processing {pdf_file}: {e}")
                logging.error(f"Error processing {pdf_file}: {e}")
    
    return all_standardized_pairs


# Standard combination with chain of thought prompt
def process_directories_combined_chunking_with_cot(directories, llm, question, rules):
    all_standardized_pairs = []
    
    for dir_path in directories:
        pdf_files = list(Path(dir_path).glob('*.pdf'))
        for pdf_file in pdf_files:
            print(f"\nProcessing document: {pdf_file}")
            try:
                vector_store_name = pdf_file.stem
                vector_store_path = os.path.join(VECTOR_STORE_DIR, f"{vector_store_name}_faiss_index")

                # Step 1: Check if the vector store already exists
                if os.path.exists(vector_store_path):
                    print(f"Vector store exists for {pdf_file}, loading...")
                    try:
                        vectorstore = FAISS.load_local(
                            vector_store_path, embedding_model, allow_dangerous_deserialization=True
                        )
                        print(f"Vector store loaded successfully for {pdf_file}.")
                    except Exception as e:
                        logging.error(f"Failed to load FAISS vector store for {pdf_file}: {e}")
                        raise e
                else:
                    print(f"Vector store does not exist for {pdf_file}, processing...")
                    
                    # Load and process the text
                    loader = PyPDFLoader(str(pdf_file))
                    documents = loader.load()

                    # Remove bibliography/references sections
                    documents = remove_bibliography_from_documents(documents)

                    # Try Semantic Chunking with NLTK
                    try:
                        # Step 2: Apply Domain-specific Chunking (optional based on your domain knowledge)
                        domain_chunks = domain_based_chunking(documents)

                        # Step 3: Apply Semantic Chunking (with NLTK)
                        semantically_chunked_docs = semantic_chunking_nltk(domain_chunks, similarity_threshold=0.75)
                        if not semantically_chunked_docs:
                            raise ValueError("No chunks produced during NLTK chunking.")

                    except Exception as nltk_error:
                        logging.error(f"Error during NLTK semantic chunking for {pdf_file}: {nltk_error}")
                        print(f"Falling back to RecursiveCharacterTextSplitter for {pdf_file}")

                        # Fallback to recursive chunking
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=512,
                            chunk_overlap=20,
                        )
                        semantically_chunked_docs = text_splitter.split_documents(documents)
                    
                    # Step 4: Apply recursive chunking
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=512,
                        chunk_overlap=20,
                    )
                    final_chunks = text_splitter.split_documents(semantically_chunked_docs)
                    
                    # Step 5: Generate embeddings and create a vector store
                    vectorstore = FAISS.from_documents(final_chunks, embedding_model)
                    vectorstore.save_local(vector_store_path)
                    print(f"Vector store created and saved for {pdf_file}.")

                # Step 6: Retrieve the most relevant chunks
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                relevant_docs = retriever.get_relevant_documents(rules)
                context = " ".join([doc.page_content for doc in relevant_docs])

                # Step 7: Generate a summary
                summary = generate_summary(llm, context)
                
                #Step 7: Generate relevant bands
                context_for_relevant_bands = context + "\n" + summary
                
                # Step 7: Use CoT logic for generating the bands
                relevant_bands = generate_bands_with_cot(llm, context_for_relevant_bands, question)

                # Step 8: Format the bands and standardize them
                formatted_bands = format_bands(llm, relevant_bands)
                pairs = extract_location_frequency_pairs(formatted_bands)
                standardized_pairs = standardize_frequencies(pairs)

                # Add document name to each pair
                standardized_pairs_with_doc = [(pdf_file.name, region, band) for region, band in standardized_pairs]
                all_standardized_pairs.extend(standardized_pairs_with_doc)

            except Exception as e:
                print(f"An error occurred while processing {pdf_file}: {e}")
                logging.error(f"Error processing {pdf_file}: {e}")
    
    return all_standardized_pairs

# Standard combination with knowledge-augmentation prompt
def process_directories_with_knowledge_augmented_generation(directories, llm, question, rules, brain_region_knowledge, frequency_bands):
    all_standardized_pairs = []
    
    for dir_path in directories:
        pdf_files = list(Path(dir_path).glob('*.pdf'))
        for pdf_file in pdf_files:
            print(f"\nProcessing document: {pdf_file}")
            try:
                vector_store_name = pdf_file.stem
                vector_store_path = os.path.join(VECTOR_STORE_DIR, f"{vector_store_name}_faiss_index")

                # Step 1: Check if the vector store already exists
                if os.path.exists(vector_store_path):
                    print(f"Vector store exists for {pdf_file}, loading...")
                    try:
                        vectorstore = FAISS.load_local(
                            vector_store_path, embedding_model, allow_dangerous_deserialization=True
                        )
                        print(f"Vector store loaded successfully for {pdf_file}.")
                    except Exception as e:
                        logging.error(f"Failed to load FAISS vector store for {pdf_file}: {e}")
                        raise e
                else:
                    print(f"Vector store does not exist for {pdf_file}, processing...")
                    
                    # Load and process the text
                    loader = PyPDFLoader(str(pdf_file))
                    documents = loader.load()

                    # Remove bibliography/references sections
                    documents = remove_bibliography_from_documents(documents)

                    # Try Semantic Chunking with NLTK
                    try:
                        # Step 2: Apply Domain-specific Chunking
                        domain_chunks = domain_based_chunking(documents)

                        # Step 3: Apply Semantic Chunking (with NLTK)
                        semantically_chunked_docs = semantic_chunking_nltk(domain_chunks, similarity_threshold=0.75)
                        if not semantically_chunked_docs:
                            raise ValueError("No chunks produced during NLTK chunking.")
                    
                    except Exception as nltk_error:
                        logging.error(f"Error during NLTK semantic chunking for {pdf_file}: {nltk_error}")
                        print(f"Falling back to RecursiveCharacterTextSplitter for {pdf_file}")

                        # Fallback to recursive chunking
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=512,
                            chunk_overlap=20,
                        )
                        semantically_chunked_docs = text_splitter.split_documents(documents)

                    # Step 4: Apply recursive chunking
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=512,
                        chunk_overlap=20,
                    )
                    final_chunks = text_splitter.split_documents(semantically_chunked_docs)
                    
                    # Step 5: Generate embeddings and create a vector store
                    vectorstore = FAISS.from_documents(final_chunks, embedding_model)
                    vectorstore.save_local(vector_store_path)
                    print(f"Vector store created and saved for {pdf_file}.")

                # Step 6: Retrieve the most relevant chunks
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                relevant_docs = retriever.get_relevant_documents(rules)
                context = " ".join([doc.page_content for doc in relevant_docs])

                # Step 5: Generate bands with knowledge augmentation
                relevant_bands = generate_bands_with_knowledge(llm, context, question, brain_region_knowledge, frequency_bands)

                # Step 6: Format the bands and standardize them
                formatted_bands = format_bands(llm, relevant_bands)
                pairs = extract_location_frequency_pairs(formatted_bands)
                standardized_pairs = standardize_frequencies(pairs)

                # Add document name to each pair
                standardized_pairs_with_doc = [(pdf_file.name, region, band) for region, band in standardized_pairs]
                all_standardized_pairs.extend(standardized_pairs_with_doc)

            except Exception as e:
                print(f"An error occurred while processing {pdf_file}: {e}")
                logging.error(f"Error processing {pdf_file}: {e}")
    
    return all_standardized_pairs


if __name__== '__main__':
    #device=0 if torch.cuda.is_available() else -1,
    # Setup of LLM pipeline
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        max_new_tokens=50,
        num_beams=3,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.7,
        early_stopping=True,
        do_sample=True,  # Enable sampling for diversity
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    directories = [
        "/home/oemerfar/omer_llm/pdf_documents/Motor Imagery",
        "/home/oemerfar/omer_llm/pdf_documents/Auditory Attention/",
        "/home/oemerfar/omer_llm/pdf_documents/IE Attention/"
    ]

    # Path to save/load FAISS indexes
    VECTOR_STORE_DIR = "/home/oemerfar/omer_llm/all_results/new_results/combined_chunk/results_combined_chunk_512_ovl_20_Meta-Llama-3.1-70B-AQLM-PV.csv"
    # Ensure the vector store directory exists
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    
    question = "What are the relevant frequency bands for the observed effects summarized above."
    rules_for_HA = ("Here are the regions of functional groupings:"
                    "\n- Frontal (F)" 
                    "\n- Motor (M)" 
                    "\n- Central (C)" 
                    "\n- Parietal (P)"
                    "\n- Temporal (T)"
                    "\n- Occipital (O)"
                    "\nHere are the relevant frequency bands:"
                    "\ndelta δ (0 < 4 Hz)"
                    "\ntheta θ (4−8 Hz)"
                    "\nalpha α (8−12 Hz)"
                    "\nbeta β (12−30 Hz)"
                    "\ngamma γ (> 30 Hz)")
    
    # Structured knowledge about brain regions and frequency bands
    brain_region_knowledge = {
        "Frontal": ["Fz", "F1", "F2", "F3", "F4", "F7", "F8", "Fpz", "Fp1", "Fp2", "AFz", "AF3", "AF4", "AF7", "AF8", "FT9", "FT10"],
        "Central": ["Cz", "C1", "C2", "C3", "C4", "C5", "C6", "CPz", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6"],
        "Motor": ["FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "CPz", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6"],
        "Parietal": ["Pz", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "TP7", "TP8", "TP9", "TP10", "POz", "PO3", "PO4", "PO7", "PO8"],
        "Temporal": ["T7", "T8", "FT7", "FT8", "TP7", "TP8"],
        "Occipital": ["Oz", "O1", "O2", "POz", "PO3", "PO4", "PO7", "PO8"]
    }

    frequency_bands = ["Delta (0-4 Hz)", "Theta (4-8 Hz)", "Alpha (8-12 Hz)", "Beta (12-30 Hz)", "Gamma (>30 Hz)"]
    
    # Choose process_directories function 
    all_standardized_pairs =  process_directories_combined_chunking(directories, llm, question, rules_for_HA)
    
    # Creating a DataFrame from the collected standardized pairs
    df = pd.DataFrame(all_standardized_pairs, columns=['Document', 'Region', 'Frequency Band'])
    #print("\nAll Standardized Pairs:")
    #print(df)
    
    csv_output_dir = '/home/oemerfar/omer_llm/all_results/new_results' 
    # Ensuring the directory exists; create it if it doesn't
    os.makedirs(csv_output_dir, exist_ok=True)

    output_csv = os.path.join(csv_output_dir, 'results_combined_chunk_512_ovl_20_Meta-Llama-3.1-70B-AQLM-PV.csv')
    df.to_csv(output_csv, index=False)
    #print(f"\nStandardized pairs saved to {output_csv}")
    
    
