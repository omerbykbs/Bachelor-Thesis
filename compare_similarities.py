from sentence_transformers import SentenceTransformer, util
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
from PyPDF2 import PdfReader
from pathlib import Path
import os

nlp = spacy.load('en_core_web_sm')

MODEL_IDS = ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-mpnet-base-v2", "all-distilroberta-v1"]

DIRECTORIES = ['/Users/omerfaruk/Bachelorarbeit/LLM_omer/pdf_documents/MI',
               '/Users/omerfaruk/Bachelorarbeit/LLM_omer/pdf_documents/AA',
               '/Users/omerfaruk/Bachelorarbeit/LLM_omer/pdf_documents/IEA']

SIMILARITY_FUNCTIONS = {
    "cosine": util.pytorch_cos_sim,
    "dot": lambda x, y: np.dot(x, y.T),
    # For Euclidean distance
    "euclidean": lambda x, y: np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2),
    # For Manhattan distance
    "manhattan": lambda x, y: np.sum(np.abs(x[:, None, :] - y[None, :, :]), axis=2)
}

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def convert_pdf_to_text(pdf_file):
    with open(pdf_file, 'rb') as file:
        pdf_reader = PdfReader(file)

        text = ""
        num_pages = len(pdf_reader.pages)

        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    return text

def get_pdf_files_from_directory(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.pdf')]

def calculate_similarity_matrix(model, topics_documents, similarity_function):
    topics = list(topics_documents.keys())
    documents = [doc for docs in topics_documents.values() for doc in docs]
    embeddings = model.encode(documents)
    
    similarity_matrix = np.zeros((len(topics), len(topics)))

    for i, topic1 in enumerate(topics):
        for j, topic2 in enumerate(topics):
            embeddings1 = embeddings[i*len(topics_documents[topic1]):(i+1)*len(topics_documents[topic1])]
            embeddings2 = embeddings[j*len(topics_documents[topic2]):(j+1)*len(topics_documents[topic2])]

            similarity = similarity_function(embeddings1, embeddings2)
            similarity_matrix[i, j] = similarity.mean().item()

    return similarity_matrix

def rank_matrix(matrix):
    score = 0
    # Criteria 1: Highest diagonal cells
    diagonal_cells = [matrix[0, 0], matrix[1, 1], matrix[2, 2]]
    max_diagonal = all(diag >= max(matrix[i, j] for i in range(3) for j in range(3) if i != j) for diag in diagonal_cells)
    if max_diagonal:
        score += 1 
    
    # Criteria 2: Off-diagonal cells should not exceed diagonal cells
    off_diagonal_cells = [matrix[0, 1], matrix[0, 2], matrix[1, 0], matrix[1, 2], matrix[2, 0], matrix[2, 1]]
    if all(off <= min(diagonal_cells) for off in off_diagonal_cells):
        score += 1  

    # Criteria 3: [1,2] and [2,1] should have second-highest values after diagonals
    sorted_values = sorted(matrix.ravel(), reverse=True)
    if matrix[1, 2] in sorted_values[3:5] and matrix[2, 1] in sorted_values[3:5]:
        score += 1  
    
    # Criteria 4: [2,0] and [0,2] should have third-highest values after [1,2] and [2,1]
    if matrix[2, 0] in sorted_values[5:7] and matrix[0, 2] in sorted_values[5:7]:
        score += 1 

    return score

def __main__():
    
    mi = get_pdf_files_from_directory(DIRECTORIES[0])
    aa = get_pdf_files_from_directory(DIRECTORIES[1])
    iea = get_pdf_files_from_directory(DIRECTORIES[2])

    topics_documents = {
        'topic1': [preprocess_text(convert_pdf_to_text(pdf)) for pdf in mi],
        'topic2': [preprocess_text(convert_pdf_to_text(pdf)) for pdf in aa],
        'topic3': [preprocess_text(convert_pdf_to_text(pdf)) for pdf in iea],
    }

    # Similarity matrices for all models and similarity functions
    all_similarity_matrices = {}  # Collect matrices for all models and similarity functions

    for similarity_function_name, similarity_function in SIMILARITY_FUNCTIONS.items():

        for model_id in MODEL_IDS:
            model = SentenceTransformer(model_id)
            # Calculating similarity matrix for each model with each similarity function
            similarity_matrix = calculate_similarity_matrix(model, topics_documents, similarity_function)
            # Unique key for each model-function combination
            all_similarity_matrices[f"{model_id}_{similarity_function_name}"] = similarity_matrix

            # Plot the similarity matrix
            subjects = ["Motor Imagery", "Auditory Attention", "Internal/External Attention"]
            plt.figure(figsize=(10, 7))
            sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=subjects, yticklabels=subjects)
            plt.xlabel('Topics')
            plt.ylabel('Topics')
            plt.title(f'Average Similarities with {similarity_function_name.title()} Between Topics for Model {model_id}')
            plt.show()

    # Apply the ranking function to each matrix
    rankings = {name: rank_matrix(matrix) for name, matrix in all_similarity_matrices.items()}
    sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    model_names, scores = zip(*sorted_rankings)
    
    # Plotting the ranking of models
    plt.figure(figsize=(12, 10))
    plt.barh(model_names, scores, color='skyblue')
    plt.xlabel('Ranking Score', fontsize=16)
    plt.title('Model Rankings Based on Similarity Matrix Criteria', fontsize=18)
    plt.gca().invert_yaxis() 
    plt.xticks(range(int(min(scores)), int(max(scores)) + 1), fontsize=14)
    plt.yticks(fontsize=14)  
    plt.show()
    
    all_MiniLM_L6_v2_cosine = np.array([
        [0.63075656, 0.47597444, 0.50073165],
        [0.47597444, 0.60193682, 0.52398843],
        [0.50073165, 0.52398849, 0.59429568]
    ])
    all_MiniLM_L12_v2_cosine = np.array([
        [0.60070735, 0.43354461, 0.4599148],
        [0.43354461, 0.56320304, 0.48002988],
        [0.4599148, 0.48002991, 0.56740552]
    ])

    topics = ["MI", "AA", "IEA"]

    # Plot the two matrices side by side for comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot for all-MiniLM-L6-v2 cosine similarity matrix
    sns.heatmap(all_MiniLM_L6_v2_cosine, annot=True, cmap='coolwarm', ax=axes[0], cbar=True, square=True)
    axes[0].set_title('Average Similarities with Cosine - all-MiniLM-L6-v2')
    axes[0].set_xticklabels(topics)
    axes[0].set_yticklabels(topics)
    
    # Plot for all-MiniLM-L12-v2 cosine similarity matrix
    sns.heatmap(all_MiniLM_L12_v2_cosine, annot=True, cmap='coolwarm', ax=axes[1], cbar=True, square=True)
    axes[1].set_title('Average Similarities with Cosine - all-MiniLM-L12-v2')
    axes[1].set_xticklabels(topics)
    axes[1].set_yticklabels(topics)

    # Display the plot
    plt.suptitle('Comparison of Average Similarity Matrices with Cosine for all-MiniLM-L6-v2 and all-MiniLM-L12-v2')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    
    __main__()