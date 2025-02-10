import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from embedding import generate_embeddings
import torch
from tqdm import tqdm
import time


if __name__ == '__main__':

    df = pd.read_csv('../data/data_splits_stratified/6-2-2_all_classes/train.csv')

    df['input_journal_title_abstract'] = '<journal>' + df['journal_name'] + '</journal>' + \
                                         '<title>' + df['title'] + '</title>' + \
                                         '<abstract>' + df['abstract'] + '</abstract>'

    column_to_embed = "input_journal_title_abstract"
    output_file_suffix = "train_ds"

    ### BERT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("running on device: {}".format(device))
    abstracts = df[column_to_embed].tolist()
    # specifying model
    checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    # "dmis-lab/biobert-v1.1"
    # "dmis-lab/biobert-v1.1"
    # "allenai/scibert_scivocab_uncased"
    # "bert-base-uncased"
    # "dmis-lab/biobert-v1.1"
    # "allenai/scibert_scivocab_uncased"
    # "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    # "dmis-lab/biobert-base-cased-v1.2"
    # "bert-base-uncased"
    # "allenai/scibert_scivocab_uncased"
    # "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint)
    mat = np.empty([len(abstracts), 768])
    abstract_batch = abstracts
    # Start timing
    start_time = time.time()

    # Process abstracts in batches and track progress with tqdm
    for i, abst in enumerate(tqdm(abstract_batch, desc="Generating Embeddings")):
        _, mat[i], _ = generate_embeddings(abst, tokenizer, model, device)
        last_iter = np.array([i])
        np.save(f'../data/variables/last_iter_batch_1_{output_file_suffix}', last_iter)

    model_name = checkpoint.replace("/","_")

    # save embedding
    np.save(f'../data/embeddings/embeddings_{model_name}_{output_file_suffix}', mat)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time for generating embeddings: {elapsed_time:.2f} seconds")



