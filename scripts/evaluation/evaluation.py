import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
import json
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def load_model(model_path):
    model = torch.load(model_path)
    return model


def load_test_data(data_dir, model_name, classification_type):
    test_file = os.path.join(data_dir, 'test.csv')
    test_df = pd.read_csv(test_file)

    # map models and auto-tokenizers 
    # TODO more elegant way to do this?
    tokenizer_name = None
    if 'bert-base-uncased' in model_name:
        tokenizer_name = 'bert-base-uncased'
    elif 'microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext' in model_name:
        tokenizer_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    elif 'microsoft_BiomedNLP-BiomedBERT-base-uncased-abstract' in model_name:
        tokenizer_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
    elif 'allenai_scibert_scivocab_uncased' in model_name:
        tokenizer_name = 'allenai/scibert_scivocab_uncased'
    elif 'dmis-lab_biobert-v1.1' in model_name:
        tokenizer_name = 'dmis-lab/biobert-v1.1'
    if tokenizer_name is None:
        raise ValueError(f"Tokenizer not found for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Concatenate 'journal_name', 'title', and 'abstract' columns to create 'text' column
    test_df['text'] = test_df['journal_name'] + ' ' + test_df['title'] + ' ' + test_df['abstract']
    # encode textual data
    test_encodings = tokenizer(test_df['text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors='pt')

    # designate label column dependent on the classification type 
    if classification_type == 'binary':
        test_labels = torch.tensor(test_df['binary_label'].values)
    elif classification_type == 'multi':
        test_labels = torch.tensor(test_df['multi_label'].values)
    else:
        raise ValueError("Invalid classification type. Must be either 'binary' or 'multi'.")

    return test_encodings, test_labels


def evaluate_model(model, test_dataloader, output_dir, model_name, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # collect data for for plotting
    predictions = []
    true_labels = []

    # set to evaluation mode
    with torch.no_grad():
        for batch in test_dataloader: # TODO larger batches?
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch[2].cpu().numpy())

    # Save predictions and true labels with the model name as CSV
    with open(os.path.join(output_dir, f'{model_name}_predictions.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prediction', 'true_label'])
        writer.writerows(zip(predictions, true_labels))

    classification_report_str = classification_report(true_labels, predictions)
    logger.info(f"Model: {model_name}\n{classification_report_str}")

    # create subdir for classification reports if does not exist
    class_report_dir = os.path.join(output_dir, 'classification_reports')
    os.makedirs(class_report_dir, exist_ok=True)
    # Save classification report with the model name
    with open(os.path.join(output_dir, f'{model_name}_classification_report.txt'), 'w') as f:
        f.write(classification_report_str)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] # normalizze

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_norm, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')

    # Create directory for confusion matrices if it doesn't exist
    confusion_matrices_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(confusion_matrices_dir, exist_ok=True)

    # Save confusion matrix as PNG
    plt.savefig(os.path.join(confusion_matrices_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()


def main(classification_type):
    if classification_type not in ['binary', 'multi']:
        raise ValueError("Invalid classification type. Must be either 'binary' or 'multi'.")

    checkpoint_dir = f"./../../models/transformers/checkpoints/{classification_type}/models"
    data_dir = "./../../data/data_splits_stratified/6-2-2_all_classes"

    logging.basicConfig(filename=f"{classification_type}_evaluation.log", filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    for model_name in os.listdir(checkpoint_dir):
        if "_FULL.pt" not in model_name:
            continue  # Skip if not a full model but rather just state dict
        model_path = os.path.join(checkpoint_dir, model_name)
        logger.info(f"Loading model from {model_path}")
        model = load_model(model_path)

        test_encodings, test_labels = load_test_data(data_dir, model_name, classification_type)

        test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)
        # Shuffle the test set
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

        output_dir = f'./../../models/transformers/evaluations/outputs/{classification_type}'
        os.makedirs(output_dir, exist_ok=True)

        evaluate_model(model, test_dataloader, output_dir, model_name, logger)



if __name__ == "__main__":
    main(classification_type='binary')  # TODO specify classification type
