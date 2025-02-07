import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def load_model(model_path, model_name, num_labels):
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Initialize model based on the model_name and number of labels
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.load_state_dict(state_dict)

    return model


def get_short_model_name(model_name):
    model_names = {
        'bert-base-uncased': 'bert-base',
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext': 'PubMedBERT',
        'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract': 'BiomedBERT',
        'allenai/scibert_scivocab_uncased': 'Scibert',
        'dmis-lab/biobert-v1.1': 'biobert',
        'michiyasunaga/BioLinkBERT-base': 'BioLinkBERT',
        'emilyalsentzer/Bio_ClinicalBERT': 'Bio_ClinicalBERT'
    }
    if model_name in model_names:
        return model_names[model_name]
    else:
        raise ValueError("Invalid model name. Cannot be mapped to short name.")


def load_test_data(data_dir, model_name, classification_type):
    test_file = os.path.join(data_dir, f'test_{classification_type}.csv')
    test_df = pd.read_csv(test_file)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def concatenate_text(row):
        text_parts = [str(row['journal_name']), str(row['title']), str(row['abstract'])]
        keywords = row['keywords']
        if pd.notna(keywords):
            keywords_list = keywords.split('|')
            text_parts.extend(keywords_list)
        return ' '.join(text_parts)

    test_df['text'] = test_df.apply(concatenate_text, axis=1)
    test_encodings = tokenizer(test_df['text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors='pt')

    if classification_type == 'animal':
        test_labels = torch.tensor(test_df['animal_label'].values)
    elif classification_type == 'other':
        test_labels = torch.tensor(test_df['other_label'].values)
    else:
        raise ValueError("Invalid classification type. Must be either 'animal' or 'other'.")

    return test_encodings, test_labels


def evaluate_model(model, test_dataloader, output_dir, model_name, logger, classification_type, label_mapping):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []

    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch[2].cpu().numpy())

    # extract short model name
    short_model_name = get_short_model_name(model_name)

    # predictions directory
    predictions_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    with open(os.path.join(predictions_dir, f'{short_model_name}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prediction', 'true_label', 'confidence'])
        writer.writerows(zip(predictions, true_labels, [max(prob) for prob in probabilities]))

    # classification report
    classification_report_str = classification_report(true_labels, predictions, target_names=list(label_mapping.keys()))
    logger.info(f"Model: {model_name}\n{classification_report_str}")
    class_report_dir = os.path.join(output_dir, 'classification_reports')
    os.makedirs(class_report_dir, exist_ok=True)
    with open(os.path.join(class_report_dir, f'{short_model_name}.txt'), 'w') as f:
        f.write(classification_report_str)
    
    # confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]    
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix_norm, annot=True, fmt=".2f", cmap="Greens")

    class_labels = list(label_mapping.keys()) # add textual labels to classes in the matrix
    plt.xticks(np.arange(len(class_labels)) + 0.5, class_labels, rotation=90)
    plt.yticks(np.arange(len(class_labels)) + 0.5, class_labels, rotation=0)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {short_model_name}')
    confusion_matrices_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(confusion_matrices_dir, exist_ok=True)
    plt.savefig(os.path.join(confusion_matrices_dir, f'{short_model_name}.png'), bbox_inches='tight')
    plt.close()


def main(classification_type, experiment_name):

    if classification_type not in ['animal', 'other']:
        raise ValueError("Invalid classification type. Must be either 'animal' or 'other'.")

    # data
    data_dir = "./../../data/data_splits_stratified/6-2-2_all_classes_enriched_with_kw_hierarchical"
    output_dir = f'./../../models/transformers/evaluations/{experiment_name}/{classification_type}'
    os.makedirs(output_dir, exist_ok=True)
    # logging
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(logs_dir, f"evaluation.log"), filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    # checkpoints
    checkpoint_dir = f"./../../models/transformers/checkpoints/{experiment_name}/{classification_type}/models"
    # models
    models_to_evaluate = [
        'allenai/scibert_scivocab_uncased',
        'dmis-lab/biobert-v1.1',
    ]

    # select appropriate labels
    label_mapping_animal = {
        'Animal-other': 0,
        'Animal-drug-intervention': 1,
        'Animal-non-drug-intervention': 2
    }
    label_mapping_other = {
        'Remaining': 0,
        'Non-systematic-review': 1,
        'Human-non-RCT-non-drug-intervention': 2,
        'Human-non-RCT-drug-intervention': 3,
        'Human-case-report': 4,
        'Human-systematic-review': 5,
        'In-vitro-study': 6,
        'Human-RCT-non-drug-intervention': 7,
        'Human-RCT-drug-intervention': 8,
        'Clinical-study-protocol': 9,
        'Human-RCT-non-intervention': 10
    }

    if classification_type == 'animal':
        label_mapping = label_mapping_animal
        num_labels = len(label_mapping_animal)
    else:
        label_mapping = label_mapping_other
        num_labels = len(label_mapping_other)

    # iterate over select models and perform evaluation
    for model_name in models_to_evaluate:
        # log evaluation
        logger.info(f"Evaluating {get_short_model_name(model_name)}")
        print(f"Evaluating {get_short_model_name(model_name)}")
        # define path to models
        model_path = os.path.join(checkpoint_dir, get_short_model_name(model_name), f"ft_{get_short_model_name(model_name)}_{classification_type}.pt")
        
        model = load_model(model_path, model_name, num_labels)
        
        # load test data
        test_encodings, test_labels = load_test_data(data_dir, model_name, classification_type)
        test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # evaluate model
        evaluate_model(model, test_dataloader, output_dir, model_name, logger, classification_type, label_mapping)

    print('Evaluations complete')

# Execute the main function for both 'animal' and 'other' classification types.
if __name__ == '__main__':
    main('animal', 'hierarchical_finetuning_27-06-24')
    main('other', 'hierarchical_finetuning_27-06-24')
