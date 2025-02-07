import pandas as pd
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Set seeds for reproducibility
np.random.seed(42)


def evaluate_predictions(predictions_file, output_dir, classification_type, label_mapping):
    # Load predictions and true labels from CSV
    df = pd.read_csv(predictions_file)
    predictions = df['prediction'].values
    true_labels = df['true_label'].values

    # Classification report
    classification_report_str = classification_report(true_labels, predictions, target_names=list(label_mapping.keys()))
    class_report_dir = os.path.join(output_dir, 'classification_reports')
    os.makedirs(class_report_dir, exist_ok=True)
    with open(os.path.join(class_report_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report_str)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]    
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix_norm, annot=True, fmt=".2f", cmap="Greens")

    class_labels = list(label_mapping.keys()) # add textual labels to classes in the matrix
    plt.xticks(np.arange(len(class_labels)) + 0.5, class_labels, rotation=90)
    plt.yticks(np.arange(len(class_labels)) + 0.5, class_labels, rotation=0)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    confusion_matrices_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(confusion_matrices_dir, exist_ok=True)
    plt.savefig(os.path.join(confusion_matrices_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()


def main(dir_name, classification_type, predictions_file, experiment_name):

    if classification_type not in ['binary', 'multi']:
        raise ValueError("Invalid classification type. Must be either 'binary' or 'multi'.")

    # Output directory
    output_dir = f'./../../models/transformers/evaluations/{experiment_name}/ensemble/{classification_type}/{dir_name}'
    os.makedirs(output_dir, exist_ok=True)

    # Label mappings
    label_mapping_multi = {
        'Remaining': 0,
        'Non-systematic-review': 1,
        'Human-non-RCT-non-drug-intervention': 2,
        'Human-non-RCT-drug-intervention': 3,
        'Human-case-report': 4,
        'Animal-other': 5,
        'Animal-drug-intervention': 6,
        'Human-systematic-review': 7,
        'In-vitro-study': 8,
        'Human-RCT-non-drug-intervention': 9,
        'Animal-non-drug-intervention': 10,
        'Human-RCT-drug-intervention': 11,
        'Clinical-study-protocol': 12,
        'Human-RCT-non-intervention': 13
    }
    label_mapping_binary = {
        'Rest': 0,
        'Animal': 1
    }

    if classification_type == 'binary':
        label_mapping = label_mapping_binary
    else:
        label_mapping = label_mapping_multi

    # Evaluate predictions
    evaluate_predictions(predictions_file, output_dir, classification_type, label_mapping)


if __name__ == "__main__":

    experiment_name = "rerun_finetuning_19-06-24"
    classification_types = ["multi", "binary"]  # Choose 'binary' or 'multi'
    dir_names = ["confidence_based_voting_predictions", "simple_voting_predictions"]
    for classification_type in classification_types:
        for dir_name in dir_names:
            predictions_file = f"./../../models/transformers/evaluations/{experiment_name}/ensemble/{classification_type}/{dir_name}/predictions.csv"
    
            main(dir_name=dir_name, classification_type=classification_type, predictions_file=predictions_file, experiment_name=experiment_name)
