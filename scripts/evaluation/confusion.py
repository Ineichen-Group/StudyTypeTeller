import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_predictions(file_path):

    return pd.read_csv(file_path)


def create_confusion_matrix(predictions, true_labels):

    return confusion_matrix(true_labels, predictions)


def plot_confusion_matrix(conf_matrix, class_names, model_name, output_dir):

    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 5))
    sns.heatmap(conf_matrix_norm, annot=True, fmt=".2f", cmap="Greens")

    plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=90, fontsize=10)
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0, fontsize=10)

    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('True', fontsize=16)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion.png'), bbox_inches='tight')
    plt.show()
    plt.close()


def generate_confusion_matrix_from_log(file_path, label_mapping, model_name, output_dir, col_true, col_prediction):
    # Load the predictions
    data = load_predictions(file_path)
    # Extract predictions and true labels
    predictions = data[col_true]
    true_labels = data[col_prediction]
    
    # Create confusion matrix
    conf_matrix = create_confusion_matrix(predictions, true_labels)
    
    # Get class names from the label mapping
    class_names = list(label_mapping.keys())
    
    # Plot and save the confusion matrix
    plot_confusion_matrix(conf_matrix, class_names, model_name, output_dir)


def main():

    label_mapping = {
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

    models = ['GPT', 'BERT']
    for model in models:

        output_dir = './'
        if model == 'BERT':
            model_name = 'SciBERT'
            file_path = './../../models/transformers/evaluations/multi/predictions/Scibert.csv'
            col_prediction = 'true_label'
            col_true = 'prediction'
        elif model == 'GPT':
            model_name = 'GPT-4' # TODO modify name
            file_path = './gpt-4-turbo-preview_enriched_kw_test_outputs_P6_P7_P11_3_P11_4_structured.csv' # TODO modify name
            col_prediction = 'accepted_label_numerical'
            col_true = 'gpt_predictions_P6_numerical'




            generate_confusion_matrix_from_log(
                                                file_path, 
                                                label_mapping, 
                                                model_name, 
                                                output_dir, 
                                                col_true, 
                                                col_prediction
                                                )


if __name__ == "__main__":
    main()
