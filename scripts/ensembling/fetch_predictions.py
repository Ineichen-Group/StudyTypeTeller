import os
import pandas as pd
import csv

def aggregate_predictions_probabilities(input_dir, output_dir, classification_type, experiment_name):

    models_to_evaluate = [
        'bert-base',
        'PubMedBERT',
        'BiomedBERT',
        'Scibert',
        'biobert',
        'BioLinkBERT',
        'Bio_ClinicalBERT'
    ]

    all_model_data = {}  # Dictionary to hold predictions, probabilities, and true labels for each model

    for model_name in models_to_evaluate:
        # Read predictions and probabilities from each model's CSV file
        predictions_file = os.path.join(input_dir, 'predictions', f'{model_name}.csv')

        model_data = []
        with open(predictions_file, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header
            for idx, row in enumerate(reader):
                prediction = int(row[0])  # convert to int
                true_label = int(row[1])  # convert to int
                confidence = float(row[2])  # convert to float
                model_data.append((idx, true_label, prediction, confidence))

        # Store predictions, probabilities, and true labels for this model in the dictionary
        all_model_data[model_name] = model_data

    # Create a DataFrame to store aggregated results
    df_predictions = pd.DataFrame()

    # Add true_label column (assuming true labels are the same for all models, based on the first model's data)
    # df_predictions['true_label'] = all_model_data[models_to_evaluate[0]][0][1]  # Assuming all models have the same true_label in the first row

    # Add predictions and probabilities columns for each model
    for model_name, data in all_model_data.items():
        original_indices, true_labels, predictions, probabilities = zip(*data)
        df_predictions[f'true_label'] = true_labels
        df_predictions[f'{model_name}_prediction'] = predictions
        df_predictions[f'{model_name}_probability'] = probabilities

    # Save aggregated results to CSV
    aggregated_csv_file = os.path.join(output_dir, f'aggregated_predictions.csv')
    df_predictions.to_csv(aggregated_csv_file, index=False)

    print(f"Aggregated predictions and probabilities saved to {aggregated_csv_file}")


if __name__ == "__main__":

    classification_type = "binary"  # Choose 'binary' or 'multi'
    experiment_name = "rerun_finetuning_19-06-24"
    input_dir = f'./../../models/transformers/evaluations/{experiment_name}/{classification_type}'
    output_dir = f'./../../models/transformers/evaluations/{experiment_name}/ensemble/aggregated_output/{classification_type}'
    os.makedirs(output_dir, exist_ok=True)

    aggregate_predictions_probabilities(input_dir, output_dir, classification_type, experiment_name)


