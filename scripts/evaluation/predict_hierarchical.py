import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, BertForSequenceClassification
import pandas as pd
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_model(model_path, model_name, num_labels):
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Initialize model based on the model_name and number of labels
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.load_state_dict(state_dict)

    return model

def load_test_data(data_dir, model_name):
    test_file = os.path.join(data_dir, 'test.csv')
    test_df = pd.read_csv(test_file)
    # test_df = test_df[:5]  # For testing, remove this line in actual usage

    # Initialize tokenizer based on the model_name
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

    return test_encodings, test_df['binary_label'].values.tolist(), test_df['multi_label'].values.tolist(), test_df[['journal_name', 'title', 'abstract', 'keywords']]

def predict(model, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask = batch  # Assuming batch is a tuple/list of tensors
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    return predictions


def remap_hierarchical_to_multi(hierarchical_pred, classification_type):
    if classification_type == 'animal':
        animal_map = {0: 5, 1: 6, 2: 10}
        return animal_map[hierarchical_pred]
    elif classification_type == 'other':
        other_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 7, 6: 8, 7: 9, 8: 11, 9: 12, 10: 13}
        return other_map[hierarchical_pred]
    else:
        raise ValueError("Unknown classification type.")


def generate_classification_report(predictions_df):
    y_true = predictions_df['true_label_multi']
    y_pred = predictions_df['hierarchical_prediction_remapped']
    
    report = classification_report(y_true, y_pred, digits=4)
    return report

def plot_confusion_matrix(y_true, y_pred, labels, title, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Greens', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(output_path)
    plt.close()


def main(experiment_name, model_name, model_name_full):

    # Specify the paths to your local model checkpoints
    binary_model_path = f'./../../models/transformers/checkpoints/rerun_finetuning_19-06-24/binary/models/{model_name}/ft_{model_name}_binary.pt'
    animal_model_path = f"./../../models/transformers/checkpoints/hierarchical_finetuning_27-06-24/animal/models/{model_name}/ft_{model_name}_animal.pt"
    other_model_path = f'./../../models/transformers/checkpoints/hierarchical_finetuning_27-06-24/other/models/{model_name}/ft_{model_name}_other.pt'

    binary_model = load_model(binary_model_path, model_name_full, num_labels=2)
    animal_model = load_model(animal_model_path, model_name_full, num_labels=3)
    other_model = load_model(other_model_path, model_name_full, num_labels=11)

    # Prepare test data
    data_dir = "./../../data/data_splits_stratified/6-2-2_all_classes_enriched_with_kw"
    test_encodings, true_labels_binary, true_labels_multi, test_text_info = load_test_data(data_dir, model_name_full)
    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'])
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Predict binary labels using binary model
    binary_predictions = predict(binary_model, test_dataloader)

    # Initialize storage for final predictions
    final_predictions = []

    # Iterate through binary predictions to predict hierarchical labels
    batch_size = test_dataloader.batch_size
    num_batches = len(test_dataloader)
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(test_encodings['input_ids']))
        
        batch_encodings = {
            'input_ids': test_encodings['input_ids'][start_idx:end_idx],
            'attention_mask': test_encodings['attention_mask'][start_idx:end_idx]
        }
        
        batch_binary_pred = binary_predictions[start_idx:end_idx]
        
        for idx, binary_pred in enumerate(batch_binary_pred):
            if binary_pred == 0:
                hierarchical_model = other_model
                classification_type = 'other'
            elif binary_pred == 1:
                hierarchical_model = animal_model
                classification_type = 'animal'
            else:
                raise ValueError("Binary prediction should be 0 or 1.")

            # Predict hierarchical labels using the corresponding model
            hierarchical_predictions = predict(hierarchical_model, DataLoader(TensorDataset(batch_encodings['input_ids'][idx:idx+1], batch_encodings['attention_mask'][idx:idx+1]), batch_size=1))
            remapped_pred = remap_hierarchical_to_multi(hierarchical_predictions[0], classification_type)

            # Store final prediction for this item
            final_predictions.append({
                'binary_prediction': binary_pred,
                'true_label_binary': true_labels_binary[start_idx + idx],
                'hierarchical_prediction': hierarchical_predictions[0],
                'hierarchical_prediction_remapped': remapped_pred,
                'true_label_multi': true_labels_multi[start_idx + idx],
            })

    # Save final predictions to CSV
    output_dir = f'./../../models/transformers/evaluations/{experiment_name}/'
    os.makedirs(output_dir, exist_ok=True)
    final_predictions_df = pd.DataFrame(final_predictions)
    final_predictions_df.to_csv(os.path.join(output_dir, f'final_predictions_{model_name}.csv'), index=False)

    # Generate classification report
    report = generate_classification_report(final_predictions_df)
    report_output_path = os.path.join(output_dir, f'classification_report_{model_name}.txt')
    with open(report_output_path, 'w') as f:
        f.write(report)
        print(report)

    plot_confusion_matrix(
        final_predictions_df['true_label_multi'], final_predictions_df['hierarchical_prediction_remapped'],
        labels=sorted(final_predictions_df['true_label_multi'].unique()), title='Confusion Matrix for Hierarchical Classification',
        output_path=os.path.join(output_dir, f'confusion_matrix_hierarchical_{model_name}.pdf')
    )



if __name__ == "__main__":
    experiment_name = "hierarchical_finetuning_27-06-24"
    main(experiment_name, "biobert", 'dmis-lab/biobert-v1.1')
    main(experiment_name, "Scibert", 'allenai/scibert_scivocab_uncased')
