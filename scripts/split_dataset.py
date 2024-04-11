import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict



def map_labels(df):

    label_map = {
        'REST': [
            'Study-protocol',
            'Human-systematic-review',
            'Non-systematic-review',
            'Human-RCT-non-drug-intervention',
            'Human-RCT-drug-intervention',
            'Human-RCT-non-intervention',
            'Human-case-report',
            'Human-non-RCT-non-drug-intervention',
            'Human-non-RCT-drug-intervention',
            'In-vitro-study',
            'Remaining'
        ],
        'ANIMAL': [
            'Animal-systematic-review',
            'Animal-non-drug-intervention',
            'Animal-drug-intervention',
            'Animal-other'
        ]
    }

    label_mapping_binary = {
        'REST': 0,
        'ANIMAL': 1
    }
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
        'Human-RCT-non-intervention': 12
    }

    def map_to_label(col, label_map):
        """ 
        Function to map flattened_labels to the corresponding label 
        """
        for label, sublabels in label_map.items():
            if col in sublabels:
                return label
        return None
        
    df['BINARY_CLASS'] = df['accepted_label'].apply(lambda x: map_to_label(x, label_map))
    df.dropna(subset=['BINARY_CLASS'], inplace=True)  # Make sure there are no empty datapoints
    df['NUM_LABEL_BINARY'] = df['BINARY_CLASS'].map(label_mapping_binary)  # Encode binary labels
    df['NUM_LABEL_MULTI'] = df['accepted_label'].map(label_mapping_multi)  # Encode multiclass labels
    return df

def concatenate_textual_data(df):
    df['text'] = df['journal_name'] + " " + df['title'] + " " + df['abstract']
    return df

def preprocess_textual_data(df):
    regex_pattern = r'[^a-zA-Z0-9\s]'
    df['text'] = df['text'].apply(lambda x: re.sub(regex_pattern, '', x.lower())).str.lower()
    return df

def get_data(df, col_name_binary, col_name_multi):
    text = df.text.values
    labels_binary = df[col_name_binary].values
    labels_multi = df[col_name_multi].values
    
    return text, labels_binary, labels_multi
    
def save_splits(train_data, val_data, test_data, save_dir):
    os.makedirs(save_dir, exist_ok=True) # create folder if does not exist
    train_data.to_csv(os.path.join(save_dir, 'train_data.csv'), index=False)
    val_data.to_csv(os.path.join(save_dir, 'val_data.csv'), index=False)
    test_data.to_csv(os.path.join(save_dir, 'test_data.csv'), index=False)
    print("Data splits have been saved successfully.")

def custom_train_test_split(text, labels_binary, labels_multi, test_size, val_size, random_state):
    # Combine text and labels for easier manipulation
    data = list(zip(text, labels_binary, labels_multi))
    
    # Separate data by class
    class_data = defaultdict(list)
    for item in data:
        class_data[item[2]].append(item)
    
    # Initialize splits
    train_data = []
    val_data = []
    test_data = []
    
    # Ensure at least one sample from each class in each split
    for class_id, samples in class_data.items():
        # Shuffle samples to ensure randomness
        np.random.shuffle(samples)
        
        # Calculate the number of samples for each split
        n_samples = len(samples)
        n_test = max(1, int(n_samples * test_size))  # Ensure at least one sample in test set
        n_val = max(1, int(n_samples * val_size))  # Ensure at least one sample in validation set
        
        # Assign samples to splits
        train_data.extend(samples[:-n_val-n_test])
        val_data.extend(samples[-n_val-n_test:-n_test])
        test_data.extend(samples[-n_test:])
    
    # Shuffle the splits
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)
    
    # Unpack the splits
    text_train, labels_binary_train, labels_multi_train = zip(*train_data)
    text_val, labels_binary_val, labels_multi_val = zip(*val_data)
    text_test, labels_binary_test, labels_multi_test = zip(*test_data)
    
    return text_train, text_val, text_test, labels_binary_train, labels_binary_val, labels_binary_test, labels_multi_train, labels_multi_val, labels_multi_test



def main():

    # Load raw data
    path_to_file = "./../data/prodigy/annotated_output/final/full_combined_dataset_1996.csv"
    df = pd.read_csv(path_to_file)

    col_name_binary = 'NUM_LABEL_BINARY'
    col_name_multi = 'NUM_LABEL_MULTI'

    df = map_labels(df=df)
    df = concatenate_textual_data(df=df)
    df = preprocess_textual_data(df=df)

    text, labels_binary, labels_multi = get_data(df, col_name_binary, col_name_multi)

    # Define split sizes and random state
    test_size = 0.1
    val_size = 0.1
    random_state = 42

    # Perform custom train-test split
    text_train, text_val, text_test, labels_binary_train, labels_binary_val, labels_binary_test, labels_multi_train, labels_multi_val, labels_multi_test = custom_train_test_split(
        text, labels_binary, labels_multi, test_size, val_size, random_state)

    # Convert the splits into DataFrames
    train_data = pd.DataFrame({'text': text_train, col_name_binary: labels_binary_train, col_name_multi: labels_multi_train})
    val_data = pd.DataFrame({'text': text_val, col_name_binary: labels_binary_val, col_name_multi: labels_multi_val})
    test_data = pd.DataFrame({'text': text_test, col_name_binary: labels_binary_test, col_name_multi: labels_multi_test})

    # Sanity check: observe distribution of classes across splits
    print(f'Train binary:\n{train_data[col_name_binary].value_counts()}')
    print(f'Train multi:\n{train_data[col_name_multi].value_counts()}')
    print(f'Val binary:\n{val_data[col_name_binary].value_counts()}')
    print(f'Val multi:\n{val_data[col_name_multi].value_counts()}')
    print(f'Test binary:\n{test_data[col_name_binary].value_counts()}')
    print(f'Test multi:\n{test_data[col_name_multi].value_counts()}')

    # sanity check: make sure all classes are represented across the splits
    train_binary = train_data[col_name_binary].nunique()
    val_binary = val_data[col_name_binary].nunique()
    test_binary = test_data[col_name_binary].nunique()
    train_multi = train_data[col_name_multi].nunique()
    val_multi = val_data[col_name_multi].nunique()
    test_multi = test_data[col_name_multi].nunique()

    if train_binary == val_binary == test_binary == 2:
        print('Both BINARY classes are present in all splits')
    else:
        print('Attention: not all BINARY classes are present in all splits')

    if train_multi == val_multi == test_multi == 13:
        print('All MULTI classes are present in all splits')
    else:
        print('Attention: not all MULTI classes are present in all splits')


    # Save splits for use with all models
    save_dir = "./../data/data_splits_stratified"
    save_splits(train_data, val_data, test_data, save_dir)

if __name__ == "__main__":
    main()

