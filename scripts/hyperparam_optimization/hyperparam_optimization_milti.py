import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import time
import wandb
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter


# Constants
PROJECT_NAME = 'SciBERT_ep6_multi' 
TOKENIZER_NAME = "allenai/scibert_scivocab_uncased"
MODEL_NAME = 'allenai/scibert_scivocab_uncased'
DATA_DIR = './../data/data_splits_stratified'
FILE_PATH = './../data/data_splits_stratified/train_data.csv'

CLASSIFICATION_TYPE = "NUM_LABEL_BINARY"
EPOCHS = 6
MAX_LENGTH = 256
SWEEP_CONFIG = {
    'method': 'bayes',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'min': 1e-5, 'max': 1e-3},
        'batch_size': {'values': [8, 16, 32]},
        "weight_decay": {"min": 1e-5, "max": 1e-3}
    }
}

def load_data(file_path, col, max_length=256, test_size=0.3):
    df = pd.read_csv(file_path)
    texts = df['text'].values
    labels = df[col].values

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    encodings = tokenizer(list(texts), truncation=True, padding='max_length', max_length=MAX_LENGTH)

    # Split data into train and validation sets
    input_ids_train, input_ids_val, labels_train, labels_val, attention_mask_train, attention_mask_val = train_test_split(
        encodings['input_ids'], labels, encodings['attention_mask'], stratify=labels, test_size=test_size
    )

    train_dataset = TensorDataset(torch.tensor(input_ids_train, dtype=torch.long),
                                  torch.tensor(attention_mask_train, dtype=torch.long),
                                  torch.tensor(labels_train, dtype=torch.long))

    val_dataset = TensorDataset(torch.tensor(input_ids_val, dtype=torch.long),
                                torch.tensor(attention_mask_val, dtype=torch.long),
                                torch.tensor(labels_val, dtype=torch.long))

    return train_dataset, val_dataset, len(set(labels))


def train():
    # Initialize a new wandb run
    with wandb.init(project=PROJECT_NAME, config=SWEEP_CONFIG) as run:
        # Set configs for the run
        config = wandb.config

        # Load data
        train_dataset, val_dataset, num_labels = load_data(FILE_PATH, col=CLASSIFICATION_TYPE, max_length=MAX_LENGTH)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

        model.to(device)
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        # Training loop
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            start_time = time.time()
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Training"):
                batch = [b.to(device) for b in batch]
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                optimizer.zero_grad()
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            wandb.log({"epoch": epoch, "train_loss": avg_loss})

            # Evaluation
            model.eval()
            val_preds, val_true = [], []
            val_loss = 0
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Evaluation"):
                batch = [b.to(device) for b in batch]
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(batch[2].cpu().numpy())

                loss = outputs.loss
                val_loss += loss.item()

            # Compute and log metrics
            val_accuracy = accuracy_score(val_true, val_preds)
            val_f1_macro = f1_score(val_true, val_preds, average='macro')
            val_f1_micro = f1_score(val_true, val_preds, average='micro')
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{EPOCHS} completed in {epoch_time:.2f}s')
            print('Validation Loss:', val_loss / len(val_loader))
            print('Validation Accuracy:', val_accuracy)
            print('Validation F1 Macro:', val_f1_macro)
            print('Validation F1 Micro:', val_f1_micro)
            wandb.log({"val_loss": val_loss / len(val_loader), "val_accuracy": val_accuracy, "f1_macro": val_f1_macro, "f1_micro": val_f1_micro})

def main():
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=PROJECT_NAME)
    wandb.agent(sweep_id, function=train)

if __name__ == "__main__":
    main()
