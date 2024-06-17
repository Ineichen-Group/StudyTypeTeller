# wandb api-keys: 
# binary: 2115d24f1160165e03f31b6d0327dde07b3a9876
# multi: 

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import os
from tqdm import tqdm
import time
import wandb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler

# Constants
PROJECT_NAME = 'SciBERT_ep6_binary'
TOKENIZER_NAME = "allenai/scibert_scivocab_uncased"
MODEL_NAME = 'allenai/scibert_scivocab_uncased'
DATA_DIR = './../data/data_splits_stratified'
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

if CLASSIFICATION_TYPE == "NUM_LABEL_BINARY":
    NUM_LABELS = 2
else:
    NUM_LABELS = 13


def load_data(max_length):
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_data.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val_data.csv'))

    print("Validation Dataset Length before tokenization:", len(val_df))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    train_encodings = tokenizer(
        train_df['text'].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    val_encodings = tokenizer(
        val_df['text'].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    print("Train Encodings Shape:", train_encodings['input_ids'].shape)
    print("Val Encodings Shape:", val_encodings['input_ids'].shape)

    train_labels = torch.tensor(train_df[CLASSIFICATION_TYPE].values)  # Assuming col_name contains binary labels (0 or 1)
    val_labels = torch.tensor(val_df[CLASSIFICATION_TYPE].values)

    print("Train Labels Shape:", train_labels.shape)
    print("Val Labels Shape:", val_labels.shape)

    train_dataset = TensorDataset(
        train_encodings['input_ids'],
        train_encodings['attention_mask'],
        train_labels
    )
    val_dataset = TensorDataset(
        val_encodings['input_ids'],
        val_encodings['attention_mask'],
        val_labels
    )

    print("Validation Dataset Length after tokenization:", len(val_dataset))

    return train_dataset, val_dataset




# def train():
#     with wandb.init() as run:
#         # Set configs for the run
#         config = run.config

#         train_dataset, val_dataset = load_data(max_length=MAX_LENGTH)
#         print("Train Dataset Length:", len(train_dataset))
#         print("Validation Dataset Length:", len(val_dataset))

#         train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
#         val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
#         print("Train Dataloader Length:", len(train_dataloader))
#         print("Validation Dataloader Length:", len(val_dataloader))

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  # num_labels=2 for binary classification
#         print(f'Model: \n{model}')
#         optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
#         total_steps = len(train_dataloader) * EPOCHS
#         print(f'Total steps: {total_steps}')
#         scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)

#         for epoch in range(EPOCHS):
#             start_time = time.time()  # Start time for the epoch
#             model.to(device)
#             model.train()

#             total_loss = 0
#             train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} Training", leave=False)
#             for batch in train_iterator: # iterate through batches
#                 batch = [b.to(device) for b in batch] # send to device
#                 inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
#                 optimizer.zero_grad()
#                 outputs = model(**inputs)
#                 loss = outputs.loss
#                 loss.backward()
#                 optimizer.step()
#                 scheduler.step()

#                 total_loss += loss.item()
#                 train_iterator.set_postfix({'train_loss': total_loss / len(train_iterator)})  # Update progress bar with current loss

#             # Log average training loss for the epoch
#             avg_train_loss = total_loss / len(train_dataloader)
#             wandb.log({"epoch": epoch, "train_loss": avg_train_loss})
            
#             # Calculate elapsed time for the epoch
#             epoch_time = time.time() - start_time
#             print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds.")


#             # Validation
#             model.eval()
#             val_loss = 0.0
#             val_pred, val_true = [],[]
#             val_iterator = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} Validation", leave=False)
#             for batch in val_iterator: # iterate through batches
#                 batch = [b.to(device) for b in batch]
#                 inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
#                 outputs = model(**inputs)
#                 if outputs is not None and outputs.loss is not None:  # Check if outputs and loss exist
#                     val_loss += outputs.loss.item()
#                     logits = outputs.logits
#                     preds = torch.argmax(logits, dim=1).cpu().numpy()
#                     val_pred.extend(preds)
#                     val_true.extend(batch[2].cpu().numpy())

#             print(f"Epoch {epoch+1}: Number of validation predictions: {len(val_true)}, Number of validation labels: {len(val_pred)}")


#             accuracy = accuracy_score(val_true, val_pred)
#             precision = precision_score(val_true, val_pred, zero_division=0)
#             recall = recall_score(val_true, val_pred, zero_division=0)
#             macro_f1 = f1_score(val_true, val_pred, zero_division=0, average='macro')
#             micro_f1 = f1_score(val_true, val_pred, zero_division=0, average='micro')

#             # Log metrics to W&B
#             wandb.log({
#                 "epoch": epoch,
#                 "val_loss": val_loss / len(val_dataloader),
#                 "accuracy": accuracy,
#                 "macro_f1": macro_f1,
#                 "micro_f1": micro_f1}
#             )

#             # Print metrics for each epoch
#             print(f"Epoch {epoch+1}: val_loss: {val_loss / len(val_dataloader)}, accuracy: {accuracy}, macro_f1: {macro_f1}, micro_f1: {micro_f1}")


def train():
    with wandb.init() as run:
        # Set configs for the run
        config = run.config

        train_dataset, val_dataset = load_data(max_length=MAX_LENGTH)
        print("Train Dataset Length:", len(train_dataset))
        print("Validation Dataset Length:", len(val_dataset))

        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        print("Train Dataloader Length:", len(train_dataloader))
        print("Validation Dataloader Length:", len(val_dataloader))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  # num_labels=2 for binary classification
        print(f'Model: \n{model}')
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        total_steps = len(train_dataloader) * EPOCHS
        print(f'Total steps: {total_steps}')
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)

        for epoch in range(EPOCHS):
            start_time = time.time()  # Start time for the epoch
            model.to(device)
            model.train()

            total_loss = 0
            train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} Training", leave=False)
            for batch in train_iterator: # iterate through batches
                batch = [b.to(device) for b in batch] # send to device
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                optimizer.zero_grad()
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                train_iterator.set_postfix({'train_loss': total_loss / len(train_iterator)})  # Update progress bar with current loss

            # Log average training loss for the epoch
            avg_train_loss = total_loss / len(train_dataloader)
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss})
            
            # Calculate elapsed time for the epoch
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds.")


            # Validation
            model.eval()
            val_loss = 0.0
            val_pred, val_true = [],[]
            val_iterator = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} Validation", leave=False)
            for batch in val_iterator: # iterate through batches
                batch = [b.to(device) for b in batch]
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                outputs = model(**inputs)
                if outputs is not None and outputs.loss is not None:  # Check if outputs and loss exist
                    val_loss += outputs.loss.item()
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    val_pred.extend(preds)
                    val_true.extend(batch[2].cpu().numpy())

            print(f"Epoch {epoch+1}: Number of validation predictions: {len(val_true)}, Number of validation labels: {len(val_pred)}")


            accuracy = accuracy_score(val_true, val_pred)
            precision = precision_score(val_true, val_pred, zero_division=0)
            recall = recall_score(val_true, val_pred, zero_division=0)
            macro_f1 = f1_score(val_true, val_pred, zero_division=0, average='macro')
            micro_f1 = f1_score(val_true, val_pred, zero_division=0, average='micro')

            # Log metrics to W&B
            wandb.log({
                "epoch": epoch,
                "val_loss": val_loss / len(val_dataloader),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1
            })

            # Print metrics for each epoch
            print(f"Epoch {epoch+1}: val_loss: {val_loss / len(val_dataloader)}, accuracy: {accuracy}, precision: {precision}, recall: {recall}, macro_f1: {macro_f1}, micro_f1: {micro_f1}")




def main():



    sweep_id = wandb.sweep(SWEEP_CONFIG, project=PROJECT_NAME)
    wandb.agent(sweep_id, function=train, count=100)

    wandb.finish() # terminate wandb session to avoid lags and overlaps


if __name__ == "__main__":
    main()
