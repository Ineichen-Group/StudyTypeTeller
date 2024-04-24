import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import pandas as pd
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt


def setup_logger(log_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'), mode='a') # TODO append logs instead of overwriting
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_data_splits(data_dir, col_name, tokenizer_name, batch_size):
    train_file = os.path.join(data_dir, 'train.csv')
    val_file = os.path.join(data_dir, 'val.csv')
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    # Concatenate 'journal_name', 'title', and 'abstract' columns to create 'text' column
    train_df['text'] = train_df['journal_name'] + ' ' + train_df['title'] + ' ' + train_df['abstract']
    val_df['text'] = val_df['journal_name'] + ' ' + val_df['title'] + ' ' + val_df['abstract']


    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_encodings = tokenizer(train_df['text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors='pt')
    val_encodings = tokenizer(val_df['text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors='pt')

    train_labels = torch.tensor(train_df[col_name].values)
    val_labels = torch.tensor(val_df[col_name].values)

    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)

    return DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size), DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)


def train_model(model_name, tokenizer_name, col_name, epochs, patience, batch_size, learning_rate, weight_decay, data_dir, save_dir, classification_type, logger):
    log_dir = os.path.join(save_dir, 'logs')  # logs are saved here
    os.makedirs(log_dir, exist_ok=True) # create if does not exist yet

    if classification_type == 'binary':
        num_labels = 2
    elif classification_type == 'multi':
        num_labels = 14 # TODO double-check the classes in new data split
    else:
        raise ValueError("Invalid classification_type. Must be either 'binary' or 'multi'.")

    train_dataloader, val_dataloader = load_data_splits(data_dir, col_name, tokenizer_name, batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_dataloader) * epochs
    num_warmup_steps = int(total_steps * 0.1) # TODO modify to adapt dynamically
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

    best_val_loss = float('inf')
    best_model_path = None
    no_improvement_count = 0  # Initialize the counter for epochs without improvement
    # for plotting
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        tqdm_train_dataloader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for batch in tqdm_train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            optimizer.zero_grad()
            outputs = model(**inputs)
            logits = outputs.logits
            loss = criterion(logits, batch[2])  # Compute cross-entropy loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            tqdm_train_dataloader.set_postfix({'training_loss': train_loss / (tqdm_train_dataloader.n + 1)})  # Update the progress bar with training loss

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                outputs = model(**inputs)
                logits = outputs.logits
                loss = criterion(logits, batch[2])  # Compute cross-entropy loss
                val_loss += loss.item()

                # Append predictions and labels for F1-score computation
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch[2].cpu().numpy())

        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)

        # Save the model if it has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_model_path:
                os.remove(best_model_path)  # Remove the previously saved best model
                os.remove(best_model_path.replace('.pt', '_FULL.pt'))  # Remove the previously saved best full model
            best_model_path = os.path.join(save_dir, f"{model_name.replace('/', '_')}_{classification_type}_epoch_{epoch+1}_val_loss_{val_loss:.4f}.pt")
            torch.save(model.state_dict(), best_model_path)  # Save state dict
            torch.save(model, best_model_path.replace('.pt', '_FULL.pt'))  # Save entire model
            no_improvement_count = 0  # Reset the counter since there's improvement

        f1 = f1_score(val_labels, val_preds, average='weighted')
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"Train Loss: {train_loss}")
        logger.info(f"Validation Loss: {val_loss}")
        logger.info(f"Validation F1-score: {f1}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Validation Loss: {val_loss}")
        print(f"Validation F1-score: {f1}")
        

        # Early stopping / patience
        if no_improvement_count >= patience:
            logger.info(f"Early stopping as validation loss didn't improve for {patience} consecutive epochs.")
            print((f"Early stopping as validation loss didn't improve for {patience} consecutive epochs."))
            break

        # Increment the counter if there's no improvement
        no_improvement_count += 1

    model_name = model_name.replace("/", "_") # to avoid dir creation error
    plot_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Plot and save the training loss
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss: {model_name}')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'{model_name}_train_loss_plot.png'))
    plt.show()
    plt.close()

    # Plot and save the validation loss
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Validation Loss: {model_name}')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'{model_name}_val_loss_plot.png'))
    plt.show()
    plt.close()


def main(classification_type):

    data_dir = "./../../data/data_splits_stratified/6-2-2_all_classes"
    save_dir = f"./../../models/transformers/checkpoints/{classification_type}"
    models_to_fine_tune = [
        ('bert-base-uncased', 'bert-base-uncased'),
        ('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'),
        ('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract', 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'),
        ('allenai/scibert_scivocab_uncased', 'allenai/scibert_scivocab_uncased'),
        ('dmis-lab/biobert-v1.1', 'dmis-lab/biobert-v1.1')
    ]


    # select label column based on classification type
    if classification_type == 'binary':
        col_name = 'binary_label'
        log_dir = './logs/binary'
    else:
        col_name = 'multi_label'
        log_dir = './logs/multi'
    
    # Ensure that the directory exists
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir)

    for model_name, tokenizer_name in models_to_fine_tune:
        logger.info(f"\n\n************** Fine-tuning {model_name} {classification_type} **************")
        print((f"\n\n************** Fine-tuning {model_name} {classification_type} **************"))
        train_model(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            col_name=col_name,
            epochs=10,
            patience=4,
            batch_size=8,
            learning_rate=5e-5,
            weight_decay=5e-4,
            data_dir=data_dir,
            save_dir=save_dir,
            classification_type=classification_type,
            logger=logger
        )

if __name__ == "__main__":
    main(classification_type='multi')
