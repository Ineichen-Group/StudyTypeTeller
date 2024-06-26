import os
import logging
import pandas as pd
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ExperimentManager:
    def __init__(self, classification_type, experiment_name):
        self.data_dir = "./../../data/data_splits_stratified/6-2-2_all_classes_enriched_with_kw"
        self.save_dir = f"./../../models/transformers/checkpoints/{experiment_name}/{classification_type}/models"
        self.log_dir = f"./../../models/transformers/checkpoints/{experiment_name}/{classification_type}/logs"
        self.classification_type = classification_type
        self.models_to_fine_tune = [
            'bert-base-uncased',
            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
            'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
            'allenai/scibert_scivocab_uncased',
            'dmis-lab/biobert-v1.1',
            'michiyasunaga/BioLinkBERT-base',
            'emilyalsentzer/Bio_ClinicalBERT',
        ]

        if classification_type == 'binary':
            self.col_name = 'binary_label'
            self.num_labels = 2
        elif classification_type == 'multi':
            self.col_name = 'multi_label'
            self.num_labels = 14  # TODO change to dynamic
        else:
            raise ValueError("Invalid classification_type. Must be either 'binary' or 'multi'.")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.log_dir, 'training.log'), mode='a')  # do not overwrite log
        file_handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(file_handler)
        return logger

    def get_short_model_name(self, model_name):
        model_names = {
            'bert-base-uncased': 'bert-base',
            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext': 'PubMedBERT',
            'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract': 'BiomedBERT',
            'allenai/scibert_scivocab_uncased': 'Scibert',
            'dmis-lab/biobert-v1.1': 'biobert',
            'michiyasunaga/BioLinkBERT-base': 'BioLinkBERT',
            'emilyalsentzer/Bio_ClinicalBERT': 'Bio_ClinicalBERT'
        }
        if model_name in model_names.keys():
            return model_names[model_name]
        else:
            raise ValueError("Invalid model name. Cannot be mapped to short name.")

    def load_data_splits(self, tokenizer_name, batch_size):
        train_file = os.path.join(self.data_dir, 'train.csv')
        val_file = os.path.join(self.data_dir, 'val.csv')
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)

        def concatenate_text(row):
            text_parts = [str(row['journal_name']), str(row['title']), str(row['abstract'])]
            keywords = row['keywords']
            if pd.notna(keywords):
                keywords_list = keywords.split('|')
                text_parts.extend(keywords_list)
            return ' '.join(text_parts)

        train_df['text'] = train_df.apply(concatenate_text, axis=1)
        val_df['text'] = val_df.apply(concatenate_text, axis=1)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        train_encodings = tokenizer(train_df['text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors='pt')
        val_encodings = tokenizer(val_df['text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors='pt')

        train_labels = torch.tensor(train_df[self.col_name].values)
        val_labels = torch.tensor(val_df[self.col_name].values)

        train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
        val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)

        return DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size), DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

    def run_experiment(self):
        for model_name in self.models_to_fine_tune:
            self.logger.info(f"\n\n************** Fine-tuning {model_name}: {self.classification_type} **************")
            print(f"\n\n************** Fine-tuning {model_name}: {self.classification_type} **************")
            train_dataloader, val_dataloader = self.load_data_splits(model_name, batch_size=8)
            model_finetuner = ModelFinetuner(
                model_name=model_name,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                col_name=self.col_name,
                num_labels=self.num_labels,
                epochs=10,
                patience=4,
                learning_rate=5e-5,
                weight_decay=5e-4,
                save_dir=self.save_dir,
                classification_type=self.classification_type,
                experiment_name=experiment_name,
                logger=self.logger
            )
            model_finetuner.finetune_model()


class ModelFinetuner:
    def __init__(self, model_name, train_dataloader, val_dataloader, col_name, num_labels, epochs, patience, learning_rate, weight_decay, save_dir, classification_type, experiment_name, logger):
        self.model_name = model_name
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.col_name = col_name
        self.num_labels = num_labels
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_dir = save_dir
        self.classification_type = classification_type
        self.experiment_name = experiment_name
        self.logger = logger
        self.model_dir = os.path.join(save_dir, ExperimentManager(classification_type, experiment_name).get_short_model_name(model_name))
        os.makedirs(self.model_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def finetune_model(self):
        model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels).to(self.device)
        optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        total_steps = len(self.train_dataloader) * self.epochs
        num_warmup_steps = int(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

        best_val_loss = float('inf')
        best_model_path = None
        no_improvement_count = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            model.train()
            train_loss = 0.0
            tqdm_train_dataloader = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch")
            for batch in tqdm_train_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                optimizer.zero_grad()
                outputs = model(**inputs)
                logits = outputs.logits
                loss = criterion(logits, batch[2])
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
                tqdm_train_dataloader.set_postfix({'training_loss': train_loss / (tqdm_train_dataloader.n + 1)})

            train_loss /= len(self.train_dataloader)
            train_losses.append(train_loss)

            model.eval()
            val_preds, val_labels = [], []
            val_loss = 0.0

            with torch.no_grad():
                for batch in self.val_dataloader:
                    batch = tuple(t.to(self.device) for t in batch)
                    inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                    outputs = model(**inputs)
                    logits = outputs.logits
                    loss = criterion(logits, batch[2])
                    val_loss += loss.item()
                    preds = torch.argmax(logits, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(batch[2].cpu().numpy())

            val_loss /= len(self.val_dataloader)
            val_losses.append(val_loss)

            checkpoint_path = os.path.join(self.model_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(self.model_dir, f"ft_{ExperimentManager(self.classification_type, self.experiment_name).get_short_model_name(self.model_name)}_{self.classification_type}.pt")
                torch.save(model.state_dict(), best_model_path)
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            f1 = f1_score(val_labels, val_preds, average='weighted')
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            self.logger.info(f"Train Loss: {train_loss}")
            self.logger.info(f"Validation Loss: {val_loss}")
            self.logger.info(f"Validation F1-score: {f1}")
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"Train Loss: {train_loss}")
            print(f"Validation Loss: {val_loss}")
            print(f"Validation F1-score: {f1}")

            if no_improvement_count >= self.patience:
                self.logger.info(f"Early stopping as validation loss didn't improve for {self.patience} consecutive epochs.")
                print(f"Early stopping as validation loss didn't improve for {self.patience} consecutive epochs.")
                break

        self.plot_losses(train_losses, val_losses, 'train_loss', 'Training Loss')
        self.plot_losses(val_losses, val_losses, 'val_loss', 'Validation Loss')

    def plot_losses(self, losses, val_losses, file_name, title):
        plt.plot(range(1, len(losses) + 1), losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{title}: {ExperimentManager(self.classification_type, self.experiment_name).get_short_model_name(self.model_name)}')
        plt.legend()
        plt.savefig(os.path.join(self.model_dir, f'{file_name}.png'))
        plt.show()
        plt.close()


if __name__ == "__main__":
    # TODO choose experiment name to avoid overwriting
    experiment_name = "rerun_finetuning_19-06-24"
    # keep this seed
    seed = 42 
    set_seed(seed)
    # TODO select from 'binary' or 'multi'
    experiment = ExperimentManager(classification_type='binary', experiment_name=experiment_name)

    experiment.run_experiment()
