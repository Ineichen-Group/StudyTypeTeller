import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from transformers import AutoTokenizer
import numpy as np

def load_model(model_path):
    # Load the trained model from the specified path
    model = torch.load(model_path)  # Adjust based on how your model was saved (full model or state_dict)
    return model

def inference_on_new_data(new_data, model_path, model_name, batch_size=8):
    """
    Perform inference on new, unlabeled data using a trained model.

    Parameters:
    - new_data (pd.DataFrame): DataFrame containing new data with columns 'PMID', 'journal_name', 'title', 'abstract', and optionally 'keywords'.
    - model_path (str): Path to the saved model.
    - model_name (str): Name of the model to load tokenizer and perform inference.
    - batch_size (int): Batch size for inference.

    Returns:
    - pd.DataFrame: DataFrame containing PMID, label predictions, and confidence scores for each sample.
    """
    # Load model and tokenizer
    model = load_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Preprocess and tokenize the new data
    def concatenate_text(row):
        text_parts = [str(row['journal_name']), str(row['title']), str(row['abstract'])]
        keywords = row.get('keywords', None)
        if pd.notna(keywords):
            keywords_list = keywords.split('|')
            text_parts.extend(keywords_list)
        return ' '.join(text_parts)

    new_data['text'] = new_data.apply(concatenate_text, axis=1)
    encodings = tokenizer(new_data['text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors='pt')
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size)

    predictions = []
    confidences = []

    # Perform inference
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = [t.to(device) for t in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            predictions.extend(preds.cpu().numpy())
            confidences.extend(probs.max(dim=1).values.cpu().numpy())

    # Construct DataFrame for output with PMID and label
    result_df = pd.DataFrame({
        'PMID': new_data['PMID'],
        'label': predictions,
        'confidence': confidences
    })

    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate model.")
  
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="michiyasunaga/BioLinkBERT-base",
        help="The name or path of the HuggingFace model to use. For example, 'michiyasunaga/BioLinkBERT-base'."
    )
    parser.add_argument(
        "--trained_model_path",
        type=str,
        default="BioLinkBERT-base_1436_model.pt",
        help="Path to the fine-tuned .pt file of the HuggingFace model."
    )
    parser.add_argument(
        "--pubmed_file",
        type=str,
        default="./pmid_contents_chunk_1.txt",
        help="File with PubMed content."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./predictions_chunk_1.txt",
        help="File name to save predictions, e.g. ./model_predictions/neuro_pubmed/predictions_chunk_0.txt."
    )

    args = parser.parse_args()
   
    model_name = args.model_name_or_path
    model_path = args.trained_model_path
    model_name_clean = model_name.split("/")[1]
    
    input_file_path = args.pubmed_file
    out_file = args.output_file

    headers = ["PMID", "Year", "journal_name", "title", "abstract", "DOI"]

    new_data = pd.read_csv(input_file_path, sep=r'\^!\^', names=headers,  engine='python')  # Change 'sep' if files use a different delimiter

    # Perform inference
    results = inference_on_new_data(new_data, model_path, model_name)
    results.to_csv(out_file)