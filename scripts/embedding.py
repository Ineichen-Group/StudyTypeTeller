import pandas as pd
import xml.etree.ElementTree as et
import os
import torch
import numpy as np
def generate_embeddings(abstracts, tokenizer, model, device, classification_model=False):
    """Generate embeddings using BERT-based model.
    Code from Luca Schmidt, see https://github.com/berenslab/pubmed-landscape/blob/main/pubmed_landscape_src/data.py

    Parameters
    ----------
    abstracts : list
        Abstract texts.
    tokenizer : transformers.models.bert.tokenization_bert_fast.BertTokenizerFast
        Tokenizer.
    model : transformers.models.bert.modeling_bert.BertModel
        BERT-based model.
    device : str, {"cuda", "cpu"}
        "cuda" if torch.cuda.is_available() else "cpu".
    classification_model: bool
        If the model was initialized as AutoModelForTokenClassification.

    Returns
    -------
    embedding_cls : ndarray
        [CLS] tokens of the abstracts.
    embedding_sep : ndarray
        [SEP] tokens of the abstracts.
    embedding_av : ndarray
        Average of tokens of the abstracts.
    """
    # preprocess the input
    inputs = tokenizer(
        abstracts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    ).to(device)

    # inference
    outputs = model(**inputs)[0].cpu().detach()

    if classification_model:
        outputs_classif_model = model(**inputs, output_hidden_states=True)#[1].cpu().detach()
        outputs = outputs_classif_model.hidden_states[-1].cpu().detach()

    embedding_av = torch.mean(outputs, [0, 1]).numpy()
    embedding_sep = outputs[:, -1, :].numpy()
    embedding_cls = outputs[:, 0, :].numpy()

    return embedding_cls, embedding_sep, embedding_av