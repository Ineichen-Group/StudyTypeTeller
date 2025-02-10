""" Classification report similar to https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
"""

from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import sklearn
from confidenceinterval.takahashi_methods import  precision_score, recall_score, f1_score
from confidenceinterval.binary_metrics import accuracy_score
from confidenceinterval.bootstrap import bootstrap_ci

def round_tuple(t, decimals=3):
    return tuple(round(num, decimals) for num in t)

def classification_report_with_ci(y_true: List[int], y_pred: List[int], 
                                  binary_method: str = 'wilson',
                                  round_ndigits: int = 3,
                                  numerical_to_label_map: Optional[Dict[int, str]] = None, 
                                  confidence_level: float = 0.95) -> pd.DataFrame:
    """
    Parameters
    ----------
    y_true : List[int]
        The ground truth labels.
    y_pred : List[int]
        The predicted categories.
    binary_method: str = 'wilson'
        The method to calculate the CI for binary proportions.
    round_ndigits: int = 3
        Number of digits to return after the decimal point.
    numerical_to_label_map: Optional[Dict[int, str]]
        Mapping from class indices to descriptive names.
    confidence_level: float, optional
        The confidence level, by default 0.95

    Returns
    -------
    pd.DataFrame
        A DataFrame containing precision, recall, F1-score, and their confidence intervals for each class,
        as well as micro and macro averages.
    """

    # Unique classes in the dataset
    classes = np.unique(y_true)

    # Validate that all unique classes are covered in the numerical_to_label_map if provided
    if numerical_to_label_map is not None:
        missing_labels = [cls for cls in classes if cls not in numerical_to_label_map]
        if missing_labels:
            raise ValueError(f'Missing labels for classes: {missing_labels}')

    data = []  # List to store row dictionaries

    # Unique classes in the dataset
    classes = np.unique(y_true)

    # Calculate precision, recall, f1 for each class treated as binary
    for class_ in classes:
        y_true_binary = [1 if y == class_ else 0 for y in y_true]
        y_pred_binary = [1 if y == class_ else 0 for y in y_pred]

        # Calculate metrics
        precision, precision_ci = precision_score(y_true_binary, y_pred_binary, average='binary', method=binary_method)
        recall, recall_ci = recall_score(y_true_binary, y_pred_binary, average='binary', method=binary_method)
        binary_f1, binary_f1_ci = f1_score(y_true_binary, y_pred_binary, confidence_level=confidence_level, average='binary')
        accuracy, accuracy_ci = accuracy_score(y_true_binary, y_pred_binary, confidence_level=confidence_level, method=binary_method)

        class_name = numerical_to_label_map[class_] if (
                    numerical_to_label_map and class_ in numerical_to_label_map) else f'Class {class_}'
        support = sum(y_true_binary)

        # Create a new row as a DataFrame and append it to the main DataFrame
        # Append new row to the list
        data.append({
            'class': class_name,
            'precision': round(precision, round_ndigits),
            'recall': round(recall, round_ndigits),
            'f1-score': round(binary_f1, round_ndigits),
            'accuracy': round(accuracy, round_ndigits),
            'precision CI': round_tuple(precision_ci, round_ndigits),
            'recall CI': round_tuple(recall_ci, round_ndigits),
            'f1-score CI': round_tuple(binary_f1_ci, round_ndigits),
            'accuracy CI': round_tuple(accuracy_ci, round_ndigits),
            'support': support
        })

    random_generator = np.random.default_rng()

    precision_micro, p_ci_micro = precision_score(y_true, y_pred, average='micro')
    precision_macro, p_ci_macro = precision_score(y_true, y_pred, average='macro')
    precision_weighted, p_ci_weighted = bootstrap_ci(y_true=y_true,
                                         y_pred=y_pred,
                                         metric=sklearn.metrics.precision_score,
                                         metric_average='weighted',
                                         confidence_level=0.95,
                                         n_resamples=9999,
                                         method='bootstrap_bca',
                                         random_state=random_generator)

    recall_micro, r_ci_micro = recall_score(y_true, y_pred, average='micro')
    recall_macro, r_ci_macro = recall_score(y_true, y_pred, average='macro')
    recall_weighted, r_ci_weighted = bootstrap_ci(y_true=y_true,
                                         y_pred=y_pred,
                                         metric=sklearn.metrics.recall_score,
                                         metric_average='weighted',
                                         confidence_level=0.95,
                                         n_resamples=9999,
                                         method='bootstrap_bca',
                                         random_state=random_generator)

    f1_micro, f1_ci_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro, f1_ci_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted, f1_ci_weighted = bootstrap_ci(y_true=y_true,
                                         y_pred=y_pred,
                                         metric=sklearn.metrics.f1_score,
                                         metric_average='weighted',
                                         confidence_level=0.95,
                                         n_resamples=9999,
                                         method='bootstrap_bca',
                                         random_state=random_generator)
    
    accuracy, accuracy_ci = bootstrap_ci(y_true=y_true,
                             y_pred=y_pred,
                             metric=sklearn.metrics.accuracy_score,
                             metric_average='n.a.',
                             confidence_level=0.95,
                             n_resamples=9999,
                             method='bootstrap_bca',
                             random_state=random_generator)
    accuracy_rounded = round(accuracy, round_ndigits)
    accuracy_ci_rounded = round_tuple(accuracy_ci, round_ndigits)

    data.append({
        'class': 'micro',
        'precision': round(precision_micro, round_ndigits),
        'recall': round(recall_micro, round_ndigits),
        'f1-score': round(f1_micro, round_ndigits),
        #'accuracy': accuracy_rounded,
        'precision CI': round_tuple(p_ci_micro, round_ndigits),
        'recall CI': round_tuple(r_ci_micro, round_ndigits),
        'f1-score CI': round_tuple(f1_ci_micro, round_ndigits),
        #'accuracy CI': accuracy_ci_rounded,
        'support': len(y_true)
    })

    data.append({
        'class': 'macro',
        'precision': round(precision_macro, round_ndigits),
        'recall': round(recall_macro, round_ndigits),
        'f1-score': round(f1_macro, round_ndigits),
        #'accuracy': accuracy_rounded,
        'precision CI': round_tuple(p_ci_macro, decimals=round_ndigits),
        'recall CI': round_tuple(r_ci_macro, decimals=round_ndigits),
        'f1-score CI': round_tuple(f1_ci_macro, decimals=round_ndigits),
        #'accuracy CI': accuracy_ci_rounded,
        'support': len(y_true)

    })
    data.append({
        'class': 'weighted avg',
        'precision': round(precision_weighted, round_ndigits),
        'recall': round(recall_weighted, round_ndigits),
        'f1-score': round(f1_weighted, round_ndigits),
        'accuracy': accuracy_rounded,
        'precision CI': round_tuple(p_ci_weighted, decimals=round_ndigits),
        'recall CI': round_tuple(r_ci_weighted, decimals=round_ndigits),
        'f1-score CI': round_tuple(f1_ci_weighted, decimals=round_ndigits),
        'accuracy CI': accuracy_ci_rounded,
        'support': len(y_true)

    })
    data.append({
        'class': 'accuracy',
        'precision': accuracy_rounded,
        'recall': accuracy_rounded,
        'f1-score': accuracy_rounded,
        'accuracy': accuracy_rounded,
        'precision CI': accuracy_ci_rounded,
        'recall CI': accuracy_ci_rounded,
        'f1-score CI': accuracy_ci_rounded,
        'accuracy CI': accuracy_ci_rounded,
        'support': len(y_true)

    })

    df = pd.DataFrame(data)

    return df