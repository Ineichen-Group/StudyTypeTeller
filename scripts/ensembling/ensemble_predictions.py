import pandas as pd
import numpy as np
import random
import os

def filter_prediction_columns(predictions_df):
    prediction_cols = [col for col in predictions_df.columns if col.endswith('_prediction')]
    return predictions_df[['true_label'] + prediction_cols]

def simple_majority_voting(predictions_df):
    random.seed(42)
    predictions = []
    
    for i in range(len(predictions_df)):
        votes = predictions_df.iloc[i, 1:].values.tolist()
        
        majority_vote = max(set(votes), key=votes.count)
        if votes.count(majority_vote) == len(votes) / 2:
            majority_vote = random.choice(votes)
        predictions.append(majority_vote)
    
    return predictions

def confidence_based_voting(predictions_df, full_df):
    predictions = []
    
    for i in range(len(predictions_df)):
        votes = predictions_df.iloc[i, 1:].values.tolist()
        confidences = {}
        
        for j, vote in enumerate(votes):
            confidence_col = predictions_df.columns[j + 1].replace('_prediction', '_probability')
            confidence = full_df.iloc[i][confidence_col]
            if vote not in confidences:
                confidences[vote] = []
            confidences[vote].append(confidence)
        
        majority_vote = max(set(votes), key=votes.count)
        if votes.count(majority_vote) == len(votes) / 2:
            average_confidences = {vote: np.mean(conf) for vote, conf in confidences.items()}
            majority_vote = max(average_confidences, key=average_confidences.get)
        predictions.append(majority_vote)
    
    return predictions

def generate_predictions_csv(predictions, true_labels, output_file):
    df = pd.DataFrame({
        'prediction': predictions,
        'true_label': true_labels
    })
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":

    experiment = "rerun_finetuning_19-06-24" # TODO modify as needed
    classification_types = ["multi", "binary"] # iterate through the two classification types

    for classification_type in classification_types:
    
        input_dir = f'./../../models/transformers/evaluations/{experiment}/ensemble/aggregated_output/{classification_type}'
        output_dir = f'./../../models/transformers/evaluations/{experiment}/ensemble/{classification_type}'
        os.makedirs(output_dir, exist_ok=True) # do not overwrite

        # Load aggregated predictions CSV
        aggregated_predictions_file = f'{input_dir}/aggregated_predictions.csv'
        full_predictions_df = pd.read_csv(aggregated_predictions_file)

        # Filter only _prediction columns and true_label
        predictions_df = filter_prediction_columns(full_predictions_df)

        # Perform simple majority voting with random tie-breaking
        predictions_simple_majority = simple_majority_voting(predictions_df)
        output_file_simple_majority = f'{output_dir}/simple_voting_predictions/predictions.csv'
        simple_voting_predictions_dir = f"{output_dir}/simple_voting_predictions"
        os.makedirs(simple_voting_predictions_dir, exist_ok=True)
        generate_predictions_csv(predictions_simple_majority, predictions_df['true_label'], output_file_simple_majority)

        # Perform majority voting with confidence tie-breaking
        predictions_confidence_based = confidence_based_voting(predictions_df, full_predictions_df)
        output_file_confidence_based = f'{output_dir}/confidence_based_voting_predictions/predictions.csv'
        confidence_based_voting_predictions_dir = f"{output_dir}/confidence_based_voting_predictions"
        os.makedirs(confidence_based_voting_predictions_dir, exist_ok=True)
        generate_predictions_csv(predictions_confidence_based, predictions_df['true_label'], output_file_confidence_based)
