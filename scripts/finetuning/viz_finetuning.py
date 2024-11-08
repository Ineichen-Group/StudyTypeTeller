import re
import pandas as pd
import matplotlib.pyplot as plt

def main():

    # Define the log file path
    log_file_path = "scripts/finetuning/training.log"

    # Lists to store parsed data
    model_names = []
    epochs = []
    validation_losses = []
    validation_f1_scores = []

    # Regular expressions to capture necessary information
    model_pattern = re.compile(r"\*\*\*\*\*\*\*\*\*\*\*\*\*\* Fine-tuning (.*): multi \*\*\*\*\*\*\*\*\*\*\*\*\*\*")
    epoch_pattern = re.compile(r"Epoch (\d+)/\d+")
    val_loss_pattern = re.compile(r"Validation Loss: ([\d\.]+)")
    val_f1_pattern = re.compile(r"Validation F1-score: ([\d\.]+)")

    # Initialize variables
    current_model = None

    # Parse the log file
    with open(log_file_path, "r") as f:
        for line in f:
            # Match model name
            model_match = model_pattern.search(line)
            if model_match:
                current_model = model_match.group(1)
            
            # Match epoch
            epoch_match = epoch_pattern.search(line)
            if epoch_match and current_model:
                current_epoch = int(epoch_match.group(1))
            
            # Match validation loss
            val_loss_match = val_loss_pattern.search(line)
            if val_loss_match and current_model:
                validation_loss = float(val_loss_match.group(1))
                validation_losses.append(validation_loss)
                epochs.append(current_epoch)
                model_names.append(current_model)
            
            # Match validation F1-score
            val_f1_match = val_f1_pattern.search(line)
            if val_f1_match and current_model:
                validation_f1 = float(val_f1_match.group(1))
                validation_f1_scores.append(validation_f1)

    # Create a DataFrame with parsed data
    df = pd.DataFrame({
        "Model": model_names,
        "Epoch": epochs,
        "Validation Loss": validation_losses,
        "Validation F1-score": validation_f1_scores
    })
    print(df)
    
    # Find the best Validation F1-score and corresponding Validation Loss for each model
    best_scores = df.loc[df.groupby("Model")["Validation F1-score"].idxmax()]

    # Plot the best Validation F1-score for each model
    plt.figure(figsize=(10, 6))
    plt.barh(best_scores['Model'], best_scores['Validation F1-score'], color='skyblue')
    plt.xlabel("Best Validation F1-score")
    plt.title("Comparison of Best Validation F1-scores for Each Model")
    plt.gca().invert_yaxis()  # Invert y-axis for better readability
    for index, value in enumerate(best_scores['Validation F1-score']):
        plt.text(round(value,2), index, f"{value:.4f}", va='center')  # Display value with 4 decimal precision

    plt.tight_layout()
    plt.savefig("scripts/finetuning/models_best_f1_valid.png")
    plt.show()

    # Plot the Validation Loss corresponding to the best Validation F1-score for each model
    plt.figure(figsize=(10, 6))
    plt.barh(best_scores['Model'], best_scores['Validation Loss'], color='salmon')
    plt.xlabel("Validation Loss at Best F1-score")
    plt.title("Validation Loss at Best Validation F1-score for Each Model")
    plt.gca().invert_yaxis()  # Invert y-axis for better readability
    for index, value in enumerate(best_scores['Validation Loss']):
        plt.text(round(value,2), index, f"{value:.4f}", va='center')  # Display value with 4 decimal precision
    plt.tight_layout()
    plt.savefig("scripts/finetuning/models_best_loss_valid.png")
    plt.show()
    
if __name__ == "__main__":
    main()