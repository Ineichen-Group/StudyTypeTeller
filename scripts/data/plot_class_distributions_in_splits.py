import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_class_distribution(train_data, val_data, test_data, label_column, save_dir):
    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot train data
    train_counts = train_data[label_column].value_counts()
    train_counts.plot(kind='bar', ax=axs[0])
    for i, count in enumerate(train_counts):
        axs[0].text(i, count + 10, str(count), ha='center', va='bottom')
    axs[0].set_title('Train Data Class Distribution')
    axs[0].set_xlabel('Class')
    axs[0].set_ylabel('Count')

    # Plot val data
    val_counts = val_data[label_column].value_counts()
    val_counts.plot(kind='bar', ax=axs[1])
    for i, count in enumerate(val_counts):
        axs[1].text(i, count + 10, str(count), ha='center', va='bottom')
    axs[1].set_title('Val Data Class Distribution')
    axs[1].set_xlabel('Class')
    axs[1].set_ylabel('Count')

    # Plot test data
    test_counts = test_data[label_column].value_counts()
    test_counts.plot(kind='bar', ax=axs[2])
    for i, count in enumerate(test_counts):
        axs[2].text(i, count + 10, str(count), ha='center', va='bottom')
    axs[2].set_title('Test Data Class Distribution')
    axs[2].set_xlabel('Class')
    axs[2].set_ylabel('Count')
    plt.tight_layout()

    # save plots
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/class_distribution_{label_column}.png")

    plt.show()


def main():
    # Path to train, val, and test data files
    filepath = './../../data/data_splits_stratified/6-2-2_all_classes_enriched_with_kw/'
    train_file = f"{filepath}train.csv"
    val_file = f"{filepath}val.csv"
    test_file = f"{filepath}test.csv"
    save_dir = f"{filepath}plots"

    # Read train, val, and test data
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    test_data = pd.read_csv(test_file)

    # Plot class distribution for binary_label
    plot_class_distribution(
                            train_data, 
                            val_data, 
                            test_data, 
                            'binary_label', 
                            save_dir)

    # Plot class distribution for multi_label
    plot_class_distribution(
                            train_data, 
                            val_data, 
                            test_data, 
                            # 'multi_label', 
                            'accepted_label',
                            save_dir)

if __name__ == "__main__":
    main()
