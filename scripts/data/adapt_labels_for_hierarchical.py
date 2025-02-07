import os
import pandas as pd


def remap_labels(df, classification_type):
    if classification_type == 'animal':
        df = df[df['multi_label'].isin([5, 6, 10])]
        df['animal_label'] = df['multi_label'].map({5: 0, 6: 1, 10: 2})
    elif classification_type == 'other':
        df = df[df['multi_label'].isin([0, 1, 2, 3, 4, 7, 8, 9, 11, 12, 13])]
        df['other_label'] = df['multi_label'].map({
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 7: 5, 8: 6, 9: 7, 11: 8, 12: 9, 13: 10
        })
    return df

def save_remapped_data(data_dir, save_dir, classification_type):

    os.makedirs(save_dir, exist_ok=True)

    # Load datasets
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    # Remap labels
    train_df = remap_labels(train_df, classification_type)
    val_df = remap_labels(val_df, classification_type)
    test_df = remap_labels(test_df, classification_type)

    # Save to new CSV files
    train_df.to_csv(os.path.join(save_dir, f'train_{classification_type}.csv'), index=False)
    val_df.to_csv(os.path.join(save_dir, f'val_{classification_type}.csv'), index=False)
    test_df.to_csv(os.path.join(save_dir, f'test_{classification_type}.csv'), index=False)


if __name__ == "__main__":
    
    data_dir = "./../../data/data_splits_stratified/6-2-2_all_classes_enriched_with_kw"
    save_dir = "./../../data/data_splits_stratified/6-2-2_all_classes_enriched_with_kw_hierarchical"

    classification_types = ['animal', 'other']
    for classification_type in classification_types:
        save_remapped_data(data_dir, save_dir, classification_type)
