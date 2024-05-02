import pandas as pd
import os


df_new = pd.DataFrame(columns=['pmid', 'journal_name', 'title', 'abstract', 'accepted_label'])
data_dir = '../../data/additional_data'

for filename in os.listdir(data_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(data_dir, filename)
        approved_label = filename.split('.txt')[0].strip()
        
        pmids = []
        titles = []
        abstracts = []
        journal_names = []
        approved_labels = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                if line.startswith('AB  -'):
                    abstracts.append(line.split('AB  - ')[1].strip())
                elif line.startswith('AN  -'):
                    pmids.append(line.split('AN  - ')[1].strip())
                elif line.startswith('T2  -'):
                    journal_names.append(line.split('T2  - ')[1].strip())
                elif line.startswith('TI  -'):
                    titles.append(line.split('TI  - ')[1].strip())
                    approved_labels.append(approved_label)

        df_add = pd.DataFrame({
            'pmid': pmids,
            'journal_name': journal_names,
            'title': titles,
            'abstract': abstracts,
            'accepted_label': approved_labels
        })

        df_new = pd.concat([df_new, df_add], ignore_index=True)

print('new data')
print(df_new.info())

print('\n\nold data')
df_old = pd.read_csv('../../data/prodigy/annotated_output/final/full_combined_dataset_1996.csv')
df_old = df_old.drop(columns=['Unnamed: 0'])
print(df_old.info())

print('combined data')
df_combined = pd.concat([df_old, df_new], ignore_index=True)
print(df_combined.info())
print(df_combined)

print(f'total journal names in the new dataset: {df_new.journal_name.nunique()}')
print(f'total journal names in the old dataset: {df_old.journal_name.nunique()}')
print(f'total journal names in the combined dataset: {df_combined.journal_name.nunique()}')

print(f'journal names in the new dataset: {df_new.journal_name.unique()}')
print(f'journal names in the old dataset: {df_old.journal_name.unique()}')

# save the final version of the enriched dataset
df_combined.to_csv('../../data/prodigy/annotated_output/final/full_enriched_dataset_2696.csv')
