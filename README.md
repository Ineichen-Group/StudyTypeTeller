# PreclinicalAbstractClassification
Objectives: develop methods to distinguish different types of pre-clinical literature based on abstract text.
# 1. Set up the environment
## Poetry
The project is build using poetry for dependency management. Instructions on how to install poetry can be found in the [documentation](https://python-poetry.org/docs/).  
To install the defined dependencies for the project, make sure you have the .toml and .lock files and run the _install_ command.
```bib
poetry install
```
The pyproject.toml file contains all relevant packages that need to be installed for running the project. The poetry.lock file is needed to ensure the same version of the installed libraries.
## Conda
For the GPT Jupyter Notebooks, we used a conda environment that can be re-created
as follows:
```bib
conda env create -f conda_environment.yml

conda activate studytype-teller
```
Please follow [this docu](https://saturncloud.io/blog/how-to-use-conda-environment-in-a-jupyter-notebook/) to make it accessible in the Notebooks environment.

# 2. Data
## Querying PubMed

To obtain the initial set of relevant PMIDs, the database was queried using a generic search string related to CNS and Psychiatric conditions, as follows:
y
`esearch -db pubmed -query 'Central nervous system diseases[MeSH] OR Mental Disorders Psychiatric illness[MeSH]' | efetch -format uid > ./pubmed/cns_psychiatric_diseases_mesh.txt
`

On 27/11/2023 this query returns 2,788,345 PMIDs, out of which we sample 5000 in [Data_Preparation_and_Postprocessing.ipynb](data%2FData_Preparation_and_Postprocessing.ipynb).

Given this list (see [cns_psychiatric_diseases_mesh_5000_sample_pmids.txt](data%2Fpubmed%2Fcns_psychiatric_diseases_mesh_5000_sample_pmids.txt)).
- Create a variable containing the list of PMIDs:

`id_list=$(paste -sd, "./pubmed/cns_psychiatric_diseases_mesh_5000_sample_pmids.txt")`
- Fetch the relevant contents from PubMed based on those IDs:

`efetch -db pubmed -id $id_list -format xml | xtract -pattern PubmedArticle -tab '^' -def "N/A" -element MedlineCitation/PMID PubDate/Year Journal/Title ArticleTitle AbstractText -block ArticleId -if ArticleId@IdType -equals doi -element ArticleId > "./pubmed/pmid_contents_mesh_query.txt"
`
- With keywords

`efetch -db pubmed -id $id_list -format xml | xtract -pattern PubmedArticle -tab '^' -def "N/A" -element MedlineCitation/PMID Journal/Title  -block KeywordList -element Keyword > "./pubmed/enriched_data_keywords.txt"
`

The data is then cleaned and prepared for annotation with prodigy in [Data_Preparation_and_Postprocessing.ipynb](data%2FData_Preparation_and_Postprocessing.ipynb).

Relevant API documentation references:
- https://www.ncbi.nlm.nih.gov/books/NBK179288/
- https://www.nlm.nih.gov/dataguide/edirect/xtract_formatting.html
- https://dataguide.nlm.nih.gov/classes/edirect-for-pubmed/samplecode4.html#specify-a-placeholder-to-replace-blank-spaces-in-the-output-table

## Data Annotation with Prodigy
The prepared data for prodigy is stored in [input](data%2Fprodigy%2Finput).
A custom recipe was developed to use prodigy for text classification and include keyword highlighting, see [recipe_textcat_patterns.py](data%2Fprodigy%2Frecipe_textcat_patterns.py).

To perform the annotations in prodigy, follow the instructions in [Prodigy_Annotator_Guide.pdf](data%2Fprodigy%2FProdigy_Annotator_Guide.pdf).

## Data Splits
We split the full dataset into train-validation-test set with a  0.6-0.2-0.2 ratio, resulting in subcorpora of  1851, 530 and 534 samples, respectively.
To ensure that all classes are present across all parts of the split, we employed a customized stratification strategy which ensures that all classes are present across all dataset splits.
- Code for data splits generation is in [generate_stratified_splits.py](scripts%2Fdata%2Fgenerate_stratified_splits.py)
- Datasets are in [data_splits_stratified](data%2Fdata_splits_stratified).

# 3. Annotation with GPT

The annotation with GPT using different prompting strategies follows the steps:
1. We prepare the prompts in a json file and assign a unique ID to each.
2. We read the test set ([6-2-2_all_classes_enriched_with_kw/test.csv](data%2Fdata_splits_stratified%2F6-2-2_all_classes_enriched_with_kw%2Ftest.csv)).
3. For each title, abstract, and eventually keywords (kw) in the test set and for each prompting strategy, we send an
individual GTP query and retrieve the predicted class.
4. We save the predictions in [predictions](models%2Fgpt%2Fpredictions).

## Multi-label classification
In this setup we directly want to classify the abstracts into one of the 14 class types.
1. The relevant prompts are in [prompt_strategies.json](models%2Fgpt%2Fprompts%2Fprompt_strategies.json).
An example prompt is given below:
```bib
   {
      "id": "P2_1",
      "text": "Determine which of these labels fits the text best: Clinical-study-protocol, Human-systematic-review, Non-systematic-review, Human-RCT-non-drug-intervention, Human-RCT-drug-intervention, Human-RCT-non-intervention, Human-case-report, Human-non-RCT-non-drug-intervention, Human-non-RCT-drug-intervention, Animal-systematic-review, Animal-drug-intervention, Animal-non-drug-intervention, Animal-other, In-vitro-study, Remaining. The classfication applies to the journal name, title, and/or abstract of a study. Respond in json format with the key: gpt_label.",
      "strategy_type": "zero_shot_applies_to"
    }
```

2. The annotation is done in [Annotate_with_GPT_Prompts_multi-label.ipynb](models%2Fgpt%2FAnnotate_with_GPT_Prompts_multi-label.ipynb).

## Binary classification
In this setup we want to classify each abstract either as ANIMAL or OTHER.
1. The relevant prompts are in [prompt_strategies_binary.json](models%2Fgpt%2Fprompts%2Fprompt_strategies_binary.json).
An example prompt is given below:
```bib
  {
      "id": "P1",
      "text": "Classify this text, choosing one of these labels: 'ANIMAL' if the text is related to animal, and 'OTHER' for any other study type. Respond in json format with the key: gpt_label.",
      "strategy_type": "zero_shot"
    }
```
2. The annotation is done in [Annotate_with_GPT_Prompts_binary.ipynb](models%2Fgpt%2FAnnotate_with_GPT_Prompts_binary.ipynb).

## Hierarchical classification
In this setup we want to first classify either as ANIMAL or OTHER. 
Then classify the abstracts within each of these two classes into one of the fine-grained categories within this class. 
1. The relevant prompts are in [prompt_strategies_hierarchical.json](models%2Fgpt%2Fprompts%2Fprompt_strategies_hierarchical.json).
An example prompt is given below:
```bib
    {
      "id": "P1_HIERARCHY",
      "text_animal": "Classify this text, choosing one of these labels: Animal-systematic-review, Animal-drug-intervention, Animal-non-drug-intervention, Animal-other. Respond in json format with the key: gpt_label.",
      "text_other": "Classify this text, choosing one of these labels: Clinical-study-protocol, Human-systematic-review, Non-systematic-review, Human-RCT-non-drug-intervention, Human-RCT-drug-intervention, Human-RCT-non-intervention, Human-case-report, Human-non-RCT-non-drug-intervention, Human-non-RCT-drug-intervention, In-vitro-study, Remaining. Respond in json format with the key: gpt_label.",
      "strategy_type": "zero_shot"
    },
```

2. The annotation is done in [Annotate_with_GPT_Prompts_hierarchical.ipynb](models%2Fgpt%2FAnnotate_with_GPT_Prompts_hierarchical.ipynb).

## GPT Results Evaluation
The evaluation of GPT predictions follows the steps:
1. We read the predictions obtained from GPT.
2. We map the predictions to their numerical representation, as well as the target annotated columns.
The mapping includes a fuzzy matching of the GPT outputs to our target labels to take into account the different
spelling variations that GPT sometimes produces.
3. We evaluate the prompts, including obtaining a classification report and a confusion matrix between predicted 
and target labels for each prompting strategy. The confidence interval calculations are from the local package
in [scripts/confidenceinterval](scripts%2Fconfidenceinterval).
4. We map the resulting dataframes to LaTeX tables that we can directly report in our paper.

### Evaluation Notebooks
1. Multi-label: tbd
2. Binary: [Evaluation_GPT_binary_with_CI.ipynb](models%2Fgpt%2FEvaluation_GPT_binary_with_CI.ipynb)
3. Hierarchical: [Evaluation_GPT_hierarchical_with_CI.ipynb](models%2Fgpt%2FEvaluation_GPT_hierarchical_with_CI.ipynb)

# 4. Annotation with BERT

We experimented with the following models from the HuggingFace library:
```bib
    models_to_fine_tune = [
                            'bert-base-uncased',
                            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                            'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
                            'allenai/scibert_scivocab_uncased',
                            'dmis-lab/biobert-v1.1',
                            'michiyasunaga/BioLinkBERT-base',
                            'emilyalsentzer/Bio_ClinicalBERT',
                            ]
```

## Hyperparameter optimization
We used the library Weights&Biases and its [Sweeps functionality](https://docs.wandb.ai/guides/sweeps) to automate and
visualize hyperparameter search. The sweep configuration was as follows:
```bib
    SWEEP_CONFIG = {
        'method': 'bayes',
        'metric': {'name': 'val_loss', 'goal': 'minimize'},
        'parameters': {
            'learning_rate': {'min': 1e-5, 'max': 1e-3},
            'batch_size': {'values': [8, 16, 32]},
            "weight_decay": {"min": 1e-5, "max": 1e-3}
    }
}
```

The code for that is in [hyperparam_optimization_binary.py](scripts%2Fhyperparam_optimization%2Fhyperparam_optimization_binary.py)
and [hyperparam_optimization_milti.py](scripts%2Fhyperparam_optimization%2Fhyperparam_optimization_milti.py).

## Models fine-tuning
The best hyperparameters were then used to fine-tune the models.
The fine-tuning code is in [finetune.py](scripts%2Ffinetuning%2Ffinetune.py).

The log outputs from that process were saved in [models/transformers/checkpoints](models%2Ftransformers%2Fcheckpoints).

## Models Evaluation

The evaluation scripts for BERT can be found in [evaluation](scripts%2Fevaluation).

The notebook [performance_w_CI.ipynb](scripts%2Fevaluation%2Fperformance_w_CI.ipynb)
contains the code to evaluate all BERT models. It also produces the confusion matrix and comparison plots of 
the best-performing GPT and BERT model.