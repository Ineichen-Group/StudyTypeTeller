# PreclinicalAbstractClassification
Objectives: develop methods to distinguish different types of pre-clinical literature based on abstract text.
# 1. Set up the environment
The project is build using poetry for dependency management. Instructions on how to install poetry can be found in the [documentation](https://python-poetry.org/docs/).  
To install the defined dependencies for the project, make sure you have the .toml and .lock files and run the _install_ command.
```bib
poetry install
```
The pyproject.toml file contains all relevant packages that need to be installed for running the project. The poetry.lock file is needed to ensure the same version of the installed libraries.

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

The annotation with GPT using different prompting strategies follows a similar structure.
1. We prepare the prompts in a json file and assign a unique ID to each.
2. We read the test set ([6-2-2_all_classes_enriched_with_kw/test.csv](data%2Fdata_splits_stratified%2F6-2-2_all_classes_enriched_with_kw%2Ftest.csv)).
3. For each title and abstract in the test set and for each prompting strategy, we send an
individual GTP query and retrieve the predicted class.
4. We save the predictions in [predictions](models%2Fpredictions).

## Multi-label classification
In this setup we directly want to classify the abstracts into one of the 14 class types.
1. The relevant prompts are in [prompt_strategies.json](models%2Fprompts%2Fprompt_strategies.json).
An example prompt is given below:
```bib
   {
      "id": "P2_1",
      "text": "Determine which of these labels fits the text best: Clinical-study-protocol, Human-systematic-review, Non-systematic-review, Human-RCT-non-drug-intervention, Human-RCT-drug-intervention, Human-RCT-non-intervention, Human-case-report, Human-non-RCT-non-drug-intervention, Human-non-RCT-drug-intervention, Animal-systematic-review, Animal-drug-intervention, Animal-non-drug-intervention, Animal-other, In-vitro-study, Remaining. The classfication applies to the journal name, title, and/or abstract of a study. Respond in json format with the key: gpt_label.",
      "strategy_type": "zero_shot_applies_to"
    }
```

2. The annotation is done in [Annotate_with_GPT_Prompts_multi-label.ipynb](models%2FAnnotate_with_GPT_Prompts_multi-label.ipynb).

## Binary classification
In this setup we want to classify each abstract either as ANIMAL or OTHER.
1. The relevant prompts are in [prompt_strategies_binary.json](models%2Fprompts%2Fprompt_strategies_binary.json).
An example prompt is given below:
```bib
  {
      "id": "P1",
      "text": "Classify this text, choosing one of these labels: 'ANIMAL' if the text is related to animal, and 'OTHER' for any other study type. Respond in json format with the key: gpt_label.",
      "strategy_type": "zero_shot"
    }
```
2. The annotation is done in [Annotate_with_GPT_Prompts_binary.ipynb](models%2FAnnotate_with_GPT_Prompts_binary.ipynb).

## Hierarchical classification
In this setup we want to first classify either as ANIMAL or OTHER. 
Then classify the abstracts within each of these two classes into one of the fine-grained categories within this class. 
1. The relevant prompts are in [prompt_strategies_hierarchical.json](models%2Fprompts%2Fprompt_strategies_hierarchical.json).
An example prompt is given below:
```bib
    {
      "id": "P1_HIERARCHY",
      "text_animal": "Classify this text, choosing one of these labels: Animal-systematic-review, Animal-drug-intervention, Animal-non-drug-intervention, Animal-other. Respond in json format with the key: gpt_label.",
      "text_other": "Classify this text, choosing one of these labels: Clinical-study-protocol, Human-systematic-review, Non-systematic-review, Human-RCT-non-drug-intervention, Human-RCT-drug-intervention, Human-RCT-non-intervention, Human-case-report, Human-non-RCT-non-drug-intervention, Human-non-RCT-drug-intervention, In-vitro-study, Remaining. Respond in json format with the key: gpt_label.",
      "strategy_type": "zero_shot"
    },
```

2. The annotation is done in [Annotate_with_GPT_Prompts_hierarchical.ipynb](models%2FAnnotate_with_GPT_Prompts_hierarchical.ipynb).


## GPT Results Evaluation
