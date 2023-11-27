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

`esearch -db pubmed -query 'Central nervous system diseases[MeSH] OR Mental Disorders Psychiatric illness[MeSH]' | efetch -format uid > ./pubmed/cns_psychiatric_diseases_mesh.txt
`

On 27/11/2023 this query returns 2,788,345 PMIDs, out of which we sample 5000 in [Data_Preparation.ipynb](data%2FData_Preparation.ipynb).

Given this list (see [cns_psychiatric_diseases_mesh_5000_sample_pmids.txt](data%2Fpubmed%2Fcns_psychiatric_diseases_mesh_5000_sample_pmids.txt)).
- Create a variable containing the list of PMIDs:

`id_list=$(paste -sd, "./pubmed/cns_psychiatric_diseases_mesh_5000_sample_pmids.txt")`
- Fetch the relevant contents from PubMed based on those IDs:

`efetch -db pubmed -id $id_list -format xml | xtract -pattern PubmedArticle -tab '^' -def "N/A" -element MedlineCitation/PMID PubDate/Year Journal/Title ArticleTitle AbstractText -block ArticleId -if ArticleId@IdType -equals doi -element ArticleId > "./pubmed/pmid_contents_mesh_query.txt"
`

The data is then cleaned and prepared for annotation with prodigy in [Data_Preparation.ipynb](data%2FData_Preparation.ipynb).

Relevant API documentation references:
- https://www.ncbi.nlm.nih.gov/books/NBK179288/
- https://www.nlm.nih.gov/dataguide/edirect/xtract_formatting.html
- https://dataguide.nlm.nih.gov/classes/edirect-for-pubmed/samplecode4.html#specify-a-placeholder-to-replace-blank-spaces-in-the-output-table

## Data Annotation with Prodigy
The prepared data for prodigy is stored in [input](data%2Fprodigy%2Finput).
A custom recipe was developed to use prodigy for text classification and include keyword highlighting, see [recipe_textcat_patterns.py](data%2Fprodigy%2Frecipe_textcat_patterns.py).

To perform the annotations in prodigy, follow the instructions in [Prodigy_Annotator_Guide.pdf](data%2Fprodigy%2FProdigy_Annotator_Guide.pdf).