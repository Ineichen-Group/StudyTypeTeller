# PreclinicalAbstractClassification
Objectives: develop methods to distinguish different types of pre-clinical literature based on abstract text.
# 1. Set up the environment
# 2. Data
## Querying PubMed

Given a list of relevant PMIDs ([pmids.txt](data%2Fpubmed%2Fpmids.txt)).
- Create a variable containing the list of PMIDs:

`id_list=$(paste -sd, "pmids.txt") `
- Fetch the relevant contents from PubMed based on those IDs:

`efetch -db pubmed -id $id_list -format xml | xtract -pattern PubmedArticle -tab '^' -def "N/A" -element MedlineCitation/PMID PubDate/Year Journal/Title ArticleTitle AbstractText -block ArticleId -if ArticleId@IdType -equals doi -element ArticleId > "./pmid_contents_v5.txt"
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