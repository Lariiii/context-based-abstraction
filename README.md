# context-based-abstraction
Event Clustering for Context-based Data Abstraction

This repository contains and documents our work in the master's seminar "Data Extraction for Process Mining" at Hasso Plattner Institute in the summer term 2020.

## Project structure
The project is structured into pipelines implemented as jupyter notebooks that process datasets using several modules including function for preprocessing, clustering, postprocessing and evaluation.

### Pipeline notebooks 
**approach_preprocessing.ipynb**
- transforms the dataset from XES to dataframes
- applies the feature generation to the dataset and stores it as separate intermediate csv for the next step
- applies encoding to the features and stores the result as separate intermediate csv for the next step

**approach_evaluation.ipynb**
- loads the encoded csv
- applies clustering (kmeans or hierarchical)
- helps evaluate the result by generating PCA plots, calculating metrics
- generates an abstracted event log based on the clustering results
- generates process models for the original and abstracted event log
- calculates metrics on the process models

**plot_results.ipynb**
- notebook used to generate plots for the final report

### Code modules (stored in /event_clustering)
**preprocessing.py** 
- code for
    - transforming datasets from XES to csv
    - feature generation
    - feature encoding

**clustering.py**
- code snippets used for the actual clustering and evaluation

**postprocessing.py**
- code to replace events with their presentative

**process_mining.py**
- code for 
    - process mining 
    - calculating metrics on the process models

## How to use
- store your dataset in a folder of your choice, recommended: use a /data directory in the project
- make sure you have jupyter notebook installed, ideally via Anaconda
- install the requirements specified in requirements.txt via `pip install -r requirements.txt`
- run the approach_preprocessing.ipynb and then the approach_evaluation.ipynb notebooks in Jupyter, ideally using Anaconda, and modify the notebook according to your dataset and the comments in the notebooks
