# context-based-abstraction
Event Clustering for Context-based Data Abstraction

This repository contains and documents our work in the master's seminar "Data Extraction for Process Mining" at Hasso Plattner Institute in the summer term 2020.

## Project structure
The project is structured into pipelines implemented as jupyter notebooks that process datasets using several modules including functions for preprocessing, clustering, postprocessing and evaluation.

### Pipeline notebooks 
**approach_preprocessing.ipynb**
- transforms the dataset from XES to dataframes
- applies the feature generation to the dataset and stores it as separate intermediate csv for the next step
- applies encoding to the features and stores the result as separate intermediate csv for the next step

**approach_evaluation.ipynb**
- loads the encoded csv
- lets you set up the experiments you would like to run
- lets you choose the amount of clusters by using the elbow method
- for each experiment
    - runs the clustering (kmeans)
    - calculates and stores the cluster metrics
    - generates an abstracted event log based on the clustering results
    - generates process models for the original and abstracted event log
    - calculates and stores metrics on the process models

### Code modules (stored in /event_clustering)
**preprocessing.py** 
- code for
    - transforming datasets from XES to csv
    - feature generation
    - feature encoding

**clustering.py**
- code snippets used for the actual clustering and evaluation

**postprocessing.py**
- code to merge consecutive events that have the same event name

**process_mining.py**
- code for loading the event logs and for visualizing the process models

## How to use
- store your dataset in a folder of your choice, recommended: use a "/data" directory in the project
- create an empty "/results" folder in your repository
- make sure you have jupyter notebook installed, ideally via Anaconda
- install the requirements specified in requirements.txt via `pip install -r requirements.txt`
- run the approach_preprocessing.ipynb and then the approach_evaluation.ipynb notebooks in Jupyter, ideally using Anaconda, and configure the notebook according to your dataset
