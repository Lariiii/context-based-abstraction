import pandas as pd

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.csv import factory as csv_exporter
from pm4py.objects.log.importer.csv import factory as csv_importer

from sklearn.feature_extraction.text import TfidfVectorizer

def transform(from_path):
    # transform log into pandas dataframe, by writing it out as csv and reading it with pandas
    log = xes_importer.apply(from_path)
    csv_path = from_path + ".csv"
    csv_exporter.export(log, csv_path)
    return pd.read_csv(csv_path)

def drop_columns(df):
    keep = ['concept:name']
    column_names = df.columns.tolist()
    for name in keep:
        column_names.remove(name)
    return df.drop(column_names, axis=1, inplace=False)

def get_vectors(df):
    tfidf = TfidfVectorizer(
        min_df = 5,
        max_df = 0.95,
        max_features = 8000,
        stop_words = 'english'
    )
    tfidf.fit(df['concept:name'])
    text_matrix = tfidf.transform(df['concept:name'])
    return text_matrix