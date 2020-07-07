import os.path
import datetime

import pandas as pd

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.csv import factory as csv_exporter
from pm4py.objects.log.importer.csv import factory as csv_importer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

TIMESTAMP_COLUMN_NAME = 'time:timestamp'
CASE_ID_COLUMN_NAME = 'case:id'
CONCEPT_NAME_COLUMN = 'concept:name'

# transform log into pandas dataframe, by writing it out as csv and reading it with pandas
def load(file_path):
    csv_path = file_path + ".csv"
    if not os.path.isfile(csv_path):
        log = xes_importer.apply(file_path)
        csv_exporter.export(log, csv_path)
    return pd.read_csv(csv_path)

def preprocess(df):
    #  transform the timestamp str to datetime object
    df[TIMESTAMP_COLUMN_NAME] = df[TIMESTAMP_COLUMN_NAME].apply(lambda x: datetime.datetime.fromisoformat(x))
    # sort events by timestamp and fix index after sorting
    return df.sort_values(by=[TIMESTAMP_COLUMN_NAME]).reset_index(drop=True)

def drop_columns(df):
    keep = [CONCEPT_NAME_COLUMN]
    column_names = df.columns.tolist()
    for name in keep:
        column_names.remove(name)
    return df.drop(column_names, axis=1, inplace=False)

### analyze df
def analyze(df, show_examples=False):
    for name in df.columns:
        print("column name: " + name)
        print("data type: " + str(type(df[name][0])))
        print("unique values: " + str(len(df[name].unique())))
        if show_examples:
            print(df[name][:10].values)
        print('\n')

### feature generation
def add_timestamp_features(df):
    min_date =  df[TIMESTAMP_COLUMN_NAME].min()
    df['feature_day_nr'] =  df[TIMESTAMP_COLUMN_NAME].apply(lambda x: (x - min_date).days)
    df['feature_weekday'] = df[TIMESTAMP_COLUMN_NAME].apply(lambda x: x.weekday())
    df['feature_hour'] =  df[TIMESTAMP_COLUMN_NAME].apply(lambda x: x.hour)

def add_previous_and_next_event_reference(df, case_id_column=CASE_ID_COLUMN_NAME, reference_column=CONCEPT_NAME_COLUMN, start_filler='start', end_filler='end'):
    if not case_id_column in df:
        return
    for case in df[case_id_column].unique():
        df.loc[df[case_id_column] == case, 'feature_previous_' + reference_column] = df.loc[df[case_id_column] == case][reference_column].shift(+1).fillna(start_filler)
        df.loc[df[case_id_column] == case, 'feature_next_' + reference_column] = df.loc[df[case_id_column] == case][reference_column].shift(-1).fillna(end_filler)
    return df

### feature encoding
def one_hot_encode(df, column, none_replacement='none'):
    enc = OneHotEncoder(handle_unknown='ignore')
    df[column].fillna(none_replacement, inplace=True)
    return pd.DataFrame(enc.fit_transform(df[[column]]).toarray(), columns = enc.get_feature_names([column]))

def tfidf_encode(df, column, vectorizer):
    vectorizer.fit_transform(df[column])
    df_encoded = pd.DataFrame(vectorizer.transform(df[column]).todense())
    df_encoded.columns = [column + '_' + x for x in vectorizer.get_feature_names()]
    return df_encoded