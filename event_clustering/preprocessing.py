import os.path
from datetime import datetime, timedelta
import sys

from itertools import product
from string import ascii_uppercase

import pandas as pd
import numpy as np

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.csv import factory as csv_exporter
from pm4py.objects.log.importer.csv import factory as csv_importer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer

TIMESTAMP_COLUMN_NAME = 'time:timestamp'
CASE_ID_COLUMN_NAME = 'case:id'
CONCEPT_NAME_COLUMN = 'concept:name'
CONCEPT_NAME_COLUMN_REPRESENTATIVE = 'concept:name_representative'

# transform log into pandas dataframe, by writing it out as csv and reading it with pandas
def load(file_path):
    csv_path = file_path + ".csv"
    if not os.path.isfile(csv_path):
        log = xes_importer.apply(file_path)
        csv_exporter.export(log, csv_path)
    return pd.read_csv(csv_path)

def preprocess(df):
    #  transform the timestamp str to datetime object
    df[TIMESTAMP_COLUMN_NAME] = pd.to_datetime(df[TIMESTAMP_COLUMN_NAME])
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
    case_lengths = df.groupby(['case:id'])['case:id'].agg(['size'])
    print("min case length: " + str(case_lengths.min()[0]))
    print("max case length: " + str(case_lengths.max()[0]))
    print("mean case length: " + str(case_lengths.mean()[0]))

### feature generation
def add_timestamp_features(df):
    min_date =  df[TIMESTAMP_COLUMN_NAME].min()
    df['feature_day_nr'] =  df[TIMESTAMP_COLUMN_NAME].apply(lambda x: (x - min_date).days)
    df['feature_weekday'] = df[TIMESTAMP_COLUMN_NAME].apply(lambda x: x.weekday())
    df['feature_hour'] =  df[TIMESTAMP_COLUMN_NAME].apply(lambda x: x.hour)

def add_event_type_representative(df, event_type_column=CONCEPT_NAME_COLUMN, representative_column=CONCEPT_NAME_COLUMN_REPRESENTATIVE):
    representative_dict = dict()
    unique_event_types = df[event_type_column].unique()
    amount_of_event_types = len(unique_event_types)
    alphabet_len = len(ascii_uppercase)
    repetitions = (float(amount_of_event_types) / float(alphabet_len)) + 1
    representatives = [''.join(i) for i in product(ascii_uppercase, repeat = int(repetitions))]
    for idx, type in enumerate(unique_event_types):
        representative_dict[type] = representatives[idx]
    df[representative_column] = df[event_type_column].map(representative_dict)

def add_event_ref(df, 
        distance, 
        case_id_column=CASE_ID_COLUMN_NAME, 
        event_name_column=CONCEPT_NAME_COLUMN_REPRESENTATIVE, 
        timestamp_column=TIMESTAMP_COLUMN_NAME,
    ):
    ref_column = 'event_ref_' + str(distance)
    time_column = 'event_ref_time_' + str(distance)

    time_filler_max = df[timestamp_column].max() + timedelta(days=1)
    time_filler_min = df[timestamp_column].min() - timedelta(days=1)

    print("Nr of cases " + str(len(df[case_id_column].unique())))
    print(datetime.now())
    counter = 1
    for case in df[case_id_column].unique():
        sys.stdout.write("\r" + str(counter))
        sys.stdout.flush()
        counter += 1
        case_rows = df.loc[df[case_id_column] == case]
        # add previous event reference
        df.loc[df[case_id_column] == case, ref_column] = case_rows[event_name_column].shift(+(-distance))
        time_replacement_df = pd.DataFrame(case_rows[timestamp_column].shift(+(-distance)))
        if distance > 0:
            time_replacement_df.loc[time_replacement_df[timestamp_column].isnull()] = time_filler_min  
            df.loc[df[case_id_column] == case, time_column] = (time_replacement_df[timestamp_column] - case_rows[timestamp_column]).map(lambda x: x.total_seconds())

        if distance < 0:
            time_replacement_df.loc[time_replacement_df[timestamp_column].isnull()] = time_filler_max
            df.loc[df[case_id_column] == case, time_column] = (case_rows[timestamp_column] -  time_replacement_df[timestamp_column]).map(lambda x: x.total_seconds())
    df.loc[df[time_column] < 0, time_column] = -1

### feature encoding
def one_hot_encode(df, column, none_replacement='none'):
    enc = OneHotEncoder(handle_unknown='ignore')
    df[column].fillna(none_replacement, inplace=True)
    return pd.DataFrame(enc.fit_transform(df[[column]]).toarray(), columns = enc.get_feature_names([column]))

def binning(df, column, n_bins, column_prefix=''):
    enc = KBinsDiscretizer(n_bins=n_bins)
    df_encoded = pd.DataFrame(enc.fit_transform(df[[column]]).toarray())
    df_encoded.columns = [column_prefix + str(col) for col in df_encoded.columns]
    return df_encoded

def tfidf_encode(df, column, vectorizer):
    vectorizer.fit_transform(df[column])
    df_encoded = pd.DataFrame(vectorizer.transform(df[column]).todense())
    df_encoded.columns = [column + '_' + x for x in vectorizer.get_feature_names()]
    return df_encoded