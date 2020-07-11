import os.path
import datetime

from itertools import product
from string import ascii_uppercase

import pandas as pd

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.csv import factory as csv_exporter
from pm4py.objects.log.importer.csv import factory as csv_importer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

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
    return df

def add_event_reference(df, nr_of_prev_events=0, nr_of_next_events=0, time_since_prev=False, time_till_next=False, case_id_column=CASE_ID_COLUMN_NAME, reference_column=CONCEPT_NAME_COLUMN_REPRESENTATIVE, timestamp_column=TIMESTAMP_COLUMN_NAME, start_filler='start', end_filler='end'):
    col_ref_prev_prefix = 'feature_previous_-'
    col_time_prev_prefix = 'feature_time_since_-'

    col_ref_next_prefix = 'feature_next_+'
    col_time_next_prefix = 'feature_time_till_+'

    if nr_of_prev_events > 0 or nr_of_next_events > 0:
        for case in df[case_id_column].unique():
            case_rows = df.loc[df[case_id_column] == case]
            # add previous event reference
            if nr_of_prev_events > 0:
                ref_replacement = case_rows[reference_column]
                if time_since_prev:
                    time_replacement = case_rows[timestamp_column]
                    for x in range(1, nr_of_prev_events + 1):
                        # add previous event references
                        ref_replacement = ref_replacement.shift(+x)
                        if (x == 1): 
                            ref_replacement = ref_replacement.fillna(start_filler)
                        else:
                            ref_replacement = ref_replacement.fillna('')
                        df.loc[df[case_id_column] == case, col_ref_prev_prefix + str(x)] = ref_replacement
                        # add time since previous event
                        time_filler = df[timestamp_column].max() + datetime.timedelta(days=1)
                        time_replacement = time_replacement.shift(+x)
                        time_replacement_df = pd.DataFrame(time_replacement)
                        time_replacement_df.loc[time_replacement_df[timestamp_column].isnull()] = time_filler
                        df.loc[df[case_id_column] == case, col_time_prev_prefix + str(x)] = case_rows[timestamp_column] -  time_replacement_df[timestamp_column]

            if nr_of_next_events > 0:
                ref_replacement = case_rows[reference_column]
                if time_since_prev:
                    time_replacement = case_rows[timestamp_column]
                    for x in range(1, nr_of_next_events + 1):
                        # add next event references
                        ref_replacement = ref_replacement.shift(-x)
                        if (x == 1): 
                            ref_replacement = ref_replacement.fillna(end_filler)
                        else:
                            ref_replacement = ref_replacement.fillna('')
                        df.loc[df[case_id_column] == case, col_ref_next_prefix + str(x)] = ref_replacement
                        # add time until next event
                        time_filler = df[timestamp_column].min() + datetime.timedelta(days=1)
                        time_replacement = time_replacement.shift(-x)
                        time_replacement_df = pd.DataFrame(time_replacement)
                        time_replacement_df.loc[time_replacement_df[timestamp_column].isnull()] = time_filler
                        df.loc[df[case_id_column] == case,  col_time_next_prefix + str(x)] = time_replacement_df[timestamp_column] - case_rows[timestamp_column]
    for name in df.columns:
        if 'feature_time' in name:
            df.loc[df[name] < datetime.timedelta(), name] = datetime.timedelta()
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