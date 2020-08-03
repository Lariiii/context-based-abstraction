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

def load(file_path):
    csv_path = file_path + ".csv"
    if not os.path.isfile(csv_path):
        log = xes_importer.apply(file_path)
        csv_exporter.export(log, csv_path)
    return pd.read_csv(csv_path)

def preprocess(df, column_name_map):
    #  transform the timestamp str to datetime object
    df[column_name_map['timestamp']] = pd.to_datetime(df[column_name_map['timestamp']])
    # sort events by timestamp and fix index after sorting
    return df.sort_values(by=[column_name_map['timestamp']]).reset_index(drop=True)

### analyze df
def analyze(df, show_examples=False):
    for name in df.columns:
        print("column name: " + name)
        print("data type: " + str(type(df[name][0])))
        print("unique values: " + str(len(df[name].unique())))
        if show_examples:
            print(df[name][:10].values)
        print('\n')
    if 'case:id' in df.columns:
        case_lengths = df.groupby(['case:id'])['case:id'].agg(['size'])
        print("min case length: " + str(case_lengths.min()[0]))
        print("max case length: " + str(case_lengths.max()[0]))
        print("mean case length: " + str(case_lengths.mean()[0]))

### feature generation
def add_timestamp_features(df, column_name_map):
    timestamp_column = column_name_map['timestamp']
    min_date =  df[timestamp_column].min()
    df['feature_day_nr'] =  df[timestamp_column].apply(lambda x: (x - min_date).days)
    df['feature_weekday'] = df[timestamp_column].apply(lambda x: x.weekday())
    df['feature_hour'] =  df[timestamp_column].apply(lambda x: x.hour)
    df['feature_time_00-06'] = df[timestamp_column].apply(lambda x: 1 if x.hour <= 6 else 0)
    df['feature_time_07-12'] = df[timestamp_column].apply(lambda x: 1 if 7 <= x.hour <= 12 else 0)
    df['feature_time_13-18'] = df[timestamp_column].apply(lambda x: 1 if 13 <= x.hour <= 18 else 0)
    df['feature_time_19-24'] = df[timestamp_column].apply(lambda x: 1 if 19 <= x.hour <= 24 else 0)

def add_event_type_representative(df, column_name_map):
    representative_dict = dict()
    unique_event_types = df[column_name_map['eventname']].unique()
    amount_of_event_types = len(unique_event_types)
    alphabet_len = len(ascii_uppercase)
    repetitions = (float(amount_of_event_types) / float(alphabet_len)) + 1
    representatives = [''.join(i) for i in product(ascii_uppercase, repeat = int(repetitions))]
    for idx, type in enumerate(unique_event_types):
        representative_dict[type] = representatives[idx]
    df[column_name_map['eventnamerepresentative']] = df[column_name_map['eventname']].map(representative_dict)

def add_event_ref(df, distance, column_name_map):
    timestamp_column = column_name_map['timestamp']
    caseid_column = column_name_map['caseid']
    eventname_column = column_name_map['eventname']
    
    ref_column = 'event_ref_' + str(distance)
    time_column = 'event_ref_time_' + str(distance)

    time_filler_max = df[timestamp_column].max() + timedelta(days=1)
    time_filler_min = df[timestamp_column].min() - timedelta(days=1)

    print("Nr of cases " + str(len(df[caseid_column].unique())))
    print(datetime.now())
    counter = 1
    for case in df[caseid_column].unique():
        sys.stdout.write("\r" + str(counter))
        sys.stdout.flush()
        counter += 1
        case_rows = df.loc[df[caseid_column] == case]
        # add event reference
        df.loc[df[caseid_column] == case, ref_column] = case_rows[eventname_column].shift(+(-distance))
        time_replacement_df = pd.DataFrame(case_rows[timestamp_column].shift(+(-distance)))
        if distance > 0:
            time_replacement_df.loc[time_replacement_df[timestamp_column].isnull()] = time_filler_min  
            df.loc[df[caseid_column] == case, time_column] = (time_replacement_df[timestamp_column] - case_rows[timestamp_column]).map(lambda x: x.total_seconds())

        if distance < 0:
            time_replacement_df.loc[time_replacement_df[timestamp_column].isnull()] = time_filler_max
            df.loc[df[caseid_column] == case, time_column] = (case_rows[timestamp_column] -  time_replacement_df[timestamp_column]).map(lambda x: x.total_seconds())
    df.loc[df[time_column] < 0, time_column] = -999999

def determine_event_position(df):
    df['feature_pos_beginning'] = 0
    df['feature_pos_middle'] = 0
    df['feature_pos_end'] = 0
    df.loc[pd.isnull(df['event_ref_-1']), 'feature_pos_beginning'] = 1
    df.loc[pd.isnull(df['event_ref_1']), 'feature_pos_end'] = 1
    df.loc[(df['feature_pos_beginning'] == 0) & (df['feature_pos_end'] == 0), 'feature_pos_middle'] = 1


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