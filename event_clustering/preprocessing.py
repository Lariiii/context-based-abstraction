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
def analyze(df, column_name_map, show_examples=False, include_casetime=False):
    timestamp_column = column_name_map['timestamp']
    caseid_column = column_name_map['caseid']

    for name in df.columns:
        print("column name: " + name)
        print("data type: " + str(type(df[name][0])))
        print("unique values: " + str(len(df[name].unique())))
        if show_examples:
            print(df[name][:10].values)
        print('\n')
    if caseid_column in df.columns:
        case_groups = df.groupby(['case:id'])
        case_lengths = case_groups['case:id'].agg(['size'])
        print("min case length: " + str(case_lengths.min()[0]))
        print("max case length: " + str(case_lengths.max()[0]))
        print("mean case length: " + str(case_lengths.mean()[0]))

        if include_casetime:
            min_case_time = sys.maxsize
            max_case_time = 0
            case_time_total = 0
            case_names = df[caseid_column].unique()

            for case in case_names:
                case_event_times = df.loc[df[caseid_column] == case, timestamp_column]
                case_duration = (case_event_times.max() - case_event_times.min()).total_seconds()
                if case_duration > max_case_time:
                    max_case_time = case_duration
                elif case_duration < min_case_time:
                    min_case_time = case_duration
                case_time_total += case_duration
            print('\n')
            print("min case duration seconds: " + str(min_case_time))
            print("min case duration hours: " + str(min_case_time / 3600))
            print("max case duration seconds: " + str(max_case_time))
            print("max case duration hours: " + str(max_case_time / 3600))
            print("mean case duration seconds: " + str(case_time_total / len(case_names)))
            print("mean case duration: hours " + str((case_time_total / len(case_names) / 3600)))

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

def add_neighbor_event(df, distance, column_name_map):
    timestamp_column = column_name_map['timestamp']
    caseid_column = column_name_map['caseid']
    eventname_column = column_name_map['eventname']
    
    neighbor_event_column = 'feature_neighbor_event_' + str(distance)
    timedif_neighbor_event_column = 'feature_timedif_neighbor_event_' + str(distance)

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
        # add event neighbor
        df.loc[df[caseid_column] == case, neighbor_event_column] = case_rows[eventname_column].shift(+(-distance))
        time_replacement_df = pd.DataFrame(case_rows[timestamp_column].shift(+(-distance)))
        if distance > 0:
            time_replacement_df.loc[time_replacement_df[timestamp_column].isnull()] = time_filler_min  
            df.loc[df[caseid_column] == case, timedif_neighbor_event_column] = (time_replacement_df[timestamp_column] - case_rows[timestamp_column]).map(lambda x: x.total_seconds())

        if distance < 0:
            time_replacement_df.loc[time_replacement_df[timestamp_column].isnull()] = time_filler_max
            df.loc[df[caseid_column] == case, timedif_neighbor_event_column] = (case_rows[timestamp_column] -  time_replacement_df[timestamp_column]).map(lambda x: x.total_seconds())
    df.loc[df[timedif_neighbor_event_column] < 0, timedif_neighbor_event_column] = -999999

def determine_start_end_event_feature(df, column_name_map):
    df['feature_position_beginning'] = 0
    df['feature_position_middle'] = 0
    df['feature_position_end'] = 0
    df.loc[pd.isnull(df['feature_neighbor_event_-1']), 'feature_position_beginning'] = 1
    df.loc[pd.isnull(df['feature_neighbor_event_1']), 'feature_position_end'] = 1
    df.loc[(df['feature_position_beginning'] == 0) & (df['feature_position_end'] == 0), 'feature_position_middle'] = 1

def determine_time_frame_feature(df, column_name_map, start_window_length, end_window_length):
    timestamp_column = column_name_map['timestamp']
    caseid_column = column_name_map['caseid']

    print("Nr of cases " + str(len(df[caseid_column].unique())))
    print(datetime.now())
    counter = 1
    for case in df[caseid_column].unique():
        sys.stdout.write("\r" + str(counter))
        sys.stdout.flush()
        counter += 1

        case_times = df.loc[df[caseid_column] == case, timestamp_column]
        case_start = case_times.min()
        case_end = case_times.max()
        case_rows = df.loc[df[caseid_column] == case]

        df.loc[df[caseid_column] == case, 'feature_window_start'] = case_rows[timestamp_column].map(lambda x: 1 if (x - case_start).total_seconds() < start_window_length else 0)
        df.loc[df[caseid_column] == case, 'feature_window_end'] = case_rows[timestamp_column].map(lambda x: 1 if (case_end - x).total_seconds() < end_window_length else 0)
        
        case_rows = df.loc[df[caseid_column] == case]
        
### feature encoding
def filter_column_names(df, prefix):
    return [column for column in df.columns if prefix in column]

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