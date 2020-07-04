import os.path
import datetime

import pandas as pd

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.csv import factory as csv_exporter
from pm4py.objects.log.importer.csv import factory as csv_importer

from sklearn.preprocessing import OneHotEncoder

TIMESTAMP_COLUMN_NAME = 'time:timestamp'

# transform log into pandas dataframe, by writing it out as csv and reading it with pandas
def load(file_path):
    csv_path = file_path + ".csv"
    if not os.path.isfile(csv_path):
        log = xes_importer.apply(file_path)
        csv_exporter.export(log, csv_path)
    df = pd.read_csv(csv_path)
    #  transform the timestamp str to datetime object
    df[TIMESTAMP_COLUMN_NAME] = df[TIMESTAMP_COLUMN_NAME].apply(lambda x: datetime.datetime.fromisoformat(x))
    return df

def drop_columns(df):
    keep = ['concept:name']
    column_names = df.columns.tolist()
    for name in keep:
        column_names.remove(name)
    return df.drop(column_names, axis=1, inplace=False)

### analyze df
def analyze(df):
    for name in df.columns:
        print("column name: " + name)
        print("data type: " + str(type(df[name][0])))
        print("unqiue values: " + str(len(df[name].unique())))
        print(df[name][:10].values)
        print('\n')
def one_hot_encode(df, column, none_replacement):
    enc = OneHotEncoder(handle_unknown='ignore')
    df[column].fillna(none_replacement, inplace=True)
    return pd.DataFrame(enc.fit_transform(df[[column]]).toarray(), columns = enc.get_feature_names([column]))