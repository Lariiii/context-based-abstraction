import os.path

import pandas as pd

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.csv import factory as csv_exporter
from pm4py.objects.log.importer.csv import factory as csv_importer

def load(file_path):
    # transform log into pandas dataframe, by writing it out as csv and reading it with pandas
    csv_path = file_path + ".csv"
    if not os.path.isfile(file_path):
        log = xes_importer.apply(file_path)
        csv_exporter.export(log, csv_path)
    return pd.read_csv(csv_path)

def drop_columns(df):
    keep = ['concept:name']
    column_names = df.columns.tolist()
    for name in keep:
        column_names.remove(name)
    return df.drop(column_names, axis=1, inplace=False)