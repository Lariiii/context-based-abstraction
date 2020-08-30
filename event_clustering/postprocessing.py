import pandas as pd
from itertools import product
from string import ascii_uppercase

# this function for each case drops events that have the same abstracted name as their preceeding event, thereby effectively merging consecutive events 
def merge_consecutive_same_events(abstracted_df, caseid_column, cluster_col):
    for case in abstracted_df[caseid_column].unique():
        case_rows = abstracted_df.loc[abstracted_df[caseid_column] == case]
        abstracted_df.drop(case_rows[case_rows[cluster_col] == case_rows[cluster_col].shift(+1)].index, inplace=True)
    return abstracted_df