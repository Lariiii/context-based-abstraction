import numpy as np
import pandas as pd

def start_end_events(log, start_name = 'start', end_name = 'end'):
    sequences = list()
    for trace in log:
        sequence = list()
        sequence.append(start_name)
        for event in trace:
            sequence.append(event)
        sequence.append(end_name)
        sequences.append(sequence)
        
    return sequences

def dataframe_init(log, next_col = 'next', previous_col = 'previous'):
    log_keys = dict(log[0][0]).keys()
    df_init = dict()
    for col in log_keys:
        df_init[col] = []

    df_init[next_col] = []
    df_init[previous_col] = []
    
    return df_init

def merge_df_list(df_list):
    return pd.concat(df_list, ignore_index = True)

def extract_next_previous(df_init, sequences, value, start_name = 'start', end_name = 'end'):
    next_column_name = 'next'
    previous_column_name = 'previous'
    df_list = list()
    
    for seq in sequences:
        dataframe = pd.DataFrame(df_init)
        
        # iterate through sequence and extract all previous and current fields
        for previous, current in zip(seq, seq[1:]):
            new_row = dict()
            if previous == start_name and current != end_name:
                new_row[previous_column_name] = start_name
            elif current != end_name:
                new_row[previous_column_name] = previous[value]
            if current != start_name and current != end_name:
                for field in current:
                    new_row[field] = current[field]
            dataframe = dataframe.append(new_row, ignore_index=True)
            
        # iterate through sequence and extract all next values
        next_values = list()
        for curr, next_event in zip(seq, seq[1:]):
            if curr != end_name:
                if next_event == end_name:
                    next_values.append(next_event)
                else:
                    next_values.append(next_event[value])
                    
        # add a NaN value to the end of the list, so that the second element is the first next element
        next_values.append(np.nan)
        dataframe[next_column_name] = next_values[1:]
        
        # drop rows that only contain NaN
        dataframe = dataframe.dropna()
        df_list.append(dataframe)
    
    final_df = merge_df_list(df_list)
    
    return final_df