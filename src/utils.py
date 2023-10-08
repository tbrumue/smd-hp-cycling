import os 
import pandas as pd

def create_heatmap_data(df:pd.DataFrame, column:str):
    '''
        Creates a heatmap data from a dataframe with a datetime index
        Args: 
            df: (dataframe) dataframe with datetime index
            column: (string) column name with Energy values 
        Returns:
            data: (dataframe) dataframe with time as index and date as columns
    '''
    assert column in df.columns, f"create_heatmap_data(): Column {column} not in dataframe"
    data_df = df.copy()
    data_df["date"] = df.index.date
    data_df["time"] = df.index.time
    # Using only the first datapoint instead of summing avoids unrealistically height demand 
    get_first = lambda x: x.iloc[0] 
    # Pivot dates and times to create a two dimensional representation
    data = data_df.pivot_table(index='time', columns='date', values=column, aggfunc=get_first, dropna=False) 
    return data

def mkdir(path):
    '''
        Checks for existence of a path and creates it if necessary: 
        Args: 
            path: (string) path to folder
    '''
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError as e:
            if not os.path.exists(path):
                raise