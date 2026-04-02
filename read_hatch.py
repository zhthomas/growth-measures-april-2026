import pandas as pd
import numpy as np

def read_hatch(filepath, min_length=10,
               remove_decline=False,
               remove_zero_timeseries=False,
               drop_const=True,
               drop_x_const=0,
               remove_formative = True):
    
    df = pd.read_csv(filepath, index_col=0)
    df.replace(0, np.nan, inplace=True)

    all_years = pd.to_numeric(df.columns, errors='coerce').dropna().astype(int)
    years_dict = {str(i): i for i in all_years}
    df = df.rename(columns=years_dict)

    if remove_decline:
        #apply function below to each row to change all values after max year to nan
        df = df.apply(replace_after_max, axis=1)
        # todo: remove only values that fall below a certain percentage of the max

    if min_length > 0:
        #remove rows with less than 10 years
        print(f"Removed timeseries with less than {min_length} data points:", (df.iloc[:,9:].isna().sum(axis=1) >= (327 - min_length)).sum())
        df = df[df.iloc[:,9:].isna().sum(axis=1) < 327 - min_length]

    if remove_zero_timeseries:
        print("Removed timeseries with no value larger than zero:", (df.iloc[:,9:].max(axis=1) <= 0).sum())
        df = df.loc[df.iloc[:,9:].max(axis=1) > 0]

    if drop_const:
        print("Removed timeseries with constant values:", (df.iloc[:,9:].min(axis=1) ==  df.iloc[:,8:].max(axis=1)).sum())
        df = df.loc[df.iloc[:,9:].min(axis=1) < df.iloc[:,9:].max(axis=1)]

    if drop_x_const > 0:
        print(f"Removed timeseries with less than or equal to {drop_x_const} different values:", (df.iloc[:,9:].nunique(axis=1) <= drop_x_const).sum())
        df = df.loc[df.iloc[:,9:].nunique(axis=1) > drop_x_const]

    if remove_formative:
        print("Removed data within the following number of timeseries that are during the formative phase:" )
        df = df.apply(replace_before_growth, axis = 1)
            
    df = df.dropna(subset=df.columns[9:], how='all')

    return df


def replace_after_max(row):
    max_year = row[9:].idxmax()
    colnames = np.arange(max_year+1, 2025, 1)
    row[colnames]=np.nan
    return row

def replace_before_growth(row):
    if row['Metric'] in ['Annual Production']:
        cap_is_greater = list(row.iloc[9:].notna())
    elif row['Metric'] in ['Cumulative Length', 'Total Number', 'Net Total Capacity', 'Installed Capacity', 'Computing Capacity', 'Cumulative Rated Power', 'Cumulative Rated Capacity', 'Cumulative Acreage', 'Installed electricity capacity']:
        cap_is_greater = [False]+[row.iloc[i] > row.iloc[i-1] for i in range(10, len(row))]
    elif row['Metric'] in ['Share of Households', 'Share of Population', 'Share of Market', 'Share of Boilers']:
        cap_is_greater = [False]+[row.iloc[i] > row.iloc[i-1] for i in range(10, len(row))]
    else: cap_is_greater = np.zeros(len(row[9:]), dtype='bool')

    for i in range(2,len(cap_is_greater)):
        if cap_is_greater[i] & cap_is_greater[i-1] & cap_is_greater[i-2]:
            start_growth = i-2
            break
        else: start_growth = i

    row.iloc[9:(start_growth+8)] = np.nan
    return(row)
