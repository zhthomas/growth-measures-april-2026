import pandas as pd
import pyarrow
from datetime import datetime
from read_hatch import *
from curve_fitting import *
from scipy.optimize import curve_fit, differential_evolution

from goodness_fit_measures import compute_mape
from predict import *

test = ''

df = read_hatch('data/HATCH_v1.5_clean.csv')
all_years = pd.to_numeric(df.columns, errors='coerce').dropna().astype(int)

#Group metrics to get cumulative sum of annual data
# correct internally inconsistent labels
df['Metric'] = df['Metric'].replace({'Annual production': 'Annual Production',
                                     'Cumulative total capacity': 'Cumulative Total Capacity'})
# Group similar 'metrics' together
group_metrics = {'Annual Production': 'annual',
                 'Cumulative Length': 'cumulative',
                 'Total Number': 'cumulative',
                'Cumulative Total Capacity': 'cumulative',
                'Total Length': 'cumulative',
                'Share of Households': 'share',
                'Net Total Capacity': 'cumulative',
                'Installed Capacity': 'cumulative',
                'Share of Population': 'share',
                'Computing Capacity': 'cumulative',
                'Share of Market': 'share',
                'Cumulative Rated Power': 'cumulative',
                'Cumulative Rated Capacity': 'cumulative',
                'Cumulative Acreage': 'cumulative',
                'Installed electricity capacity': 'cumulative',
                'Share of Boilers': 'share'}
df.insert(loc = 6, column='metric_grouped', value = df['Metric'].map(group_metrics))

if test == 'single':
    slice_df = df.loc['Nuclear Energy_Annual Production_UA'][all_years].transpose().dropna()
    # slice_df = df.loc['Sugar Output_Annual Production_JM'][all_years].transpose().dropna()
    # slice_df = df.loc['Railroad_Cumulative Length_FR'][all_years].transpose().dropna()
    # slice_df = df.loc['Aquaculture Production_Annual production_CV'][all_years].transpose().dropna()
    tech_name = slice_df.name

    print(len(slice_df))

    years = slice_df.index.to_numpy(dtype='float')
    values = slice_df.to_numpy(dtype='float')

    # if df.loc[tech_name]['metric_grouped'] == 'annual':
    #     values = values.cumsum()
            
    start_index = 0
    count_test = 1
    count_train = 5
    all_results = []

    while start_index <= 100:
        while count_train <= 40:
            trainx = years[start_index:(start_index+count_train)]
            trainy = values[start_index:(start_index+count_train)]
            if len(trainx) < ((start_index+count_train)-start_index):
                count_test = 1
                count_train += 5
                continue
            results = make_curve_fits(trainx, trainy, tech_name=tech_name, verbosity=0)
            for r in results: r['Start_index'] = start_index
            for r in results: r['Length_train'] = len(trainx)

            while count_test <= 40:
                testx = years[(start_index+count_train):(start_index+count_train+count_test)]
                testy = values[(start_index+count_train):(start_index+count_train+count_test)]
                if len(testx) < ((start_index+count_train+count_test)-(start_index+count_train)):
                    if count_test == 1:
                        count_test += 4
                    else: 
                        count_test += 5
                    continue              
                hindcast_results = predict_mape(results, testx, testy)
                if count_test == 1: 
                    count_test += 4
                else:
                    count_test += 5
                all_results += hindcast_results

            count_test = 1
            count_train += 5
        
        count_train = 5
        start_index += 10

    output_results = pd.DataFrame(all_results)
    output_results.to_parquet('./results/test_cont_hindcast.parquet')
    output_results.to_csv("./results/test_conthind.csv")

# run fits on all timeseries
else:

    start_time = datetime.now()
    # suppress overflow warnings
    np.seterr(over='ignore', under='ignore')

    all_results = []

    if test == 'snippet':
        n = 50
        loop_ids = np.arange(0, n)
        print(f"{start_time} - Start running loop over beginning of timeseries (n={n})")
        file_suffix = f'_beginning{n}'
    elif test == 'random_sample':
        n = 50
        loop_ids = np.random.choice(np.arange(0, len(df.index)), size=n, replace=False)
        loop_ids = np.sort(loop_ids)
        print(f"{start_time} - Start running loop over random sample of timeseries (n={n})")
        file_suffix = f'_randomsample{n}'
    else:
        loop_ids = np.arange(0, len(df.index))
        print(f"{start_time} - Start running loop over all timeseries")
        file_suffix = '_all'

    all_results = []

    for i in loop_ids:
        sl = df.iloc[i, 10:].transpose().dropna()
        tech_name = sl.name
        print(f"{datetime.now() - start_time} - Fitting {i}: {tech_name}")

        years = sl.index.to_numpy(dtype='float')

        if years.min() < 0 or years.max() > 2050:
            print("Warning: Time is not in years:", years)
            continue

        values = sl.to_numpy(dtype='float')

        # if df.loc[tech_name]['metric_grouped'] == 'annual':
        #     values = values.cumsum()

        start_index = 0
        count_test = 1
        count_train = 5

        while start_index < 100:

            while count_train < 40:
                trainx = years[start_index:(start_index+count_train)]
                trainy = values[start_index:(start_index+count_train)]
                if len(trainx) < ((start_index+count_train)-start_index):
                    count_test = 1
                    count_train += 5
                    continue
                results = make_curve_fits(trainx, trainy, tech_name=tech_name, verbosity=0)
                for r in results: r['Start_index'] = start_index
                for r in results: r['Length_train'] = len(trainx)

                while count_test < 40:
                    testx = years[(start_index+count_train):(start_index+count_train+count_test)]
                    testy = values[(start_index+count_train):(start_index+count_train+count_test)]
                    if len(testx) < ((start_index+count_train+count_test)-(start_index+count_train)):
                        if count_test == 1:
                            count_test += 4
                        else: 
                            count_test += 5
                        continue
                    hindcast_results = predict_mape(results, testx, testy)
                    if count_test == 1:
                        count_test += 4
                    else: 
                        count_test += 5
                    all_results += hindcast_results

                count_test = 1
                count_train += 5

            count_train = 5
            start_index += 5

    today = datetime.today().strftime('%Y-%m-%d')

    output_results = pd.DataFrame(all_results)
    output_results.to_parquet(f"./results/conthind_results{file_suffix}_{today}.parquet")
    if test in ['snippet', 'random_sample']: output_results.to_csv(f"./results/conthind_results{file_suffix}_{today}.csv")

    print("Loop run finished.")