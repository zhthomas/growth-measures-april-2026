import pandas as pd
from datetime import datetime
from read_hatch import *
from curve_fitting import *
from scipy.optimize import curve_fit, differential_evolution

from goodness_fit_measures import compute_mape
from predict import *

test = ''

df = read_hatch('data/HATCH_v1.5_clean.csv')
all_years = pd.to_numeric(df.columns, errors='coerce').dropna().astype(int)
    

if test == 'single':
    slice_df = df.loc['Railroad_Cumulative Length_FR'][all_years].transpose().dropna()
    # slice_df = df.loc['Aquaculture Production_Annual production_CV'][all_years].transpose().dropna()
    tech_name = slice_df.name

    print(len(slice_df))

    years = slice_df.index.to_numpy(dtype='float')
    values = slice_df.to_numpy(dtype='float')

    if len(years) < 20:
        trainx = years[:-5]
        trainy = values[:-5]
        results = make_curve_fits(trainx, trainy, tech_name=tech_name, verbosity=1)
        
        testx = years[-5:]
        testy = values[-5:]
        hindcast_results = predict_mape(results, testx, testy)
    else: 
        half_index = len(years)//2
        trainx = years[:-half_index]
        trainy = values[:-half_index]
        results = make_curve_fits(trainx, trainy, tech_name=tech_name, verbosity=1)

        testx = years[-half_index:]
        testy = values[-half_index:]
        hindcast_results = predict_mape(results, testx, testy)

    
    output_results = pd.DataFrame(hindcast_results)
    output_results.to_csv("./results/test_hindcasting_results.csv")

    # run fits on all timeseries
else:

    start_time = datetime.now()
    # suppress overflow warnings
    np.seterr(over='ignore', under='ignore')

    all_results = []

    if test == 'snippet':
        n = 500
        loop_ids = np.arange(0, n)
        print(f"{start_time} - Start running loop over beginning of timeseries (n={n})")
        file_suffix = f'_beginning{n}'
    elif test == 'random_sample':
        n = 500
        loop_ids = np.random.choice(np.arange(0, len(df.index)), size=n, replace=False)
        loop_ids = np.sort(loop_ids)
        print(f"{start_time} - Start running loop over random sample of timeseries (n={n})")
        file_suffix = f'_randomsample{n}'
    else:
        loop_ids = np.arange(0, len(df.index))
        print(f"{start_time} - Start running loop over all timeseries")
        file_suffix = '_all'

    for i in loop_ids:
        sl = df.iloc[i, 9:].transpose().dropna()
        tech_name = sl.name
        print(f"{datetime.now() - start_time} - Fitting {i}: {tech_name}")

        years = sl.index.to_numpy(dtype='float')

        if years.min() < 0 or years.max() > 2050:
            print("Warning: Time is not in years:", years)
            continue

        values = sl.to_numpy(dtype='float')

        if len(years) < 20:
            trainx = years[:-5]
            trainy = values[:-5]
            results = make_curve_fits(trainx, trainy, tech_name=tech_name, verbosity=0)
            
            testx = years[-5:]
            testy = values[-5:]
            hindcast_results = predict_mape(results, testx, testy)
        else: 
            half_index = len(years)//2
            trainx = years[:-half_index]
            trainy = values[:-half_index]
            results = make_curve_fits(trainx, trainy, tech_name=tech_name, verbosity=0)

            testx = years[-half_index:]
            testy = values[-half_index:]
            hindcast_results = predict_mape(results, testx, testy)

        all_results += hindcast_results

    today = datetime.today().strftime('%Y-%m-%d')

    output_results = pd.DataFrame(all_results)
    output_results.to_csv(f"./results/hindcasting_results{file_suffix}_{today}.csv")

    print("Loop run finished.")