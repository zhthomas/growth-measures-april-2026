# import stuff
import pandas as pd
from datetime import datetime
from curve_fitting import *
from plotting_utils import *
from timeseries_stats import *
from read_hatch import *

save_figs = False
save_dir = "./figures/individual-fits/"
set_test = ''  # set to False or 'random_sample' 'single' or 'beginning' for tests on different subsamples
set_stats_only = True #set to True if the only desired output is the time series stats

def run_single_test_fit(stats_only=False):
    tmp = df[df['Technology Name'] == 'Offshore Wind Energy']
    tmp = tmp.set_index('Region')[all_years]
    tmp.transpose().plot()

    tech_name = 'Railroad'
    metric = 'Cumulative Length'
    region = 'FRA'
    # slice = df[(df['Technology Name'] == tech_name) &
    #           (df['Metric'] == metric) &
    #           (df['Region'] == region)][all_years].transpose().dropna()

    # slice_df = df.loc['Offshore Wind Energy_Installed electricity capacity_DE'][all_years].transpose().dropna()
    slice_df = df.loc['Railroad_Cumulative Length_FR'][all_years].transpose().dropna()
    # slice_df = df.loc['Aquaculture Production_Annual production_CV'][all_years].transpose().dropna()
    tech_name = slice_df.name
    technology = df['Technology Name'].loc[slice_df.name]

    print(len(slice_df))
    slice_df.plot()
    years = slice_df.index.to_numpy(dtype='float')
    values = slice_df.to_numpy(dtype='float')

    stats_dict = compute_timeseries_stats(years, values)
    stats_dict['tech_name'] = tech_name
    print('Timeseries stats summary:')
    print(stats_dict)

    if not stats_only:
        results = make_curve_fits(years, values, tech_name=tech_name, verbosity=1)
        for r in results: r['Technology Name'] = technology
        fit_params = pd.DataFrame(results)
        fit_params.to_csv("./results/test_fitting_parameters.csv")

        if save_figs:
            plot_curve_fits(years, values, results, file_path=save_dir + 'test_all_fits.png')

    print("Test run finished.")

# run fits on all timeseries
def run_all_fits(test='', stats_only=False):

    start_time = datetime.now()
    # suppress overflow warnings
    np.seterr(over='ignore', under='ignore')

    all_results = []
    all_stats = []

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
        technology = df['Technology Name'].iloc[i]
        print(f"{datetime.now() - start_time} - Fitting {i}: {tech_name}")

        years = sl.index.to_numpy(dtype='float')

        if years.min() < 0 or years.max() > 2050:
            print("Warning: Time is not in years:", years)
            continue

        values = sl.to_numpy(dtype='float')

        stats_dict = compute_timeseries_stats(years, values)
        stats_dict['tech_name'] = tech_name
        stats_dict['Technology Name'] = technology
        all_stats.append(stats_dict)

        if not stats_only:
            rs = make_curve_fits(years, values, tech_name=tech_name, verbosity=0)
            for r in rs: r['Technology Name'] = technology
            if save_figs:
                plot_curve_fits(years, values, rs, show_plot=False, tech_name=tech_name,
                                file_path=save_dir + 'fits_' + tech_name + '.png')

            all_results += rs

    today = datetime.today().strftime('%Y-%m-%d')

    if not stats_only:
        # store fit parameters
        fit_params = pd.DataFrame(all_results)
        fit_params.to_csv(f"./results/fitting_parameters{file_suffix}_{today}.csv")

    # store curve stats
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(f"./results/timeseries_stats{file_suffix}_{today}.csv")

    print("Loop run finished.")


if __name__ == "__main__":

    # read tech data
    df = read_hatch('data/HATCH_v1.5_clean.csv')
    all_years = pd.to_numeric(df.columns, errors='coerce').dropna().astype(int)

    if set_test == 'single':
        run_single_test_fit(stats_only=set_stats_only)
    else:
        run_all_fits(test=set_test, stats_only=set_stats_only)
