import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from curve_fitting import *

plt.rcParams['figure.facecolor'] = 'white'


# ==============================================================
# routine for plotting of single time series with different fits

def plot_curve_fits(x, y,
                    fit_results, # usually list of dicts, but can also be single dict, pandas.Series or pandas.DataFrame
                    file_path='',
                    ax=None,
                    ylabel='',
                    show_plot=True,
                    log_plot=False,
                    tech_name='',
                    linestyle='--',
                    verbosity=0,
                    xlabel='',
                    title=''):
    
    if isinstance(fit_results, pd.Series):
        iter_obj = [fit_results.to_dict()]

    elif isinstance(fit_results, dict):
        iter_obj = [fit_results]

    elif isinstance(fit_results, pd.DataFrame):
        iter_obj = fit_results.to_dict('records')
    
    else:
        iter_obj = fit_results

    if ax:
        axis = ax
    else:
        fig, axis = plt.subplots(1, 1, figsize=(7, 5))
    
    # plot historical data
    axis.plot(x, y, 'ko', label="Historical data", alpha=0.5)

    for result in iter_obj:
        if verbosity > 1:
            print(result)
        
        if not result['fit_success']:
            if verbosity > 0:
                print(f'Not plotting function for failed fit:  {result}')
            continue

        if verbosity > 0:
            print(f'Plotting function for parameters:  {result}')

        popt = []
        for parameter in ['a', 'b', 'c', 'd', 'e']:
            if parameter in result.keys():
                if np.isfinite(result[parameter]):
                    popt.append(result[parameter])

        perr = []
        for parameter in ['a', 'b', 'c', 'd', 'e']:
            if parameter + '_std' in result.keys():
                if np.isfinite(result[parameter]):
                    perr.append(result[parameter + '_std'])

        r_squared = result['r_squared']

        if result['functional_form'] == 'logistic':
            axis.plot(x, logistic_func(x, *popt), c='r', linestyle=linestyle,
                      label='Logistic') #f'Logistic ({x[0]}-{x[-1]})\nb: {popt[1]:.4f}, R^2: {r_squared:.4f}'

        elif result['functional_form'] == 'exponential':
            axis.plot(x, exp_func(x, *popt), c='g', linestyle=linestyle,
                      label='Exponential') #f'Exponential ({x[0]}-{x[-1]})\nb: {popt[1]:.4f}, R^2: {r_squared:.4f}'

        elif result['functional_form'] == 'softplus':
            axis.plot(x, softplus_func(x, *popt), c='m', linestyle=linestyle,
                      label='Softplus') #f'Softplus ({x[0]}-{x[-1]})\nb: {popt[1]:.4f}, R^2: {r_squared:.4f}'

        elif result['functional_form'] == 'logistic-linear-cont':
            axis.plot(x, logistic_linear_cont(x, *popt), c='b', linestyle=linestyle,
                      label='Logistic with linear cont') #f'Logistic with lin cont ({x[0]}-{x[-1]})\nb: {popt[1]:.4f}, R^2: {r_squared:.4f}'

        elif result['functional_form'] == 'log-lin':
            axis.plot(x, np.exp(linear_func(x, *popt)), c='y', linestyle=linestyle,
                      label='Exponential log-linear') #f'Exponential log-lin ({x[0]}-{x[-1]})\nb: {popt[1]:.4f}'

        elif result['functional_form'] == 'linear':
            axis.plot(x, linear_func(x, *popt), c='c', linestyle=linestyle,
                      label='Linear') #f'Linear ({x[0]}-{x[-1]})\nb: {popt[1]:.4f}, R^2: {r_squared:.4f}'

        elif result['functional_form'] == 'gompertz':
            axis.plot(x, gompertz_func3(x, *popt), c='k', linestyle=linestyle,
                      label='Gompertz') #f'Gompertz ({x[0]}-{x[-1]})\nb: {popt[1]:.4f}, R^2: {r_squared:.4f}'

        elif result['functional_form'] == 'richards':
            axis.plot(x, richards_func(x, *popt), c='darkgreen', linestyle=linestyle,
                      label='Richards') #f'Richards fit ({x[0]}-{x[-1]})\nb: {popt[1]:.4f}, R^2: {r_squared:.4f}'

        elif result['functional_form'] == 'bass':
            axis.plot(x, bass_func(x, *popt), c='brown', linestyle=linestyle,
                      label='Bass') #f'Bass fit ({x[0]}-{x[-1]})\nb: {popt[1]:.4f}, R^2: {r_squared:.4f}'

        elif result['functional_form'] == 'bertalanffy':
            axis.plot(x, bertalanffy_func3(x, *popt), c='violet', linestyle=linestyle,
                      label='Bertalanffy') #f'Bertalanffy fit ({x[0]}-{x[-1]})\nb: {popt[1]:.4f}, R^2: {r_squared:.4f}'

    if not (ylabel == ''):
        axis.set_ylabel(ylabel, fontsize = 12)

    if not (xlabel == ''):
        axis.set_xlabel(xlabel, fontsize = 12)

    if not (title == ''):
        axis.set_title(title, fontsize = 12)
    
    axis.legend()

    # rescale axis
    lim = axis.get_ylim()
    if verbosity > 0:
        print(lim)
    if lim[0] < ((-0.1) * np.nanmax(y)):
        axis.set_ylim([(-0.1) * np.nanmax(y), lim[1]])
        lim = axis.get_ylim()
    if lim[1] > (1.4 * np.nanmax(y)):
        axis.set_ylim([lim[0], 1.4 * np.nanmax(y)])
        if verbosity > 0:
            print("Adjusting ylim:", axis.get_ylim())

    if tech_name:
        axis.set_title("Data: {}".format(tech_name))

    if log_plot:

        axis.set_yscale('log')
        # rescale axis
        lim = axis.get_ylim()
        if lim[0] <= 0:
            if np.min(y) > 0:
                axis.set_ylim([np.min(y), lim[1]])
            else:
                axis.set_ylim([0.0001, lim[1]])

    if file_path:
        fig.savefig(file_path, bbox_inches='tight')

    if not show_plot and not ax:
        plt.tight_layout()
        plt.close(fig)

    plt.tight_layout()

    if ax:
        return axis
    else:
        return fig
