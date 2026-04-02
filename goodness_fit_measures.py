# utility functions for calculating goodness of fit measures

import numpy as np
from sklearn.metrics import mean_absolute_percentage_error


def compute_r_squared(xdata, ydata, popt, func):
    residuals = ydata - func(xdata, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

# Mallow's C_p
def compute_mcp(xdata, ydata, popt, func):
    residuals = ydata - func(xdata, *popt)
    ss_res = np.sum(residuals**2)
    mse = ss_res/len(xdata)
    mcp = (ss_res + 2*len(popt)*mse)/len(xdata)
    # 2*number of predictors*variance of error for each measurement
    return mcp

# Bayesian information criterion
def compute_bic(xdata, ydata, popt, func):
    residuals = ydata - func(xdata, *popt)
    ss_res = np.sum(residuals**2)
    mse = ss_res/len(xdata)
    bic = (ss_res + np.log(len(xdata))*len(popt)*mse)/len(xdata)
    return bic

# adjusted r_squared
def compute_adj_r_squared(xdata, ydata, popt, func):
    residuals = ydata - func(xdata, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    adj_r_squared = 1 - ((ss_res/(len(xdata)-len(popt)-1)) / (ss_tot/(len(xdata)-1)))
    return adj_r_squared


# Mean absolute percentage error (MAPE)
def compute_mape(xdata, ydata, popt, func):
    # note: mean_absolute_percentage_error from sklearn returns a relative value, not percentage (1 means 100%)
    # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html
    try:
        mape = mean_absolute_percentage_error(ydata, func(xdata, *popt))
    except ValueError as error:
        print("Warning: computing MAPE failed.")
        print(error)
        mape = np.nan
    return mape


def compute_all_goodness_of_fit(xdata, ydata, popt, func):
    pars = [xdata, ydata, popt, func]
    return [compute_r_squared(*pars), compute_adj_r_squared(*pars),
            compute_mcp(*pars), compute_bic(*pars), compute_mape(*pars)]
