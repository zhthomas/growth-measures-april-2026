# imports
import numpy as np
import numpy.ma as ma
from scipy.optimize import curve_fit, differential_evolution

from goodness_fit_measures import *


# helper function for collecting results
def results_to_dict(functional_form, fit_procedure, fit_success, popt, pcov, fit_goodness):
    paras = ['a', 'b', 'c', 'd', 'e']
    row = {'functional_form': functional_form, 'fit_procedure': fit_procedure, 'fit_success': fit_success,
           'r_squared': fit_goodness[0], 'adj_r_squared': fit_goodness[1],
           'MCp': fit_goodness[2], 'BIC': fit_goodness[3], 'MAPE': fit_goodness[4]}

    for i in range(len(popt)):
        row[paras[i]] = popt[i]

    if pcov is not None:
        perr = np.sqrt(np.diag(pcov))
        for i in range(len(popt)):
            row["{}_std".format(paras[i])] = perr[i]
    return row


def compound_growth_rate_from_points(value_t1, value_t2, t_diff):
    # compound exponential rate from points
    return (value_t2 / value_t1) ** (1. / t_diff) - 1


def compound_growth_rate_from_series(x, y):
    # mask zero values
    nonzero_idx = np.nonzero(y)[0]
    return compound_growth_rate_from_points(y[nonzero_idx[0]], y[nonzero_idx[-1]],
                                            x[nonzero_idx[-1]] - x[nonzero_idx[0]])


def central_value(x):
    return (x[0] + x[-1]) / 2

# ========= functions for differential evolution optimization ================


def compute_sum_of_squared_error(params, func, x_data, y_data):
    return np.sum((y_data - func(x_data, *params)) ** 2)


def compute_sum_of_abs_error(params, func, x_data, y_data):
    return np.sum(np.abs(y_data - func(x_data, *params)))


def compute_sum_of_relative_error(params, func, x_data, y_data):
    return np.sum(np.abs(y_data - func(x_data, *params)) / y_data)


# differential_evolution optimization wrapper
def fit_differential_evolution(param_bounds, func, x_data, y_data, method='ols', seed=4, strategy='best1bin'):
    if method == 'ols':
        objective_func = compute_sum_of_squared_error
    elif method == 'abs':
        objective_func = compute_sum_of_abs_error
    elif method == 'mre':
        objective_func = compute_sum_of_relative_error
    else:
        # default squared errors
        objective_func = compute_sum_of_squared_error

    result = differential_evolution(objective_func, param_bounds,
                                    args=(func, x_data, y_data), seed=seed,
                                    strategy=strategy, popsize=25)
    return result.x

# =============
# fit quadratic

def quadratic_func(x, a, b):
    return b*x**2 + a

def fit_quadratic(x, y, verbosity=0):
    try:
        popt_quad, pcov_quad = curve_fit(quadratic_func, x, y)
    
    except Exception as error:
        popt_quad = str(error)
        pcov_quad = str(error)
    
    return popt_quad, pcov_quad

# ======================================
# fit exponential with different methods


def linear_func(x, a, b):
    return a + b * x


def exp_func(x, a, b, c):
    return a * np.exp(b * (x - c))


def exp_func2(x, a, b):
    return a * np.exp(b * x)


def fit_exponential(x, y, fit_logged=False, restrict_exponential=-1, verbosity=0):

    bounds = [[0, 0, 1000], [np.inf, 100, 2500]]

    if len(y) < 2:
        results_dict = {'functional_form': 'exponential', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'DataError', 'error_msg': 'Fit params > n'}
        return results_dict

    if restrict_exponential > 0:
        xr = x[:restrict_exponential]
        yr = y[:restrict_exponential]
    elif restrict_exponential == 0:
        # restrict to end of increase
        idx = np.argmax(y)
        xr = x[:idx]
        yr = y[:idx]
    else:
        xr = x
        yr = y
    
    compound_growth_rate = compound_growth_rate_from_series(xr, yr)

    if verbosity > 0:
        print("Growth rate from points:", compound_growth_rate)

    if fit_logged:
        # exponential on logged data
        try:
            popt_loglin, pcov_loglin, infodict, mesg, ier = curve_fit(linear_func, x, np.log(y), maxfev=50000, full_output=True)

            gof_loglin = compute_all_goodness_of_fit(x, np.log(y), popt_loglin, linear_func)

            results_dict = results_to_dict('log-linear', 'curve_fit',
                                           True, popt_loglin, pcov_loglin, gof_loglin)
            if verbosity > 2:
                results_dict['infodict'] = infodict
            results_dict['fit_message'] = mesg
            results_dict['int_flag'] = ier

        except RuntimeError as error:
            results_dict = {'functional_form': 'log-linear', 'fit_procedure': 'curve_fit',
                            'fit_success': False, 'error_type': 'RuntimeError', 'error_msg': str(error)}

        except ValueError as error:
            results_dict = {'functional_form': 'log-lin', 'fit_procedure': 'curve_fit',
                            'fit_success': False, 'error_type': 'ValueError', 'error_msg': str(error)}

        if verbosity > 0:
            print(results_dict)

    else:
        # exponential with nonlinear least squares

        if (compound_growth_rate < 100) and (compound_growth_rate > 0):
            init_exp = [np.mean(yr), compound_growth_rate, central_value(xr)]
        else:
            init_exp = [np.mean(yr), 1, central_value(xr)]

        if verbosity > 0:
            print("Initial parameters for fitting exponential:", init_exp)

        try:
            popt_exp, pcov_exp, infodict, mesg, ier = curve_fit(exp_func, xr, yr, p0=init_exp,
                                           maxfev=50000, bounds=bounds,
                                           x_scale=init_exp, verbose=verbosity, full_output=True)
            if verbosity > 1:
                print(popt_exp)

            gof_exp = compute_all_goodness_of_fit(xr, yr, popt_exp, exp_func)

            results_dict = results_to_dict('exponential', 'curve_fit',
                                           True, popt_exp, pcov_exp, gof_exp)
            if verbosity > 2:
                results_dict['infodict'] = infodict
            results_dict['fit_message'] = mesg
            results_dict['int_flag'] = ier

        except RuntimeError as error:
            results_dict = {'functional_form': 'exponential', 'fit_procedure': 'curve_fit',
                            'fit_success': False, 'error_type': 'RuntimeError', 'error_msg': str(error)}

        except ValueError as error:
            results_dict = {'functional_form': 'exponential', 'fit_procedure': 'curve_fit',
                            'fit_success': False, 'error_type': 'ValueError', 'error_msg': str(error)}

        if verbosity > 0:
            print(results_dict)

    return results_dict


# =====================
# logistic fit

def logistic_func(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))


def logistic_func2(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d


def logged_logistic_func(x, a, b, c):
    return np.log(a) - np.log(1 + np.exp(-b * (x - c)))


def fit_logistic(x, y, verbosity=0):

    if len(y) < 3:
        results_dict = {'functional_form': 'logistic', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'DataError', 'error_msg': 'Fit params > n'}
        return results_dict

    compound_growth_rate = compound_growth_rate_from_series(x, y)

    bounds = [[0, 0, 1000], [np.inf, 100, 2500]]

    if (compound_growth_rate < 100) and (compound_growth_rate > 0):
        init_log = [np.max(y), compound_growth_rate, central_value(x)]
    else:
        init_log = [np.max(y), 1, central_value(x)]

    if verbosity > 0:
        print("Initial parameters for fitting logistic:", init_log)

    try:
        popt_log, pcov_log, infodict, mesg, ier = curve_fit(logistic_func, x, y, p0=init_log,
                                       maxfev=50000, bounds=bounds, x_scale=init_log,
                                       verbose=verbosity, full_output=True)

        gof_log = compute_all_goodness_of_fit(x, y, popt_log, logistic_func)
    
        results_dict = results_to_dict('logistic', 'curve_fit',
                                       True, popt_log, pcov_log, gof_log)
        if verbosity > 2:
            results_dict['infodict'] = infodict
        results_dict['fit_message'] = mesg
        results_dict['int_flag'] = ier

    except RuntimeError as error:
        results_dict = {'functional_form': 'logistic', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'RuntimeError', 'error_msg': str(error)}

    except ValueError as error:
        results_dict = {'functional_form': 'logistic', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'ValueError', 'error_msg': str(error)}

    if verbosity > 0:
        print(results_dict)

    return results_dict
    
# ==========================
# linear fit


def fit_linear(x, y, verbosity=0, reason = 'fit_all'):

    if len(y) < 2:
        results_dict = {'functional_form': 'linear', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'DataError', 'error_msg': 'Fit params > n'}
        return results_dict

    try:
        popt_lin, pcov_lin, infodict, mesg, ier = curve_fit(linear_func, x, y, maxfev=50000, full_output=True)

        gof_lin = compute_all_goodness_of_fit(x, y, popt_lin, linear_func)
    
        results_dict = results_to_dict('linear', 'curve_fit',
                                       True, popt_lin, pcov_lin, gof_lin)
        if verbosity > 2:
            results_dict['infodict'] = infodict
        results_dict['fit_message'] = mesg
        results_dict['int_flag'] = ier

        if reason == 'stats': return popt_lin, pcov_lin

    except RuntimeError as error:
        results_dict = {'functional_form': 'linear', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'RuntimeError', 'error_msg': str(error)}

    except ValueError as error:
        results_dict = {'functional_form': 'linear', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'ValueError', 'error_msg': str(error)}

    if verbosity > 0:
        print(results_dict)

    return results_dict


# ==========================
# softplus fit

def softplus_func(x, a, b, c):
    return a / b * np.log(1 + np.exp(b * (x - c)))


# alternative version of softplus function dealing with overflow in exp
def softplus_func2(x, a, b, c):
    x_t = b * (x - c)

    xm_smaller = ma.masked_greater(x_t, 100)  # for values > 700 there is an overflow
    xm_greater = ma.masked_less_equal(x_t, 100)

    comp = ma.log(1 + ma.exp(xm_smaller))

    return a / b * (comp.filled(0) + xm_greater.filled(0))


def fit_softplus(x, y, verbosity=0):

    if len(y) < 3:
        results_dict = {'functional_form': 'softplus', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'DataError', 'error_msg': 'Fit params > n'}
        return results_dict

    bounds = [[0, 0, 1000], [np.inf, 100, 2500]]

    compound_growth_rate = compound_growth_rate_from_series(x, y)

    if (compound_growth_rate < 100) and (compound_growth_rate > 0):
        init_sp = [central_value(y), compound_growth_rate, central_value(x)]
    else:
        init_sp = [central_value(y), 1, central_value(x)]
    
    if verbosity > 0:
        print("Initial parameters for fitting softplus:", init_sp)
    
    try:
        popt_sp, pcov_sp, infodict, mesg, ier = curve_fit(softplus_func2, x, y, p0=init_sp,
                                     maxfev=50000, bounds=bounds, x_scale=init_sp, verbose=verbosity, full_output=True)

        gof_sp = compute_all_goodness_of_fit(x, y, popt_sp, softplus_func2)

        results_dict = results_to_dict('softplus', 'curve_fit',
                                       True, popt_sp, pcov_sp, gof_sp)
        if verbosity > 2:
            results_dict['infodict'] = infodict
        results_dict['fit_message'] = mesg
        results_dict['int_flag'] = ier

    except RuntimeError as error:
        results_dict = {'functional_form': 'softplus', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'RuntimeError', 'error_msg': str(error)}

    except ValueError as error:
        results_dict = {'functional_form': 'softplus', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'ValueError', 'error_msg': str(error)}
    
    if verbosity > 0:
        print(results_dict)

    return results_dict


# ==========================
# Logistic with linear continuation

def linear_continuation(x, a, b, c):
    return a * (0.5 + b*(x - c)/4)


def logistic_linear_cont(x, a, b, c):
    logistic = a / (1 + np.exp(-b * (x - c)))
    linear = linear_continuation(x, a, b, c)
    return np.max([logistic, linear], axis=0)


def fit_logistic_linear_cont(x, y, verbosity=0):

    if len(y) < 3:
        results_dict = {'functional_form': 'logistic-linear-cont', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'DataError', 'error_msg': 'Fit params > n'}
        return results_dict

    bounds = [[0, 0, 1000], [np.inf, 100, 2500]]

    compound_growth_rate = compound_growth_rate_from_series(x, y)

    if (compound_growth_rate < 100) and (compound_growth_rate > 0):
        init_llc = [central_value(y), compound_growth_rate, central_value(x)]
    else:
        init_llc = [central_value(y), 1, central_value(x)]

    central_value(y)

    if verbosity > 0:
        print("Initial parameters for fitting softplus:", init_llc)

    try:
        popt, pcov, infodict, mesg, ier = curve_fit(logistic_linear_cont, x, y, p0=init_llc,
                               maxfev=50000, bounds=bounds, verbose=verbosity, full_output=True)

        gof_llc = compute_all_goodness_of_fit(x, y, popt, logistic_linear_cont)

        results_dict = results_to_dict('logistic-linear-cont', 'curve_fit',
                                       True, popt, pcov, gof_llc)
        if verbosity > 2:
            results_dict['infodict'] = infodict
        results_dict['fit_message'] = mesg
        results_dict['int_flag'] = ier

    except RuntimeError as error:
        results_dict = {'functional_form': 'logistic-linear-cont', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'RuntimeError', 'error_msg': str(error)}

    except ValueError as error:
        results_dict = {'functional_form': 'logistic-linear-cont', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'ValueError', 'error_msg': str(error)}

    if verbosity > 0:
        print(results_dict)

    return results_dict


# ================
# Gompertz fit

def gompertz_func(x, asymptote, growth_rate, time_shift):
    return asymptote * np.exp(-np.exp(-growth_rate * (x - time_shift)))


def gompertz_func2(x, a, b, c, d):
    return (a - d) * np.exp(-b * np.exp(-c * x)) + d


def gompertz_func3(x, a, b, c):
    return a * np.exp(-np.exp(-b * (x - c)))


def fit_gompertz(x, y, verbosity=0):

    if len(y) < 3:
        results_dict = {'functional_form': 'gompertz', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'DataError', 'error_msg': 'Fit params > n'}
        return results_dict

    compound_growth_rate = compound_growth_rate_from_series(x, y)
    par_bounds_de = [[0, 100 * np.max(y)], [1e-12, 10], [0, 2 * np.max(x)]]

    # try to get initial parameters from differential evolution
    init_gomp = fit_differential_evolution(par_bounds_de, gompertz_func3, x, y)

    # test whether these are good; if not, use standard initial parameters
    if compute_r_squared(x, y, init_gomp, gompertz_func3) < 0:
        if (compound_growth_rate < 100) and (compound_growth_rate > 0):
            init_gomp = [np.max(y), compound_growth_rate, central_value(x)]
        else:
            init_gomp = [np.max(y), 1, central_value(x)]

    if verbosity > 0:
        print("Initial parameters for fitting Gompertz from differential evolution:", init_gomp)

    bounds = [[0, 0, 0], [np.inf, np.inf, np.inf]]

    try:
        popt_gomp, pcov_gomp, infodict, mesg, ier = curve_fit(gompertz_func3, x, y, p0=init_gomp,
                                         maxfev=50000, bounds=bounds, x_scale=init_gomp,
                                         verbose=verbosity, full_output=True)

        gof_gomp = compute_all_goodness_of_fit(x, y, popt_gomp, gompertz_func3)
    
        results_dict = results_to_dict('gompertz', 'curve_fit',
                                       True, popt_gomp, pcov_gomp, gof_gomp)
        if verbosity > 2:
            results_dict['infodict'] = infodict
        results_dict['fit_message'] = mesg
        results_dict['int_flag'] = ier

    except RuntimeError as error:
        results_dict = {'functional_form': 'gompertz', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'RuntimeError', 'error_msg': str(error)}

    except ValueError as error:
        results_dict = {'functional_form': 'gompertz', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'ValueError', 'error_msg': str(error)}

    if verbosity > 0:
        print(results_dict)

    return results_dict


# ================
# Richards fit

def richards_func(x, a, b, c, d):
    return a * np.power(1 + (1 / d) * np.exp(-b * (x - c)), -d)


def richards_func2(x, a, b, c, d, e):
    return (c - e) * np.power(1 + (1 / d) * np.exp(-a * (x - b)), -d) + e


def fit_richards(x, y, verbosity=0):

    if len(y) < 4:
        results_dict = {'functional_form': 'richards', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'DataError', 'error_msg': 'Fit params > n'}
        return results_dict

    compound_growth_rate = compound_growth_rate_from_series(x, y)
    bounds = [[0, 0, 1000, 1], [np.inf, 10, 2500, 10]]

    if (compound_growth_rate < 100) and (compound_growth_rate > 0):
        if compound_growth_rate*10 > 1:
            init_rich = [np.max(y), compound_growth_rate, central_value(x), compound_growth_rate*10]
        else:
            init_rich = [np.max(y), compound_growth_rate, central_value(x), 3]
    else:
        init_rich = [np.max(y), 1, central_value(x), 3]

    if verbosity > 0:
        print("Initial parameters for fitting Richards:", init_rich)

    try:
        popt_rich, pcov_rich, infodict, mesg, ier = curve_fit(richards_func, x, y, p0=init_rich,
                                         maxfev=50000, bounds=bounds, x_scale=init_rich,
                                         verbose=verbosity, full_output=True)

        gof_rich = compute_all_goodness_of_fit(x, y, popt_rich, richards_func)
    
        results_dict = results_to_dict('richards', 'curve_fit',
                                       True, popt_rich, pcov_rich, gof_rich)
        if verbosity > 2:
            results_dict['infodict'] = infodict
        results_dict['fit_message'] = mesg
        results_dict['int_flag'] = ier

    except RuntimeError as error:
        results_dict = {'functional_form': 'richards', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'RuntimeError', 'error_msg': str(error)}

    except ValueError as error:
        results_dict = {'functional_form': 'richards', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'ValueError', 'error_msg': str(error)}

    if verbosity > 0:
        print(results_dict)

    return results_dict

# ================
# Bass fit


def bass_func(x, a, b, c, d):
    # return c * (1 - np.exp(-(a + b) * (x-d))) / (1 + (b / a) * np.exp(-(a + b) * x))
    return a * (1 - np.exp(-(d + b) * (x-c))) / (1 + (b / d) * np.exp(-(d + b) * (x - c)))

def bass_func2(x, a, b, c, d, e):
    return (a - e) * (1 - np.exp(-(d + b) * (x-c))) / (1 + (b / d) * np.exp(-(d + b) * (x - c))) + e


def fit_bass(x, y, verbosity=0):

    if len(y) < 4:
        results_dict = {'functional_form': 'bass', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'DataError', 'error_msg': 'Fit params > n'}
        return results_dict

    # bounds = [[0, 0, y[-1], 1000], [5, 5, np.inf, 2500]]
    bounds = [[0, 0, 1000, 0], [np.inf, 10, 2500, 1]]

    compound_growth_rate = compound_growth_rate_from_series(x, y)
    
    if (compound_growth_rate < 100) and (compound_growth_rate > 0):
        # init_bass = [compound_growth_rate/10, compound_growth_rate, np.max(y), central_value(x)]
        init_bass = [np.max(y), compound_growth_rate,  x[0], compound_growth_rate/10]
    else:
        # init_bass = [0.03, 0.38, np.max(y), central_value(x)]
        init_bass = [np.max(y), 0.38, x[0], 0.03]

    if verbosity > 0:
        print("Initial parameters for fitting Bass:", init_bass)

    try:
        popt_bass, pcov_bass, infodict, mesg, ier = curve_fit(bass_func, x, y, p0=init_bass,
                                         maxfev=50000, bounds=bounds, x_scale=init_bass,
                                         verbose=verbosity, full_output=True)

        gof_bass = compute_all_goodness_of_fit(x, y, popt_bass, bass_func)
    
        results_dict = results_to_dict('bass', 'curve_fit',
                                       True, popt_bass, pcov_bass, gof_bass)
        if verbosity > 2:
            results_dict['infodict'] = infodict
        results_dict['fit_message'] = mesg
        results_dict['int_flag'] = ier

    except RuntimeError as error:
        results_dict = {'functional_form': 'bass', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'RuntimeError', 'error_msg': str(error)}

    except ValueError as error:
        results_dict = {'functional_form': 'bass', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'ValueError', 'error_msg': str(error)}

    if verbosity > 0:
        print(results_dict)

    return results_dict


# ================
# Bertalanffy fit

def bertalanffy_func(x, asymptote, growth_rate, time_shift, d):
    return asymptote * (1 - d * np.exp(-growth_rate * (x - time_shift))) ** 3


def bertalanffy_func2(x, a, b, c, d, e):
    return (c - e) * (1 - d * np.exp(-a(x - b))) ** 3 + e


def bertalanffy_func3(x, a, b, c):
    return a * (1 - np.exp(-b * (x - c))) ** 3


def fit_bertalanffy(x, y, verbosity=0):

    if len(y) < 3:
        results_dict = {'functional_form': 'bertalanffy', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'DataError', 'error_msg': 'Fit params > n'}
        return results_dict

    bounds = [[0, 0, 1000], [np.inf, 100, 2500]]

    compound_growth_rate = compound_growth_rate_from_series(x, y)

    if (compound_growth_rate < 100) and (compound_growth_rate > 0):
        init_bert = [np.max(y), compound_growth_rate, central_value(x)]
    else:
        init_bert = [np.max(y), 1, central_value(x)]

    if verbosity > 0:
        print("Initial parameters for fitting Bertalanffy:", init_bert)

    try:
        popt_bert, pcov_bert, infodict, mesg, ier = curve_fit(bertalanffy_func3, x, y, p0=init_bert,
                                         maxfev=50000, bounds=bounds, x_scale=init_bert,
                                         verbose=verbosity, full_output=True)

        gof_bert = compute_all_goodness_of_fit(x, y, popt_bert, bertalanffy_func3)

        results_dict = results_to_dict('bertalanffy', 'curve_fit',
                                       True, popt_bert, pcov_bert, gof_bert)
        if verbosity > 2:
            results_dict['infodict'] = infodict
        results_dict['fit_message'] = mesg
        results_dict['int_flag'] = ier

    except RuntimeError as error:
        results_dict = {'functional_form': 'bertalanffy', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'RuntimeError', 'error_msg': str(error)}

    except ValueError as error:
        results_dict = {'functional_form': 'bertalanffy', 'fit_procedure': 'curve_fit',
                        'fit_success': False, 'error_type': 'ValueError', 'error_msg': str(error)}

    if verbosity > 0:
        print(results_dict)

    return results_dict


# =======================================
# fit all different functional forms

def make_curve_fits(x, y,
                    verbosity=0,
                    tech_name=None):

    results = []

    for fit_function in [fit_exponential, fit_linear, fit_logistic,
                         fit_gompertz, fit_softplus, fit_logistic_linear_cont,
                         fit_bass, fit_bertalanffy, fit_richards]:
        fit_result = fit_function(x, y, verbosity=verbosity)
        fit_result['technology'] = tech_name
        results.append(fit_result)

    return results
