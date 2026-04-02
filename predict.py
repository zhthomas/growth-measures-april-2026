from curve_fitting import *
from goodness_fit_measures import compute_mape

#it would be nice to have a dictionary for the functions 

#map index of results list to function, with a different number of arguments for each
def map_to_function(results, testx, testy):
    if results['functional_form'] == 'exponential':
        popt = [results['a'], results['b'], results['c']]
        mape_of_i = compute_mape(testx, testy, popt, func = exp_func)

        popt_upp = [results['a']+results['a_std'], results['b']+results['b_std'], results['c']]
        popt_low = [results['a']-results['a_std'], results['b']-results['b_std'], results['c']]
        yupp = exp_func(testx[-1], *popt_upp)
        ylow = exp_func(testx[-1], *popt_low)
        try:
            CI_size = (yupp - ylow)/testy[-1]
        except:
            CI_size = np.nan
        if yupp > ylow: 
            rv_above = int(testy[-1] > yupp)
            rv_below = int(testy[-1] < ylow)
            outside_CI = rv_above - rv_below #0 if inside CI, -1 if below CI, +1 if above CI 
        else: outside_CI = 2

    elif results['functional_form'] == 'linear':
        popt = [results['a'], results['b']]
        mape_of_i = compute_mape(testx, testy, popt, func = linear_func)
        
        popt_upp = [results['a']+results['a_std'], results['b']+results['b_std']]
        popt_low = [results['a']-results['a_std'], results['b']-results['b_std']]
        yupp = linear_func(testx[-1], *popt_upp)
        ylow = linear_func(testx[-1], *popt_low)
        try:
            CI_size = (yupp - ylow)/testy[-1]
        except:
            CI_size = np.nan
        if yupp > ylow: 
            rv_above = int(testy[-1] > yupp)
            rv_below = int(testy[-1] < ylow)
            outside_CI = rv_above - rv_below #0 if inside CI, -1 if below CI, +1 if above CI 
        else: outside_CI = 2

    elif results['functional_form'] == 'logistic':
        popt = [results['a'], results['b'], results['c']]
        mape_of_i = compute_mape(testx, testy, popt, func = logistic_func)
        
        popt_upp = [results['a']+results['a_std'], results['b']+results['b_std'], results['c']-results['c_std']]
        popt_low = [results['a']-results['a_std'], results['b']-results['b_std'], results['c']+results['c_std']]
        yupp = logistic_func(testx[-1], *popt_upp)
        ylow = logistic_func(testx[-1], *popt_low)
        try: 
            CI_size = (yupp - ylow)/testy[-1]
        except:
            CI_size = np.nan
        if yupp > ylow: 
            rv_above = int(testy[-1] > yupp)
            rv_below = int(testy[-1] < ylow)
            outside_CI = rv_above - rv_below #0 if inside CI, -1 if below CI, +1 if above CI 
        else: outside_CI = 2

    elif results['functional_form'] == 'gompertz':
        popt = [results['a'], results['b'], results['c']]
        mape_of_i = compute_mape(testx, testy, popt, func = gompertz_func3)
        
        popt_upp = [results['a']+results['a_std'], results['b']+results['b_std'], results['c']-results['c_std']]
        popt_low = [results['a']-results['a_std'], results['b']-results['b_std'], results['c']+results['c_std']]
        yupp = gompertz_func3(testx[-1], *popt_upp)
        ylow = gompertz_func3(testx[-1], *popt_low)
        try:
            CI_size = (yupp - ylow)/testy[-1]
        except:
            CI_size = np.nan
        if yupp > ylow: 
            rv_above = int(testy[-1] > yupp)
            rv_below = int(testy[-1] < ylow)
            outside_CI = rv_above - rv_below #0 if inside CI, -1 if below CI, +1 if above CI 
        else: outside_CI = 2

    elif results['functional_form'] == 'softplus':
        popt = [results['a'], results['b'], results['c']]
        mape_of_i = compute_mape(testx, testy, popt, func = softplus_func)
        
        popt_upp = [results['a']+results['a_std'], results['b']+results['b_std'], results['c']-results['c_std']]
        popt_low = [results['a']-results['a_std'], results['b']-results['b_std'], results['c']+results['c_std']]
        yupp = softplus_func(testx[-1], *popt_upp)
        ylow = softplus_func(testx[-1], *popt_low)
        try:
            CI_size = (yupp - ylow)/testy[-1]
        except:
            CI_size = np.nan
        if yupp > ylow: 
            rv_above = int(testy[-1] > yupp)
            rv_below = int(testy[-1] < ylow)
            outside_CI = rv_above - rv_below #0 if inside CI, -1 if below CI, +1 if above CI 
        else: outside_CI = 2

    elif results['functional_form'] == 'logistic-linear-cont':
        popt = [results['a'], results['b'], results['c']]
        mape_of_i = compute_mape(testx, np.log(testy), popt, func = logistic_linear_cont)
        
        popt_upp = [results['a']+results['a_std'], results['b']+results['b_std'], results['c']-results['c_std']]
        popt_low = [results['a']-results['a_std'], results['b']-results['b_std'], results['c']+results['c_std']]
        yupp = logistic_linear_cont(testx[-1], *popt_upp)
        ylow = logistic_linear_cont(testx[-1], *popt_low)
        try:
            CI_size = (yupp - ylow)/testy[-1]
        except:
            CI_size = np.nan
        if yupp > ylow: 
            rv_above = int(testy[-1] > yupp)
            rv_below = int(testy[-1] < ylow)
            outside_CI = rv_above - rv_below #0 if inside CI, -1 if below CI, +1 if above CI 
        else: outside_CI = 2

    elif results['functional_form'] == 'bass':
        popt = [results['a'], results['b'], results['c'], results['d']]
        mape_of_i = compute_mape(testx, testy, popt, func = bass_func)
        
        popt_upp = [results['a']+results['a_std'], results['b']+results['b_std'], results['c']-results['c_std'], results['d']+results['d_std']]
        popt_low = [results['a']-results['a_std'], results['b']-results['b_std'], results['c']+results['c_std'], results['d']-results['d_std']]
        yupp = bass_func(testx[-1], *popt_upp)
        ylow = bass_func(testx[-1], *popt_low)
        try:
            CI_size = (yupp - ylow)/testy[-1]
        except:
            CI_size = np.nan
        if yupp > ylow: 
            rv_above = int(testy[-1] > yupp)
            rv_below = int(testy[-1] < ylow)
            outside_CI = rv_above - rv_below #0 if inside CI, -1 if below CI, +1 if above CI 
        else: outside_CI = 2

    elif results['functional_form'] == 'bertalanffy':
        popt = [results['a'], results['b'], results['c']]
        mape_of_i = compute_mape(testx, testy, popt, func = bertalanffy_func3)
        
        popt_upp = [results['a']+results['a_std'], results['b']+results['b_std'], results['c']-results['c_std']]
        popt_low = [results['a']-results['a_std'], results['b']-results['b_std'], results['c']+results['c_std']]
        yupp = bertalanffy_func3(testx[-1], *popt_upp)
        ylow = bertalanffy_func3(testx[-1], *popt_low)
        try:
            CI_size = (yupp - ylow)/testy[-1]
        except:
            CI_size = np.nan
        if yupp > ylow: 
            rv_above = int(testy[-1] > yupp)
            rv_below = int(testy[-1] < ylow)
            outside_CI = rv_above - rv_below #0 if inside CI, -1 if below CI, +1 if above CI 
        else: outside_CI = 2

    elif results['functional_form'] == 'richards':
        popt = [results['a'], results['b'], results['c'], results['d']]
        mape_of_i = compute_mape(testx, testy, popt, func = richards_func)
        
        popt_upp = [results['a']+results['a_std'], results['b']+results['b_std'], results['c']-results['c_std'], results['d']+results['d_std']]
        popt_low = [results['a']-results['a_std'], results['b']-results['b_std'], results['c']+results['c_std'], results['d']-results['d_std']]
        yupp = richards_func(testx[-1], *popt_upp)
        ylow = richards_func(testx[-1], *popt_low)
        try:
            CI_size = (yupp - ylow)/testy[-1]
        except:
            CI_size = np.nan
        if yupp > ylow: 
            rv_above = int(testy[-1] > yupp)
            rv_below = int(testy[-1] < ylow)
            outside_CI = rv_above - rv_below #0 if inside CI, -1 if below CI, +1 if above CI 
        else: outside_CI = 2

    else: mape_of_i = 'Function not indexed'

    return mape_of_i, outside_CI, CI_size 

def predict_mape(results, testx, testy):
    output = []
    for i in results:
        if i['fit_success']: mape_of_i = map_to_function(i, testx, testy)
        else: mape_of_i = np.nan, np.nan, np.nan
        i_dict = dict([(j, i[j]) for j in i])
        i_dict['Length_test']= len(testx)
        i_dict['Hindcast_MAPE']= mape_of_i[0]
        i_dict['outside_CI'] = mape_of_i[1]
        i_dict['CI_size'] = mape_of_i[2]
        output.append(i_dict)
        

    return output
