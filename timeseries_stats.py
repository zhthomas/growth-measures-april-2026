import numpy as np
from scipy.stats import linregress#, spearmanr
# from curve_fitting import fit_quadratic, quadratic_func, fit_linear, linear_func
# from goodness_fit_measures import compute_r_squared

# def last_m_quadratic_reg(t, y, m=10):
#     try:
#         t = t[-m:]
#         y = y[-m:]
#     except Exception as e:
#         print(m)

#     popt_quad, pcov_quad = fit_quadratic(t, y)
#     quad_r2 = compute_r_squared(t, y, popt_quad, quadratic_func)

#     popt_lin, pcov_lin = fit_linear(t, y, reason='stats')
#     lin_r2 = compute_r_squared(t, y, popt_lin, linear_func)

#     if type(popt_quad) == str:
#         return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

#     #convexity is given by paramter b (second parameter)
#     sign_quad = np.sign(popt_quad[1]+np.diag(pcov_quad)[1]) + np.sign(popt_quad[1]-np.diag(pcov_quad)[1])
#     #slope is given by parameter b (second parameter)
#     sign_linear = np.sign(popt_lin[1]+np.diag(pcov_lin)[1]) + np.sign(popt_lin[1]-np.diag(pcov_lin)[1])

#     #goal is to return the signs of [convexity, slope] if different from 0
#     #if upper bound of CI is one sign, and lower bound is opposite, then the mean is 0
#     #if both upper bound of CI are the same sign, then mean is whatever that sign is
#     #I don't think it's likely that the lower or upper bound is exactly 0, such that the mean is 0 but counted as pos/neg
#     return [np.sign(sign_quad), round(popt_quad[1],4), quad_r2, np.sign(sign_linear), round(popt_lin[1],4), lin_r2]
                 

def compute_autocorrelation(x, lag=1):
    return np.corrcoef(x[:-lag], x[lag:])[0][1]


def compute_roughness(x):
    d1 = np.diff(x)
    if np.std(d1) > 0:
        norm_d1 = (d1 - np.mean(d1)) / np.std(d1)
    else:
        return 0
    roughness = (np.diff(norm_d1) ** 2) / 4
    return np.mean(roughness)


def compute_timeseries_stats(t, y):
    cdict = dict()
    cdict['n'] = len(y)

    dt = np.diff(t)
    cdict['t_min'] = np.min(t)
    cdict['t_max'] = np.max(t)
    cdict['dt_min'] = np.min(dt)
    cdict['dt_max'] = np.max(dt)
    cdict['dt_mean'] = np.mean(dt)
    cdict['dt_std'] = np.std(dt)

    cdict['y_median'] = np.median(y)
    cdict['y_mean'] = np.mean(y)
    cdict['y_std'] = np.std(y)
    cdict['y_min'] = np.min(y)
    cdict['y_max'] = np.max(y)

    if len(y) < 2:
        return cdict

    cdict['y_min_rel_pos'] = np.argmin(y) / (len(y) - 1)
    cdict['y_max_rel_pos'] = np.argmax(y) / (len(y) - 1)

    # comparison of mean after the maximum to the maximum
    if np.argmax(y) == (len(y) - 1):
        cdict['y_pct_mean_drop_after_max'] = 0
    else:
        mean_after_max = np.mean(y[np.argmax(y) + 1:])
        cdict['y_pct_mean_drop_after_max'] = 100 * (np.max(y) - mean_after_max) / np.max(y)

    res = linregress(t, y)
    cdict['y_trend_slope'] = res.slope
    cdict['y_trend_pval'] = res.pvalue

    dy = np.diff(y)
    cdict['dy_mean'] = np.mean(dy)
    cdict['dy_std'] = np.std(dy)
    if np.mean(dy) == 0:
        cdict['dy_cv'] = np.nan
    else:
        cdict['dy_cv'] = np.std(dy) / np.mean(dy)
    res = linregress(t[1:], dy)
    cdict['dy_trend_slope'] = res.slope
    cdict['dy_trend_pval'] = res.pvalue

    ddy = np.diff(dy)
    cdict['ddy_mean'] = np.mean(ddy)
    cdict['ddy_std'] = np.std(ddy)
    res = linregress(t[2:], ddy)
    cdict['ddy_trend_slope'] = res.slope
    cdict['ddy_trend_pval'] = res.pvalue

    cdict['ar1'] = compute_autocorrelation(y)
    cdict['ar2'] = compute_autocorrelation(y, lag=2)
    cdict['roughness'] = compute_roughness(y)

    last_m = int(len(y)/5)
    if last_m<10: last_m=10
    cdict['lastm'] = last_m

    quadreg = linregress(t[-last_m:]**2, y[-last_m:])
    cdict['lastm_convexity_sign'] = np.sign(np.sign(quadreg.slope+quadreg.stderr) + np.sign(quadreg.slope-quadreg.stderr))
    cdict['lastm_convexity_mean'] = round(quadreg.slope, 4)
    cdict['lastm_convexity_pvalue'] = quadreg.pvalue

    linereg = linregress(t[-last_m:], y[-last_m:])
    cdict['lastm_slope_sign'] = np.sign(np.sign(linereg.slope+linereg.stderr) + np.sign(linereg.slope-linereg.stderr))
    cdict['lastm_slope_mean'] = round(linereg.slope, 4)
    cdict['lastm_lin_pvalue'] = linereg.pvalue

    # convexity = last_m_quadratic_reg(t,y, last_m)
    # cdict['lastm_convexity_sign'] = convexity[0]
    # cdict['lastm_convexity_mean'] = convexity[1]
    # cdict['lastm_quad_r2'] = convexity[2]
    # cdict['lastm_slope_sign'] = convexity[3]
    # cdict['lastm_slope_mean'] = convexity[4]
    # cdict['lastm_lin_r2'] = convexity[5]

    #if continuous exponential growth >>1, if saturated growth 0-1, if declining growth <0
    cdict['ratio_last1m_v_first'] = (y[-1]-y[-last_m])/(y[-1]-y[0])

    return cdict
