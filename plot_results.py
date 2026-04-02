import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

# todo: adapt code to new results reporting structure
# Work in progress in jupyter notebook "Plot_stats_and_fit_results.ipynb"

fit_params = pd.read_csv("./results/historical_growth_rates_fitting_parameters_all.csv", index_col=[0])
fit_params.groupby("curve type")[["b", "r_squared"]].describe().transpose()

fit_params[["curve type", "b"]].groupby("curve type").hist(range=(0, 2), bins=20)

# histograms of r squared
sns.displot(x='r_squared', col="curve type", data=fit_params[fit_params["r_squared"] > 0])

sns.catplot(x='curve type', y='r_squared', data=fit_params[fit_params["r_squared"] > 0], kind='boxen')

sns.catplot(x='curve type', y='r_squared', data=fit_params[fit_params["r_squared"] > 0],
            hue="value type", jitter=0.3, alpha=0.7)

fit_params[fit_params["r_squared"] > 0].groupby("curve type")["r_squared"].mean()

fit_params.groupby("curve type")["r_squared"].min()

fig, ax = plt.subplots(1, 1, figsize=(8,8))

type_x = "logistic"
type_y = "softplus"

merged_df = fit_params[fit_params["curve type"] == type_x].merge(fit_params[fit_params["curve type"] == type_y], on="technology")
merged_df.plot.scatter(x="b_x", y="b_y", ax=ax, alpha=0.5)
xy_max = 1.5
ax.plot([0, xy_max], [0, xy_max])
ax.set_ylim([0, xy_max])
ax.set_xlim([0, xy_max])
ax.set_xlabel(type_x)
ax.set_ylabel(type_y)

fig, ax = plt.subplots(1, 1, figsize=(8,8))

type_x = "logistic"
type_y = "exponential"

merged_df = fit_params[fit_params["curve type"] == type_x].merge(fit_params[fit_params["curve type"] == type_y], on="technology")
merged_df.plot.scatter(x="r_squared_x", y="r_squared_y", ax=ax, alpha=0.5)
xy_min = 0.9
ax.plot([xy_min, 1], [xy_min, 1])
ax.set_ylim([xy_min, 1])
ax.set_xlim([xy_min, 1])
ax.set_xlabel(type_x)
ax.set_ylabel(type_y)

sns.catplot(x='curve type', y='r_squared', data=fp_types[fp_types["r_squared"] > 0],
            hue="value type 2", jitter=0.3)

sns.catplot(x='curve type', y='r_squared', data=fp_types[fp_types["r_squared"] > 0],
            hue="technology type", jitter=0.3)

sns.catplot(x='curve type', y='r_squared', data=fp_types[fp_types["r_squared"] > 0],
            hue="scale", jitter=0.3)

sns.catplot(x='curve type', y='b', data=fp_types[fp_types["r_squared"] > 0],
            hue="scale", jitter=0.3)
plt.ylim([0, 2])

#Table 2: adj r2 for each time series
#columns: model forms
#rows: either each time series OR average for metrics (cumulative vs annual capacity)
#data: r2

#Table 3: model parameters
#columns: model forms
#rows: variance of each parameter (one row each for growth rate, saturation, inflection year)

