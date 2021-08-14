# John Manzo
# DSC530-301T
# August 14, 2021

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

# import dataset, convert to dataframe
data_csv = pd.read_csv('data_JManzo.csv')
dataset = pd.DataFrame(data_csv)

# returns the count of +-1.5(IQR) outliers
def outlierCount(x):
    count = 0
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    fence = iqr * 1.5
    for outlier in x:
        if outlier < q1 - fence or outlier > q3 + fence:
            count += 1
    return count

# dataset variables
volatile_acidity = dataset[['volatile acidity']]
citric_acid = dataset[['citric acid']]
chlorides = dataset[['chlorides']]
sulphates = dataset[['sulphates']]
alcohol = dataset[['alcohol']]
quality = dataset[['quality']]

# VARIABLE 1: Volatile Acidity
# histogram
plt.hist(volatile_acidity)
plt.title('Volatile Acidity (dm^3)')
plt.show()
# boxplot
plt.boxplot(volatile_acidity)
plt.title('Volatile Acidity (dm^3)')
plt.show()
# summary statistics
print(f'Volatile Acidity Count: {len(volatile_acidity)}')
print(f'Volatile Acidity Min: {volatile_acidity.min()}')
print(f'Volatile Acidity Mean: {volatile_acidity.mean()}')
print(f'Volatile Acidity Median: {volatile_acidity.median()}')
print(f'Volatile Acidity Mode: {volatile_acidity.mode()}')
print(f'Volatile Acidity Max: {volatile_acidity.max()}')
print(f'Volatile Acidity Variance: {volatile_acidity.var()}')
print(f'Volatile Acidity Std Dev: {volatile_acidity.std()}')
print(f'Volatile Acidity Skewness: {volatile_acidity.skew()}')
# covert dataframe to list of values for outlierCount()'s if statement
dataset_list = volatile_acidity.values.tolist()
# calculate outlier count and percentage
outlier_count = outlierCount(dataset_list)
outlier_percentage = round(
    outlierCount(dataset_list) / len(dataset_list) * 100, 2)
print(f'Volatile Acidity Outlier Count: {outlier_count}')
print(f'Volatile Acidity Outlier Percentage: {outlier_percentage}\n')

# VARIABLE 2: Citric Acid
# histogram
plt.hist(citric_acid)
plt.title('Citric Acid (dm^3)')
plt.show()
# boxplot
plt.boxplot(citric_acid)
plt.title('Citric Acid (dm^3)')
plt.show()
# summary statistics
print(f'Citric Acid Count: {len(citric_acid)}')
print(f'Citric Acid Min: {citric_acid.min()}')
print(f'Citric Acid Mean: {citric_acid.mean()}')
print(f'Citric Acid Median: {citric_acid.median()}')
print(f'Citric Acid Mode: {citric_acid.mode()}')
print(f'Citric Acid Max: {citric_acid.max()}')
print(f'Citric Acid Variance: {citric_acid.var()}')
print(f'Citric Acid Std Dev: {citric_acid.std()}')
print(f'Citric Acid Skewness: {citric_acid.skew()}')
# covert dataframe to list of values for outlierCount()'s if statement
dataset_list = citric_acid.values.tolist()
# calculate outlier count and percentage
outlier_count = outlierCount(dataset_list)
outlier_percentage = round(
    outlierCount(dataset_list) / len(dataset_list) * 100, 2)
print(f'Citric Acid Outlier Count: {outlier_count}')
print(f'Citric Acid Outlier Percentage: {outlier_percentage}\n')

# VARIABLE 3: Chlorides
# histogram
plt.hist(chlorides)
plt.title('Chlorides (dm^3)')
plt.show()
# boxplot
plt.boxplot(chlorides)
plt.title('Chlorides (dm^3)')
plt.show()
# summary statistics
print(f'Chlorides Count: {len(chlorides)}')
print(f'Chlorides Min: {chlorides.min()}')
print(f'Chlorides Mean: {chlorides.mean()}')
print(f'Chlorides Median: {chlorides.median()}')
print(f'Chlorides Mode: {chlorides.mode()}')
print(f'Chlorides Max: {chlorides.max()}')
print(f'Chlorides Variance: {chlorides.var()}')
print(f'Chlorides Std Dev: {chlorides.std()}')
print(f'Chlorides Skewness: {chlorides.skew()}')
# covert dataframe to list of values for outlierCount()'s if statement
dataset_list = chlorides.values.tolist()
# calculate outlier count and percentage
outlier_count = outlierCount(dataset_list)
outlier_percentage = round(
    outlierCount(dataset_list) / len(dataset_list) * 100, 2)
print(f'Chlorides Outlier Count: {outlier_count}')
print(f'Chlorides Outlier Percentage: {outlier_percentage}\n')

# VARIABLE 4: Sulphates
# histogram
plt.hist(sulphates)
plt.title('Sulphates (dm^3)')
plt.show()
# boxplot
plt.boxplot(sulphates)
plt.title('Sulphates (dm^3)')
plt.show()
# summary statistics
print(f'Sulphates Count: {len(sulphates)}')
print(f'Sulphates Min: {sulphates.min()}')
print(f'Sulphates Mean: {sulphates.mean()}')
print(f'Sulphates Median: {sulphates.median()}')
print(f'Sulphates Mode: {sulphates.mode()}')
print(f'Sulphates Max: {sulphates.max()}')
print(f'Sulphates Variance: {sulphates.var()}')
print(f'Sulphates Std Dev: {sulphates.std()}')
print(f'Sulphates Skewness: {sulphates.skew()}')
# covert dataframe to list of values for outlierCount()'s if statement
dataset_list = sulphates.values.tolist()
# calculate outlier count and percentage
outlier_count = outlierCount(dataset_list)
outlier_percentage = round(
    outlierCount(dataset_list) / len(dataset_list) * 100, 2)
print(f'Sulphates Outlier Count: {outlier_count}')
print(f'Sulphates Outlier Percentage: {outlier_percentage}\n')

# VARIABLE 5: Alcohol
# histogram
plt.hist(alcohol)
plt.title('Alcohol (%vol)')
plt.show()
# boxplot
plt.boxplot(alcohol)
plt.title('Alcohol (%vol)')
plt.show()
# summary statistics
print(f'Alcohol Count: {len(alcohol)}')
print(f'Alcohol Min: {alcohol.min()}')
print(f'Alcohol Mean: {alcohol.mean()}')
print(f'Alcohol Median: {alcohol.median()}')
print(f'Alcohol Mode: {alcohol.mode()}')
print(f'Alcohol Max: {alcohol.max()}')
print(f'Alcohol Variance: {alcohol.var()}')
print(f'Alcohol Std Dev: {alcohol.std()}')
print(f'Alcohol Skewness: {alcohol.skew()}')
# covert dataframe to list of values for outlierCount()'s if statement
dataset_list = alcohol.values.tolist()
# calculate outlier count and percentage
outlier_count = outlierCount(dataset_list)
outlier_percentage = round(
    outlierCount(dataset_list) / len(dataset_list) * 100, 2)
print(f'Alcohol Outlier Count: {outlier_count}')
print(f'Alcohol Outlier Percentage: {outlier_percentage}\n')

# VARIABLE 6: Quality
# histogram
plt.hist(quality)
plt.title('Quality (Scale 1-10)')
plt.show()
# boxplot
plt.boxplot(quality)
plt.title('Quality (Scale 1-10)')
plt.show()
# summary statistics
print(f'Quality Count: {len(quality)}')
print(f'Quality Min: {quality.min()}')
print(f'Quality Mean: {quality.mean()}')
print(f'Quality Median: {quality.median()}')
print(f'Quality Mode: {quality.mode()}')
print(f'Quality Max: {quality.max()}')
print(f'Quality Variance: {quality.var()}')
print(f'Quality Std Dev: {quality.std()}')
print(f'Quality Skewness: {quality.skew()}')
# covert dataframe to list of values for outlierCount()'s if statement
dataset_list = quality.values.tolist()
# calculate outlier count and percentage
outlier_count = outlierCount(dataset_list)
outlier_percentage = round(
    outlierCount(dataset_list) / len(dataset_list) * 100, 2)
print(f'Quality Outlier Count: {outlier_count}')
print(f'Quality Outlier Percentage: {outlier_percentage}\n')

# CREATE/COMPARE PMFs
# subset of observations w/ above average alcohol
dataset2 = dataset[lambda m: m['alcohol'] > 10.42]
# set probability value maps
prob1 = dataset['quality'].value_counts(normalize=True)  # n=1599
prob2 = dataset2['quality'].value_counts(normalize=True)  # n=683
# create comparison plots
pmf_plots = plt.subplots()
pmf_plots = sns.lineplot(x=prob1.index, y=prob1.values, color='blue',
                         drawstyle='steps-pre', label='All Alcohol')
pmf_plots = sns.lineplot(x=prob2.index, y=prob2.values, color='red',
                         drawstyle='steps-pre', label='High Alcohol')
pmf_plots.set(xlabel="Quality Score", ylabel="Probability")
plt.legend()
plt.title('PMF Comparison')
plt.show()

# CREATE/COMPARE CDFs
# create pdfs
counts1, bin_edges1 = np.histogram(dataset['volatile acidity'], density=False,
                                   bins=100)  # n=1599
counts2, bin_edges2 = np.histogram(dataset2['volatile acidity'], density=False,
                                   bins=100)  # n=683
pdf1 = counts1 / sum(counts1)
pdf2 = counts2 / sum(counts2)
# create cdfs
cdf1 = np.cumsum(pdf1)
cdf2 = np.cumsum(pdf2)
# create comparison plots
cdf_plots = plt.subplots()
cdf_plots = plt.plot(bin_edges1[1:], cdf1,color='blue', label='All Alcohol')
cdf_plots = plt.plot(bin_edges2[1:], cdf2, color='red', label='High Alcohol')
plt.xlabel("Volatile Acidity (dm^3)")
plt.ylabel("CDF")
plt.legend()
plt.title('CDF Comparison')
plt.show()

# ANALYTICAL DISTRIBUTION
# Normal model, linear scale
counts, bin_edges = np.histogram(dataset['citric acid'], density=False,
                                 bins=100)
pdf = counts / sum(counts)
cdf = np.cumsum(pdf)
mean = 0.270976
sigma = 0.194801
xs = np.linspace(0, 1, 1599)
ps = stats.norm.cdf(xs, mean, sigma)
plots = plt.subplots()
plots = plt.plot(xs, ps, label='Model', color='0.6')
plots = plt.plot(bin_edges[1:], cdf, color='blue', label='Citric Acid')
plt.xlabel("Citric Acid (dm^3)")
plt.ylabel("CDF")
plt.legend()
plt.title('Normal Distribution')
plt.show()

# CREATE TWO SCATTER PLOTS
# sulphates & alcohol
subset1 = dataset[['sulphates', 'alcohol']].dropna(how='any')
subset1 = subset1.values
sul = subset1[:, 0]
alc = subset1[:, 1]
cov_arr = np.stack((sul, alc), axis=0)
plt.scatter(sul, alc, s=20, alpha=0.075)
plt.xlabel("Sulphates (dm^3)")
plt.ylabel("Alcohol (%vol)")
plt.title('Scatter Plot One')
plt.show()
print('Corr', stats.pearsonr(sul, alc)[0])
print('SpearmanCorr', stats.spearmanr(sul, alc)[0])
print('Covariance', np.cov(sul, alc))
print()

# citric acid & chlorides
subset2 = dataset[['citric acid', 'chlorides']].dropna(how='any')
subset2 = subset2.values
cacid = subset2[:, 0]
chlor = subset2[:, 1]
cov_arr = np.stack((cacid, chlor), axis=0)
plt.scatter(cacid, chlor, s=20, alpha=0.075)
plt.xlabel("Citric Acid (dm^3)")
plt.ylabel("Chlorides (dm^3)")
plt.title('Scatter Plot Two')
plt.show()
print('Corr', stats.pearsonr(cacid, chlor)[0])
print('SpearmanCorr', stats.spearmanr(cacid, chlor)[0])
print('Covariance', np.cov(cacid, chlor))
print()

# TEST CORRELATION
# subsets with increasing values on n
test_set1 = dataset[['alcohol', 'quality']].dropna(how='any').loc[0:228]
test_set2 = dataset[['alcohol', 'quality']].dropna(how='any').loc[0:456]
test_set3 = dataset[['alcohol', 'quality']].dropna(how='any').loc[0:684]
test_set4 = dataset[['alcohol', 'quality']].dropna(how='any').loc[0:912]
test_set5 = dataset[['alcohol', 'quality']].dropna(how='any').loc[0:1140]
test_set6 = dataset[['alcohol', 'quality']].dropna(how='any').loc[0:1368]
test_set7 = dataset[['alcohol', 'quality']].dropna(how='any')
test_set1 = test_set1.values
test_set2 = test_set2.values
test_set3 = test_set3.values
test_set4 = test_set4.values
test_set5 = test_set5.values
test_set6 = test_set6.values
test_set7 = test_set7.values
# print n, correlation, p-value
test_data = test_set7[:, 0], test_set7[:, 1]
xs, ys = test_data
print(f'{len(test_set7)}\t{stats.pearsonr(abs(xs), abs(ys))}')
test_data = test_set6[:, 0], test_set6[:, 1]
xs, ys = test_data
print(f'{len(test_set6)}\t{stats.pearsonr(abs(xs), abs(ys))}')
test_data = test_set5[:, 0], test_set5[:, 1]
xs, ys = test_data
print(f'{len(test_set5)}\t{stats.pearsonr(abs(xs), abs(ys))}')
test_data = test_set4[:, 0], test_set4[:, 1]
xs, ys = test_data
print(f'{len(test_set4)}\t{stats.pearsonr(abs(xs), abs(ys))}')
test_data = test_set3[:, 0], test_set3[:, 1]
xs, ys = test_data
print(f'{len(test_set3)}\t{stats.pearsonr(abs(xs), abs(ys))}')
test_data = test_set2[:, 0], test_set2[:, 1]
xs, ys = test_data
print(f'{len(test_set2)}\t{stats.pearsonr(abs(xs), abs(ys))}')
test_data = test_set1[:, 0], test_set1[:, 1]
xs, ys = test_data
print(f'{len(test_set1)}\t{stats.pearsonr(abs(xs), abs(ys))}')
print()

# REGRESSION ANALYSIS
# simple linear
model = smf.ols('quality ~ alcohol', data=dataset)
results = model.fit()
print(results.summary())
# residual plots
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(results, 'alcohol', fig=fig)
plt.show()
print()

# multiple linear
model = smf.ols("quality ~ alcohol + Q('volatile acidity') + sulphates",
                data=dataset)
results = model.fit()
print(results.summary())
# residual plots
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(results, 'alcohol', fig=fig)
plt.show()
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(results, "Q('volatile acidity')", fig=fig)
plt.show()
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(results, 'sulphates', fig=fig)
plt.show()