# -*- coding: utf-8 -*-
"""
PREDICTING THE LIKELIHOOD OF E-SIGNING A LOAN BASED ON FINANCIAL HISTORY
Rahul Chellani

PROBLEM STATEMENT:
This data is for a 'Fintech' company that specializes on loans. It offers low APR loans to applicants
based on their financial habits, as almost all lending companies do. This company has partnered with
a P2P lending marketplace that provides real time leads(loan-applicants). The number of conversions
from these leads are satisfactory.
The company tasks you with creating a model that predicts whether or not these leads will complete the
electronic signature phase of the loan application (a.k.a. e_signed). The company seeks to leverage
this model to identify less 'quality' applicants (e.g. those who are not responding to the onboarding process),
and experiment with giving them different onboarding screens.
Because the applicants arrived through a marketplace, we have access to their financial data before
the onboarding process begins. This data includes personal information like age and time employed, as well as
other financial metrics. Our company utilizes these financial data points to create risk scores based on
many different risk factors.
We are given the set of scores from algorithms built by the finance and engineering teams. Furthermore,
the marketplace itself provides us with their own lead quality scores.

OBJECTIVE:
Develop a model to predict the quality applicants, that are ones who reach a key part 
of loan application process.

Part-1 of Project: EDA
"""

##### IMPORTING DATA #####

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('financial_data.csv')


##### EDA #####

dataset.head()
dataset.columns
dataset.describe()
dataset.info()

# Cleaning Data
# Removing NaN
dataset.isna().any()

# For plotting
dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedule', 'e_signed'])
dataset2.head()

# Histograms
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(dataset2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])

    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    
    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Checking the correlation with response variable
dataset2.corrwith(dataset.e_signed).plot.bar(
        figsize = (20, 10), title = "Correlation with E Signed", fontsize = 15,
        rot = 45, grid = True)

# Correlation Matrix to check the correlation between the variables
sns.set(style="white", font_scale=2)
# Compute the correlation matrix
corr = dataset2.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize = 40)
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
