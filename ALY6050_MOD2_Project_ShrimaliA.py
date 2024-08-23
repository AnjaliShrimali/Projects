#!/usr/bin/env python
# coding: utf-8

# # Project-2

# ## Part_1:
# 
# ### (i) Perform a simulation of 10,000 benefit-cost ratios for Dam 1 project and 10,000 such simulations for Dam 2 project. Note that the two simulations should be independent of each other. Let these two ratios be denoted by 𝛼1 and 𝛼2 for the dams 1 and 2 projects respectively.
# 

# In[49]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulation parameters for Dam #1
min_val_dam1 = [10, 20, 5, 15, 10, 30]
likely_val_dam1 = [30, 50, 20, 40, 30, 70]
max_val_dam1 = [50, 80, 40, 60, 50, 100]

# Simulation parameters for Dam #2
min_val_dam2 = [15, 25, 10, 20, 15, 40]
likely_val_dam2 = [40, 60, 30, 50, 40, 80]
max_val_dam2 = [60, 90, 50, 70, 60, 120]

#Perform simulation for Dam #1
random_val_dam1 = np.random.triangular(
    min_val_dam1, likely_val_dam1, max_val_dam1,
    size=(10000, len(min_val_dam1)))

#total benefits for dam_1
total_benefits_dam1 = np.sum(random_val_dam1, axis=1)
print("total_benefits_dam1:", total_benefits_dam1)

#total cost for dam_1
total_costs_dam1 = np.random.uniform(100, 500, 10000)
print("total_costs_dam1:", total_costs_dam1)

#benefit cost ratios for dam_1 
alpha1_dam = total_benefits_dam1 / total_costs_dam1
print("alpha1_dam:", alpha1_dam)

#Perform simulation for Dam #2
random_values_dam2 = np.random.triangular(
    min_values_dam2, likely_values_dam2, max_values_dam2,
    size=(10000, len(min_values_dam2)))

#total benefits for dam_2
total_benefits_dam2 = np.sum(random_values_dam1, axis=1)
print("total_benefits_dam2:", total_benefits_dam2)

#total cost for dam_2
total_costs_dam2 = np.random.uniform(100, 500, 10000)
print("total_costs_dam2:", total_costs_dam2)

# benefit cost ratios for dam_2.
alpha2_dam = total_benefits_dam2 / total_costs_dam2
print("alph2_dam:", alpha2_dam)


# 
# ### (ii) Construct both a tabular and a graphical frequency distribution for 𝛼1 and 𝛼2 separately (a tabular and a graphical distribution for 𝛼1 , and a tabular and a graphical distribution for 𝛼2 - a total of 4 distributions).

# In[39]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(alpha1_dam, bins=50, edgecolor='black')
plt.title('Benefit-Cost Ratio Distribution for Dam #1')
plt.xlabel('Benefit-Cost Ratio')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(alpha2_dam, bins=50, edgecolor='black')
plt.title('Benefit-Cost Ratio Distribution for Dam #2')
plt.xlabel('Benefit-Cost Ratio')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# ### For each of the two dam projects, perform the necessary calculations in order to complete the following table. Python users should display the table as a “data frame”. Remember to create two such tables – one table for Dam #1 and another table for Dam #2.

# In[40]:


df_dam1 = pd.DataFrame({
    'Total Benefits': total_benefits_dam1,
    'Total Costs': total_costs_dam1,
    'Benefit-Cost Ratio': alpha1_dam
})

print("df_dam1:",
     df_dam1)

df_dam2 = pd.DataFrame({
    'Total Benefits': total_benefits_dam2,
    'Total Costs': total_costs_dam2,
    'Benefit-Cost Ratio': alpha2_dam
})
print("df_dam2:",
     df_dam2)


# ### (iii) For each of the two dam projects, perform the necessary calculations in order to complete the following table. Python users should display the table as a “data frame”. Remember to create two such tables – one table for Dam 1 and another table for Dam 2.

# In[41]:


# extracting values for dam1
observed_stats_dam1 = df_dam1[['Total Benefits', 'Total Costs', 'Benefit-Cost Ratio']].describe().loc[['mean', 'std']].values.flatten()

theoretical_stats_dam1 = np.array([np.mean(total_benefits_dam1), np.std(total_benefits_dam1),
                                   np.mean(total_costs_dam1), np.std(total_costs_dam1),
                                   np.mean(alpha1_dam), np.std(alpha1_dam)])
# creating a table
table_dam1 = pd.DataFrame({
    'Statistic': ['Mean of Total Benefits', 'SD of Total Benefits', 'Mean of Total Costs', 'SD of Total Costs', 'Mean of Benefit-Cost Ratio', 'SD of Benefit-Cost Ratio'],
    'Observed': observed_stats_dam1,
    'Theoretical': theoretical_stats_dam1
})
print(table_dam1)


# In[42]:


# extracting values for dam2
observed_stats_dam2 = df_dam2[['Total Benefits', 'Total Costs', 'Benefit-Cost Ratio']].describe().loc[['mean', 'std']].values.flatten()

theoretical_stats_dam2 = np.array([np.mean(total_benefits_dam2), np.std(total_benefits_dam2),
                                   np.mean(total_costs_dam2), np.std(total_costs_dam2),
                                   np.mean(alpha2_dam), np.std(alpha2_dam)])
# creating a table
table_dam2 = pd.DataFrame({
    'Statistic': ['Mean of Total Benefits', 'SD of Total Benefits', 'Mean of Total Costs', 'SD of Total Costs', 'Mean of Benefit-Cost Ratio', 'SD of Benefit-Cost Ratio'],
    'Observed': observed_stats_dam2,
    'Theoretical': theoretical_stats_dam2
})
print(table_dam2)


# ## Part-2
# 
# ### Use your observation in Question (ii) of Part 1 to select a theoretical probability distribution that, in your judgement, is a good fit for the distribution of 𝛼1 . Next, use the Chi-squared Goodness-of-fit test to verify whether your selected distribution was a good fit for the distribution of 𝛼1 . Describe the rational for your choice of the probability distribution and a description of the outcomes of your Chi-squared test in your report. In particular, indicate the values of the Chi-squared test statistic and the P-value of your test in your report, and interpret those values.

# In[44]:


from scipy.stats import norm
from scipy.stats import chi2
import seaborn as sns

# fitting a normal distribution
mu, std = norm.fit(alpha1_dam)

plt.hist(alpha1_dam, bins=50, edgecolor='black', density=True, alpha=0.7)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title('Benefit-Cost Ratio Distribution for Dam #1')
plt.xlabel('Benefit-Cost Ratio')
plt.ylabel('Density')
plt.show()


# In[45]:


observed, bins = np.histogram(alpha1_dam, bins=50, density=True)
midpoints = (bins[:-1] + bins[1:]) / 2
expected = norm.pdf(midpoints, mu, std) * np.sum(np.diff(bins))

df = len(midpoints) - 1

# chi-square
chi_statistic = np.sum((observed - expected)**2 / expected)
# p-value
p_value = 1 - chi2.cdf(chi_statistic, df)

chi_squared_results = pd.DataFrame({
    'Statistic': ['Chi-squared Test Statistic', 'P-value'],
    'Value': [chi_statistic, p_value]})
chi_squared_results


# ## Part-3
# 
# ### Use the results of your simulations and perform the necessary calculations in order to complete the table below. Python users should display the table as a “data frame”.

# In[46]:


from scipy.stats import skew

# statistics for dam1

stats_dam1 = {
    'Minimum': np.min(alpha1_dam),
    'Maximum': np.max(alpha1_dam),
    'Mean': np.mean(alpha1_dam),
    'Median': np.median(alpha1_dam),
    'Variance': np.var(alpha1_dam),
    'Standard Deviation': np.std(alpha1_dam),
    'Skewness': skew(alpha1_dam),
    'P(>2)': np.mean(alpha1_dam > 2),
    'P(>1.8)': np.mean(alpha1_dam > 1.8),
    'P(>1.5)': np.mean(alpha1_dam > 1.5),
    'P(>1.2)': np.mean(alpha1_dam > 1.2),
    'P(>1)': np.mean(alpha1_dam > 1)
}

# statisticts for dam2

stats_dam2 = {
    'Minimum': np.min(alpha2_dam),
    'Maximum': np.max(alpha2_dam),
    'Mean': np.mean(alpha2_dam),
    'Median': np.median(alpha2_dam),
    'Variance': np.var(alpha2_dam),
    'Standard Deviation': np.std(alpha2_dam),
    'Skewness': skew(alpha2_dam),
    'P(>2)': np.mean(alpha2_dam > 2),
    'P(>1.8)': np.mean(alpha2_dam > 1.8),
    'P(>1.5)': np.mean(alpha2_dam > 1.5),
    'P(>1.2)': np.mean(alpha2_dam > 1.2),
    'P(>1)': np.mean(alpha2_dam > 1)
}


# In[47]:


# creating dataframe for dataframe 1

df_stats_dam1 = pd.DataFrame(stats_dam1, index=['Dam 1'])
df_stats_dam1


# In[48]:


# creating dataframe for dataframe 2

df_stats_dam2 = pd.DataFrame(stats_dam2, index=['Dam 2'])
df_stats_dam2


# In[50]:


# estimating probability that alpha1 > alpha2

probability_alpha1_greater_than_alpha2 = np.mean(alpha1_dam > alpha2_dam)
print("Probability that alpha1 is greater than alpha2:", probability_alpha1_greater_than_alpha2)

