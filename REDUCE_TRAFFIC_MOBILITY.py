#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


road = pd.read_csv("road-accidentss.csv")


# In[3]:


road


# In[6]:


# Save the number of rows columns as a tuple
rows_and_cols = road.shape
print('There are {} rows and {} columns.\n'.format(
    rows_and_cols[0], rows_and_cols[1]))
# Generate an overview of the DataFrame
road_information = road.info()
print(road_information)


# In[8]:


# Display the last five rows of the DataFrame
road.tail()


# 3.create a textual and graphical summary of the data

# In[10]:


# import seaborn and make plots appear inline
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Compute the summary statistics of all columns in the `road` DataFrame
sum_stat_car = road.describe()
print(sum_stat_car)
# Create a pairwise scatter plot to explore the data
sns.pairplot(sum_stat_car)


# 4. Quantify the association of features and accidents

# In[11]:


# Compute the correlation coefficent for all column pairs
corr_columns = road.corr()
corr_columns


# # 5. Fit a multivariate linear regression

# In[17]:


# Import the linear model function from sklearn
from sklearn import linear_model
# Create the features and target DataFrames
features = road[['perc_fatl_speed', 'perc_fatl_alcohol', 'perc_fatl_1st_time']]
target = road['drvr_fatl_col_bmiles']
# Create a linear regression object
reg = linear_model.LinearRegression()
# Fit a multivariate linear regression model
reg.fit(features, target)
# Retrieve the regression coefficients
fit_coef = reg.coef_
fit_coef


# # 6. Perform PCA on standardized data

# In[19]:


import numpy as np

# Standardize and center the feature columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Import the PCA class function from sklearn
from sklearn.decomposition import PCA
pca = PCA()

# Fit the standardized data to the pca
pca.fit(features_scaled)
# Plot the proportion of variance explained on the y-axis of the bar plot
import matplotlib.pyplot as plt
plt.bar(range(1, pca.n_components_ + 1),  pca.explained_variance_ratio_)
plt.xlabel('Principal component #')
plt.ylabel('Proportion of variance explained')
plt.xticks([1, 2, 3])

# Compute the cumulative proportion of variance explained by the first two principal components
two_first_comp_var_exp = pca.explained_variance_ratio_[0].cumsum()[0] + pca.explained_variance_ratio_[1].cumsum()[0]
print("The cumulative variance of the first two principal components is {}".format(
    round(two_first_comp_var_exp, 5)))


# # 7. Visualize the first two principal components

# In[20]:


# Transform the scaled features using two principal components
pca = PCA(n_components = 2)
p_comps = pca.fit_transform(features_scaled)

# Extract the first and second component to use for the scatter plot
p_comp1 = p_comps[:, 0]
p_comp2 = p_comps[:, 1]

# Plot the first two principal components in a scatter plot
plt.scatter(p_comp1, p_comp2)


# # 8. Find clusters of similar states in the data

# In[21]:


# Import KMeans from sklearn
from sklearn.cluster import KMeans

# A loop will be used to plot the explanatory power for up to 10 KMeans clusters
ks = range(1, 10)
inertias = []
for k in ks:
    # Initialize the KMeans object using the current number of clusters (k)
    km = KMeans(n_clusters=k, random_state=8)
    # Fit the scaled features to the KMeans object
    km.fit(features_scaled)
    # Append the inertia for `km` to the list of inertias
    inertias.append(km.inertia_)
    
# Plot the results in a line plot
plt.plot(ks, inertias, marker='o')


# # 9. KMeans to visualize clusters in the PCA scatter plot

# In[22]:


# Create a KMeans object with 3 clusters, use random_state=8 
km = KMeans(n_clusters = 3, random_state = 8)

# Fit the data to the `km` object
km.fit(features_scaled)

# Create a scatter plot of the first two principal components
# and color it according to the KMeans cluster assignment 
plt.scatter(p_comps[:, 0], p_comps[:, 1], c = km.labels_)


# # 10. Visualize the feature differences between the clusters

# In[25]:


# Create a new column with the labels from the KMeans clustering
road['cluster'] = km.labels_

# Reshape the DataFrame to the long format
melt_car = pd.melt(road, id_vars = ['cluster'], var_name ='measurement', value_name = 'percent', 
                                                   value_vars =['perc_fatl_speed', 'perc_fatl_alcohol', 'perc_fatl_1st_time'])

# Create a violin plot splitting and coloring the results according to the km-clusters
sns.violinplot(melt_car['percent'], melt_car['measurement'], hue = melt_car['cluster'])


# # 11. Compute the number of accidents within each cluster

# In[38]:


# Read in the new dataset
miles_drivens = pd.read_csv('miles-drivens.csv', sep=',')

display(miles_drivens.head())


# In[40]:


# Merge the `road` DataFrame with the `miles_drivens` DataFrame
road_miles = road.merge(miles_drivens, on='state')


# In[41]:


# Create a new column for the number of drivers involved in fatal accidents
road_miles['num_drvr_fatl_col'] = (road_miles['drvr_fatl_col_bmiles'] * road_miles['million_miles_annually']) / 1000

display(road_miles.head())


# In[42]:


# Calculate the number of states in each cluster and their 'num_drvr_fatl_col' mean and sum.
count_mean_sum = road_miles.groupby('cluster')['num_drvr_fatl_col'].agg(['count', 'mean', 'sum'])
count_mean_sum


# In[43]:


# Create a barplot of the total number of accidents per cluster
sns.barplot(x='cluster', y='num_drvr_fatl_col', data=road_miles, estimator=sum, ci=None)


# In[ ]:


12. Make a decision when there is no clear right choice
cluster_sum=.......

