# %% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data

# %% [markdown]
# ### Step one: Review these two datasets and brainstorm problems that could be addressed with the dataset. Identify a question for each dataset.
#Problems/ questions to consider:
#  - What is the target variable?
# ### College Completion Dataset Question:
# - Can we determine if an institution is Public or Private based on student count and how much financial aid this institution gives?

# %%
# Load the dataset
cc_institution = pd.read_csv('cc_institution_details.csv')

# %% [markdown]
# ### Step two: Work through the steps outlined in the examples to include the following elements:
# Write a generic question that this dataset could address.
# - Does institution type (Public or Private) have an impact on student count and how much financial aid this institution gives?
# ### What is a independent Business Metric for your problem? 
# - Independent Business Metric: Assuming that a specific institution type provides better aid over others along with having a larger student pool,
#can we predict whether an institution is Public or Private based on these metrics thereby helping prospective students make informed decisions?

# %% [markdown]
# ### Data preparation: 
# Correct variable type/class as needed
cc_institution.info()

# %%
# ### categorical columns should be type 'category'
cols = ["control"]
cc_institution[cols] = cc_institution[cols].astype('category')

# %% [markdown]
# Collapse factor levels as needed
# - since our question focuses on public vs private we collapse 'Private not-for-profit' and 'Private for-profit' into one category called 'Private'
cc_institution['control'] = cc_institution['control'].replace({
    'Private not-for-profit': 'Private', 
    'Private for-profit': 'Private'
}).astype('category') 

# %%
# verify collapse factor levels worked
print(cc_institution['control'].value_counts())

# %% [markdown]
# one-hot encoding factor variables
category_list = list(cc_institution.select_dtypes('category'))
cc_institution_encoded = pd.get_dummies(cc_institution, columns=category_list)
cc_institution_encoded.info()

# %% [markdown]
# Normalize the continuous variables
# continuous variables: 'student_count' and 'aid_value'; use Min-Max scaling to put them between 0 and 1
numeric_cols = ['student_count', 'aid_value']
# remove missing values
cc_institution = cc_institution.dropna(subset=numeric_cols)
cc_institution_encoded = cc_institution_encoded.dropna(subset=numeric_cols)

cc_institution_encoded[numeric_cols] = MinMaxScaler().fit_transform(cc_institution_encoded[numeric_cols])
print(cc_institution_encoded[numeric_cols])

# verify normalization of continuous variables worked
cc_institution_encoded[numeric_cols].describe()

# %% [markdown]
# Drop unneeded variables
cc_institution_clean = cc_institution_encoded.drop(['student_count', 'aid_value'], axis=1)
# we drop all columns but our target and two features were focusing on which are: control', 'student_count', 'aid_value'.
cc_clean = cc_institution[['control', 'student_count', 'aid_value']].copy()
cc_clean.info()

# %% [markdown]
# Create target variable if needed
# turn 'Public' into 1 and 'Private' into 0
cc_clean['target'] = cc_clean['control'].map({'Public': 1, 'Private': 0}).astype('category')

# %% [markdown]
# Calculate the prevalence of the target variable
# Prevalence tells us what % of the data is our positive class (Public)
prevalence = (cc_clean.target.value_counts()[1] / len(cc_clean.target))
print(f"Prevalence: {prevalence:.2%}")

# %% [markdown]
# Create the necessary data partitions (Train, Tune, Test)
# First split: Train (60%) and the rest (40%)
train, test = train_test_split(
    cc_clean, 
    train_size=0.6, 
    stratify=cc_clean.target,
    random_state=42)

# Verify the split sizes
print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")

# Second split: Split the 'test' into Tune and Test (50/50 of the remaining 40%)
tune, test = train_test_split(
    test, 
    train_size=0.5, 
    stratify=test.target,
    random_state=42)

print(f"Train size: {len(train)}, Tune size: {len(tune)}, Test size: {len(test)}")

################################################################################

# %% [markdown]
### Placement Dataset Question:
# Step two: Work through the steps outlined in the examples to include the following elements:
# Write a generic question that this dataset could address.
# - Are former work experience / grades the strongest indicator of a student's placement status?
# ##### What is a independent Business Metric for your problem? Think about the case study examples we have discussed in class.
# - Concerning student hireability rates, do factors such as work exprience or grades give some insight into a students potentail placement 
# assuming that higher placement rates result are a result of better academic performance and work experience, can we predict which students will struggle 
# to get placed based on their thereby helping students identify areas for improvement to increase placement chances?
# %%
# Load the dataset
placement_data = pd.read_csv('Placement_Data_Full_Class.csv')
# %% [markdown]
### Data preparation: 
# * Correct variable type/class as needed
placement_data.info()
# %%
#categorical columns should be type 'category'
cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]
placement_data[cols] = placement_data[cols].astype('category')
# %% [markdown]
# - Collapse factor levels as needed
# we collapse factor levels here, however no collapsing is needed for this dataset 
# %% [markdown]
# - One-hot encoding factor variables
category_list = list(placement_data.select_dtypes('category'))
placement_data_encoded = pd.get_dummies(placement_data, columns=category_list)
placement_data_encoded.info()
# %% [markdown]
# - Normalize the continuous variables
# - To address our problem our continuous variables are 'ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p'
numeric_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
# Remove missing values so scaling works
placement_data = placement_data.dropna(subset=numeric_cols)
placement_data_encoded = placement_data_encoded.dropna(subset=numeric_cols)
# Min-Max scaling to put numeric_cols between 0 and 1 - essentislly a probability
placement_data_encoded[numeric_cols] = MinMaxScaler().fit_transform(placement_data_encoded[numeric_cols])
print(placement_data_encoded[numeric_cols])
# Verify normalization
placement_data_encoded[numeric_cols].describe()
# %% [markdown]
# - Drop unneeded variables
placement_data_clean = placement_data_encoded.drop(numeric_cols, axis=1)
# %%
# we drop all columns except our target and features were looking at which are: 'status', 'ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p'
placement_clean = placement_data_encoded[['status_Placed', 'ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'workex_Yes']].copy()
placement_clean.info()
# %% [markdown]
# - Create target variable if needed
placement_clean['target'] = placement_clean['status_Placed'].astype('category')# %% [markdown]
# - Calculate the prevalence of the target variable
# Prevalence tells us what % of the data is our positive class (Placed)
prevalence = (placement_clean.target.value_counts()[1] / len(placement_clean.target))
print(f"Prevalence: {prevalence:.2%}")
# %% [markdown]
# - Create the necessary data partitions (Train, Tune, Test)
# ##### First split: Train (60%) and the rest (40%)
train, test = train_test_split(
    placement_clean, 
    train_size=0.6, 
    stratify=placement_clean.target,
    random_state=42)
# Verify the split sizes
print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")
# second split: splitting 'test' into Tune and Test (so 50/50 of the remaining 40%)
tune, test = train_test_split(
    test, 
    train_size=0.5, 
    stratify=test.target,
    random_state=42)
print(f"Train size: {len(train)}, Tune size: {len(tune)}, Test size: {len(test)}")

# %%
#Step three: What do your instincts tell you about the data. Can it address your problem, what areas/items are you worried about?
# College Completion Dataset:
# - I believe the data can address the problem, " Does institution type (Public or Private) have an impact on student count and how much financial aid this institution gives?" as it contains features that encompass student count, the amount of aid given by each institution as well as the institution type which is our target variable.
# Some concerns however are missing data within the student count and aid value features which are vital to address our question and hence lack of completeness could prove to be a problem. Additionally, the dataset is imbalanced with a higher prevalence of public institutions which may impact model performance. Another concern regarding the dataset is whether student count and aid value alone are measured consistenly across institutions and if they are sufficient to predict institution type accurately.

#Step three: What do your instincts tell you about the data. Can it address your problem, what areas/items are you worried about?
# Placement Dataset:
# - I believe the data can address the problem, "Are former work experience / grades the strongest indicator of a student's placement status?" as it contains features that encompass academic performance across various levels of education as well as work experience which are relevant to our target variable, placement status.
# Some concerns however are missing data within the continuous features which are vital to address our question and hence lack of completeness could prove to be a problem. Additionally, our dataset fails to consider external factors such as job market conditions, company recruitment cycles,etc. which could also impact placement status.