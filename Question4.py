# %% [markdown]
# Create functions for your two pipelines that produces the train and test datasets. The end result should be a series of functions that can be called to produce the train and test datasets for each of 
# your two problems that includes all the data prep steps you took. This is essentially creating a DAG for your data prep steps. Imagine you will need to do this for multiple problems in the future so creating functions that can be reused is important. 
# You don’t need to create one full pipeline function that does everything but rather a series of smaller functions that can be called in sequence to produce the final datasets. Use your judgement on how to break up the functions.
# %%# %% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data

# %%
def process_cc_data(url):
    # Load the dataset
    cc_institution = pd.read_csv(url)
    
    # collapse factor levels as needed
    cc_institution['control'] = cc_institution['control'].replace({
        'Private not-for-profit': 'Private', 
        'Private for-profit': 'Private' })
    
    # categorical columns should be type 'category'
    cols = ["control"]
    cc_institution[cols] = cc_institution[cols].astype('category')

    # Normalize the continuous variables
    numeric_cols = ['student_count', 'aid_value']
    # remove missing values
    cc_institution = cc_institution.dropna(subset=numeric_cols)
    cc_institution[numeric_cols] = MinMaxScaler().fit_transform(cc_institution[numeric_cols])
    
    # Create target variable if needed
    # turn 'Public' into 1 and 'Private' into 0
    cc_institution['target'] = cc_institution['control'].map({'Public': 1, 'Private': 0}).astype('int')
    
    # Calculate the prevalence of the target variable
    prevalence = (cc_institution.target.value_counts()[1] / len(cc_institution.target))
    cc_institution['target'] = cc_institution['target'].astype('category')
    # Drop unneeded variables
    # we drop all columns but our target and two features were focusing on
    cc_dt = cc_institution[['target', 'student_count', 'aid_value']].copy()
    
    # Partition data (Using 60% for Train as per your Step 8)
    Train, Test = train_test_split(cc_dt, train_size=0.6, stratify=cc_dt.target, random_state=42)
    Tune, Test = train_test_split(Test, train_size=0.5, stratify=Test.target, random_state=42)
    
    return Train, Tune, Test, prevalence
# %%
def process_placement_data(url):
    # Load the dataset
    placement_data = pd.read_csv(url)
    
    # categorical columns should be type 'category'
    cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]
    placement_data[cols] = placement_data[cols].astype('category')
    
    # Normalize the continuous variables
    # - To address our problem our continuous variables are 'ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p'
    numeric_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
    # Remove missing values so scaling works
    placement_data = placement_data.dropna(subset=numeric_cols)
    # Min-Max scaling to put numeric_cols between 0 and 1 - essentislly a probability
    placement_data[numeric_cols] = MinMaxScaler().fit_transform(placement_data[numeric_cols])
    
    # One-hot encoding factor variables
    category_list = list(placement_data.select_dtypes('category'))
    placement_1h = pd.get_dummies(placement_data, columns=category_list)
    
    # Create target variable if needed
    placement_1h['target'] = placement_1h['status_Placed'].astype(int)
    
    # Calculate the prevalence of the target variable
    # Prevalence tells us what % of the data is our positive class (Placed)
    prevalence = (placement_1h.target.value_counts()[1] / len(placement_1h.target))
    
    # Drop unneeded variables
    # we drop all columns except our target and features were looking at
    placement_dt = placement_1h[['target', 'ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'workex_Yes']].copy()
    
    # Create the necessary data partitions (Train, Tune, Test)
    # First split: Train (60%) and the rest (40%)
    Train, Test = train_test_split(placement_dt, train_size=0.6, stratify=placement_dt.target, random_state=42)
    # second split: splitting 'test' into Tune and Test (so 50/50 of the remaining 40%)
    Tune, Test = train_test_split(Test, train_size=0.5, stratify=Test.target, random_state=42)
    
    return Train, Tune, Test, prevalence

# %%
# Call the College Completion function
cc_train, cc_tune, cc_test, cc_prev = process_cc_data('cc_institution_details.csv')

# Call the Placement function
place_train, place_tune, place_test, place_prev = process_placement_data('Placement_Data_Full_Class.csv')

# Print one result to prove it worked
print(f"College Train set size: {len(cc_train)}")
print(f"Placement Train set size: {len(place_train)}")
# %%
