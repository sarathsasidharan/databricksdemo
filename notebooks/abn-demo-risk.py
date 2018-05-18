# Databricks notebook source
import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import KFold 
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.metrics import roc_auc_score 
from sklearn.feature_selection import SelectFromModel 
from sklearn.feature_selection import SelectPercentile 
from sklearn.feature_selection import f_classif 
from sklearn.feature_selection import SelectKBest 
from sklearn.pipeline import Pipeline 

# COMMAND ----------

# Mount Blob Storage onto DBFS
dbutils.fs.mount(
  source = "wasbs://abndemo@abndemo.blob.core.windows.net",
  mount_point = "/mnt/abn",
  extra_configs = {"fs.azure.account.key.abndemo.blob.core.windows.net": "TkuHdQxXlNxQ0wKnNNJMcFTVYw1CRIeBmSdEKRuGDxQDfDk748n2MUVotxZw/+tFWY7P39SzIbFsw9ytbx3J/Q=="})

# COMMAND ----------

# Read in CSV Files
data_NLD_Q_df = pd.read_csv("/dbfs/mnt/abn/data_NLD_Q.csv",index_col='time', parse_dates=['time'])
data_DEU_Q_df = pd.read_csv("/dbfs/mnt/abn/data_DEU_Q.csv",index_col='time', parse_dates=['time'])
data_USA_Q_df = pd.read_csv("/dbfs/mnt/abn/data_USA_Q.csv",index_col='time', parse_dates=['time'])
data_NLD_M_df = pd.read_csv("/dbfs/mnt/abn/data_NLD_M.csv",index_col='time', parse_dates=['time'])
data_DEU_M_df = pd.read_csv("/dbfs/mnt/abn/data_DEU_M.csv",index_col='time', parse_dates=['time'])
data_USA_M_df = pd.read_csv("/dbfs/mnt/abn/data_USA_M.csv",index_col='time', parse_dates=['time'])  

# COMMAND ----------

data_USA_M_df.head(5)

# COMMAND ----------

#    Concatenate data 
data_Q = pd.concat([data_NLD_Q_df,data_DEU_Q_df,data_USA_Q_df],axis=1) 
data_M = pd.concat([data_NLD_M_df,data_DEU_M_df,data_USA_M_df],axis=1)     

# COMMAND ----------

#    Change dates to end of month/quarter 
data_Q.index = pd.to_datetime(data_Q.index) + pd.offsets.MonthEnd(3)   
data_M.index = pd.to_datetime(data_M.index) + pd.offsets.MonthEnd(1) 

# COMMAND ----------

#Transform quarterly data into monthly data 
data_Q_M = data_Q.resample('M').pad()

# COMMAND ----------

#Link datasets 
data_final = pd.concat([data_M, data_Q_M],axis=1)

# COMMAND ----------

#Define features 
data_final.columns = ['F_' + c for c in data_final.columns] 

# COMMAND ----------

#Define unemployment target 
data_final['T_LFHUTTTT_ST_NLD_M'] = data_final['F_LFHUTTTT_ST_NLD_M'].shift(-12)  

# COMMAND ----------

#   Remove observations without a target 
data_final = data_final[~data_final['T_LFHUTTTT_ST_NLD_M'].isnull()] 

# COMMAND ----------

 #   Define binary target 
data_final['T_LFHUTTTT_ST_NLD_M_diff'] = data_final['T_LFHUTTTT_ST_NLD_M'] - data_final['F_LFHUTTTT_ST_NLD_M']     
data_final['T_LFHUTTTT_ST_NLD_M_bin'] = np.where(data_final['T_LFHUTTTT_ST_NLD_M_diff']>0, 1, 0)  

# COMMAND ----------

#   Remove features with nan values   
data_final = data_final.dropna(axis=1) 

# COMMAND ----------

#Drop non binary targets 
columns_to_drop = ['T_LFHUTTTT_ST_NLD_M_diff', 'T_LFHUTTTT_ST_NLD_M'] 

# COMMAND ----------

data_final = data_final[data_final.columns.drop(columns_to_drop)] 

# COMMAND ----------

#   Drop non binary targets 
features = [c for c in data_final.columns if c[0:2]=='F_'] 
X = data_final[features] 
y = data_final['T_LFHUTTTT_ST_NLD_M_bin'] 

# COMMAND ----------

#   Drop non binary targets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# COMMAND ----------

#   Select above a certain threshold according to gradient boosting 
select = SelectFromModel(GradientBoostingClassifier(), threshold=0.015) 

# COMMAND ----------

#   Use Gradient boosting model 
model = GradientBoostingClassifier(n_estimators=100) 

# COMMAND ----------

#   selection and modelling together in pipeline 
pipeline = Pipeline([('select', select), ('model', model)]) 

# COMMAND ----------

#   perform grid search 
parameter_grid = {} 
grid_cv = GridSearchCV(model, parameter_grid,scoring='roc_auc', cv=KFold(n_splits=2)) 
grid_cv.fit(X_train, y_train) 
np.array(grid_cv.cv_results_['mean_test_score']) 

# COMMAND ----------

#   test gini 
y_pred_prob_test = grid_cv.predict_proba(X_test)[:,1] 
2*roc_auc_score(y_test, y_pred_prob_test)-1 

# COMMAND ----------

y_pred_prob_test


# COMMAND ----------

data_Q_M.describe()

# COMMAND ----------

data_Q_M.dtypes

# COMMAND ----------

