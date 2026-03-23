import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1) Load dataset
housing = pd.read_csv("housing.csv")


#  2) Create a stratified test set
housing['income_cat'] = pd.cut(housing['median_income'], 
                               bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop('income_cat', axis=1) # WE WILL WORK ON THIS DATA
    strat_test_set = housing.loc[test_index].drop('income_cat', axis=1) # SET ASIDE THIS DATA


#  3) We wil now work on copy of training data
housing = strat_train_set.copy()

#  4) Seperate features and labels
housing_label = housing['median_house_value'].copy()
housing = housing.drop('median_house_value', axis=1)

#  5) Seperate categorical and numerical column
num_attribs = housing.drop('ocean_proximity', axis=1).columns.tolist()
cat_attribs = ['ocean_proximity']


# 6) Creating Pipeline

# for numerical values
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# for categorical values
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown='ignore'))
])


# 7) Creating Full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs), 
    ("cat_attribs", cat_pipeline, cat_attribs)
])

# 8) Transforming the data
housing_prepared = full_pipeline.fit_transform(housing)

# 9) Train the model

# a) Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_label)
lin_pre = lin_reg.predict(housing_prepared)
# lin_rmse = root_mean_squared_error(housing_label, lin_pre)
lin_rmses = -cross_val_score(lin_reg, housing_prepared, housing_label, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(lin_rmses).describe())
# print(f"The root means squared error for linear reggresion is {lin_rmses}")


# b) Decision Tree Model
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_label)
dec_pre = dec_reg.predict(housing_prepared)
# dec_rmse = root_mean_squared_error(housing_label, dec_pre)
dec_rmses = -cross_val_score(dec_reg, housing_prepared, housing_label, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(dec_rmses).describe())
# print(f"The root means squared error for Decision Tree is {dec_rmses}")


# c) Random Forest Model
random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(housing_prepared, housing_label)
random_forest_pre = random_forest_reg.predict(housing_prepared)
# random_forest_rmse = root_mean_squared_error(housing_label, random_forest_pre)
random_forest_rmses = -cross_val_score(random_forest_reg, housing_prepared, housing_label, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(random_forest_rmses).describe())
# print(f"The root means squared error for Random Forest is {random_forest_rmse}")
