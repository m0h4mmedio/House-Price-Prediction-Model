import os
import joblib 
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

MODEL_FILE = "model.pkl"
PIPELINE_FILE = 'pipeline.pkl'

def build_pipeline(num_attribs, cat_attribs):

    num_pipline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler()),
    ])
    cat_pipline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    pipeline = ColumnTransformer([
        ('num', num_pipline, num_attribs),
        ('cat', cat_pipline, cat_attribs)
    ])

    return pipeline

if not os.path.exists(MODEL_FILE):
    # Load the dataset
    housing = pd.read_csv("housing.csv")

    # Split the dataset into training and testing sets
    housing['income_cat'] = pd.cut(housing['median_income'], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing['income_cat']) : 
        housing.loc[test_index].drop('income_cat', axis=1).to_csv("input.csv", index=False)
        housing = housing.loc[train_index].drop('income_cat', axis=1)

    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value', axis=1)

    num_attribs = housing_features.drop('ocean_proximity', axis=1).columns.tolist()
    cat_attribs = ['ocean_proximity']

    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE, compress=3)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is trained successfully, Congrats !")

else:   
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('input.csv').drop(columns=['income_cat'], errors='ignore')
    test_labels = input_data['median_house_value']
    test_features = input_data.drop('median_house_value', axis=1)

    transformed_input = pipeline.transform(test_features)
    predictions = model.predict(transformed_input)

    print("RMSE:", root_mean_squared_error(test_labels, predictions))
    print("MAE:", mean_absolute_error(test_labels, predictions))
    print("R²:", r2_score(test_labels, predictions))

    print("Inference is complete, results saved to output.csv Enjoy!")