import pandas as pd
import numpy as np

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings(action='ignore')

import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, default=False)
# parser.add_argument('--surv_input', '-s', type=str, default=False)
parser.add_argument('--output', '-o', type=str, default=False)
parser.add_argument('--cancer', '-c', type = str)

config = parser.parse_args()

def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored([status for status, time in y],[time for status, time in y], prediction)
    return result[0]


cancer_list = [config.cancer]
for cancer in cancer_list:
    stdscaler = StandardScaler()
    X = pd.read_table("./../data/"+cancer+"/"+config.input, header=None)
    X = np.array(X)
    y = pd.read_table("./../data/"+cancer+"/surv_df.tsv")
    y = np.array([(status, time) for status, time in zip(y['status'], y['time'])], dtype=[('status', 'bool'), ('time', '<f4')])
    perf_list = []
    train_idx_dict, test_idx_dict = torch.load("./../data/"+cancer+"/split_save.pt")
    for iteration in range(len(train_idx_dict)):

        train_idx = train_idx_dict[iteration]
        test_idx = test_idx_dict[iteration]

        X_train = X[train_idx]
        y_train = y[train_idx,]


        params = { 'n_estimators' : [20,50,100,200],
                   'max_depth' : [None],
                   'min_samples_leaf' : [2,4,8,16,24],
                   'min_samples_split' : [2,4,8,16]
                 }
        surv_rf_model = RandomSurvivalForest(random_state=0)
        grid_cv = GridSearchCV(surv_rf_model, param_grid = params, cv = 3, n_jobs = -1,scoring=score_survival_model)
        grid_cv.fit(X_train, y_train)

        print('Optimal Hyperparameter: ', grid_cv.best_params_)
        print('Best prediction score: {:.4f}'.format(grid_cv.best_score_))
        X_test = X[test_idx]
        y_test = y[test_idx,]

        surv_rf_model_ = RandomSurvivalForest(**grid_cv.best_params_)
        surv_rf_model_.fit(X_train, y_train)
        pred = surv_rf_model_.predict(X_test)
        c_idx = surv_rf_model_.score(X_test,y_test)
        print(c_idx)
        perf_list.append(c_idx)
    print(cancer, config.input, "Avg : ", np.mean(perf_list))
    if config.output:
        torch.save(perf_list, "./../data/"+cancer+"/"+config.output)