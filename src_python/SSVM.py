import pandas as pd
import numpy as np
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import train_test_split
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
    X = stdscaler.fit_transform(np.array(X))
    y = pd.read_table("./../data/"+cancer+"/surv_df.tsv")
    y = np.array([(status, time) for status, time in zip(y['status'], y['time'])], dtype=[('status', 'bool'), ('time', '<f4')])
    perf_list = []
    train_idx_dict, test_idx_dict = torch.load("./../data/"+cancer+"/split_save.pt")

    for iteration in range(len(train_idx_dict)):

        train_idx = train_idx_dict[iteration]
        test_idx = test_idx_dict[iteration]

        X_train = X[train_idx]
        y_train = y[train_idx,]


        params = { 'alpha' : [100,10,5,2,1,0.5,0.1,0.01,0.001,0.0001],
                  'max_iter' : [3000],
                  'kernel':['poly','rbf','linear'],
                  'gamma':[0.0001,0.00001,0.000001]
                 }

        surv_svm_model = FastKernelSurvivalSVM()
        grid_cv = GridSearchCV(surv_svm_model, param_grid = params, cv = 3, n_jobs = -1,scoring=score_survival_model)
        grid_cv.fit(X_train, y_train)

        print('Optimal Hyperparmeter: ', grid_cv.best_params_)
        print('Best prediction score: {:.4f}'.format(grid_cv.best_score_))
        X_test = X[test_idx]
        y_test = y[test_idx,]

        surv_svm_model_ = FastKernelSurvivalSVM(**grid_cv.best_params_)
        surv_svm_model_.fit(X_train, y_train)
        pred = surv_svm_model_.predict(X_test)
        c_idx = surv_svm_model_.score(X_test,y_test)

        print(c_idx)
        perf_list.append(c_idx)
    print(cancer, config.input, "Avg : ", np.mean(perf_list))
    if config.output:
        torch.save(perf_list, "./../data/"+cancer+"/"+config.output)