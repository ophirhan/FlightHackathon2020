import datetime
import math
import time

import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib as plt
from sys import argv
from model import FlightPredictor
import re
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

if __name__ == '__main__':
    x = pd.read_csv(argv[1], dtype={"DelayFactor": str})
    pred = FlightPredictor()

    # separate x from y
    y_delay = x["ArrDelay"].values
    y_factor = pd.get_dummies(x["DelayFactor"], columns=['DelayFactor']).values
    x = x.drop(columns=['ArrDelay', 'DelayFactor'])


    x = pred.pre_processing(x)
    # categorized = ['DayOfWeek', 'Reporting_Airline', 'Origin', 'Dest']
    # for i in x:
    #     if i not in categorized:
    #         delay_corr = np.cov(x[i], y_delay)[0][1] / (np.std(x[i]) * np.std(y_delay))
    #         factor_corr = np.cov(x[i], y_factor)[0][1] / (np.std(x[i]) * np.std(y_factor))
    #         print("delay Correlation according to " + i + ": " + str(round(delay_corr, 5)))
    #         print("factor Correlation according to " + i + ": " + str(round(delay_corr, 5)))

    # # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(x, y_delay, test_size=test_size, random_state=seed)
    #
    # # fit model no training data
    # model = XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
    #             max_depth=5, alpha=10, n_estimators=10)
    # model.fit(X_train, y_train)
    #
    # model.save_model(str(time.localtime()))
    #
    # y_pred = model.predict(X_test)
    # print(y_pred)

    # predictions = [round(value) for value in y_pred]

    # mse = math.sqrt(np.mean(y_pred - y_test) ** 2)
    # print(mse)

    model = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=4, num_class=5, objective="multi:softmax"))


    model.fit(X_train, y_train)
    model.save_model("classifier " + str(time.localtime()))

    y_pred = model.predict(X_test)
    error = np.where(y_pred != y_test, 1, 0)
    print(np.sum(error) / len(y_test))
