import datetime

import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib as plt
from sys import argv
from model import FlightPredictor



if __name__ == '__main__':
    x = pd.read_csv(argv[1], dtype={"DelayFactor": str})
    pred = FlightPredictor()

    # separate x from y
    y_delay = x["ArrDelay"].values
    # y_factor = pd.get_dummies(x["DelayFactor"], columns=['DelayFactor']).values todo
    y_factor = x["DelayFactor"]
    x = x.drop(columns=['ArrDelay', 'DelayFactor'])
    y_delay = pd.factorize(y_delay)
    # pred.p

    dropped = ['Tail_Number', 'Flight_Number_Reporting_Airline',
                            'FlightDate', 'OriginState', 'DestState']
    x = x.drop(columns=dropped)  # 8000 unique?! yes! of course
    for i in x:
        if i not in dropped:
            delay_corr = np.cov(x[i], y_delay)[0][1] / (np.std(x[i]) * np.std(y_delay))
            factor_corr = np.cov(x[i], y_factor)[0][1] / (np.std(x[i]) * np.std(y_factor))
            print("delay Correlation according to " + i + ": " + str(round(delay_corr, 3)))
            print("factor Correlation according to " + i + ": " + str(round(delay_corr, 3)))

