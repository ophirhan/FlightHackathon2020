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
    y_factor = pd.get_dummies(x["DelayFactor"], columns=['DelayFactor']).values
    x = x.drop(columns=['ArrDelay', 'DelayFactor'])

    pred.p


    x = x.drop(columns=['Tail_Number', 'Flight_Number_Reporting_Airline',
                            'FlightDate', 'OriginState', 'DestState'])  # 8000 unique?! yes! of course



