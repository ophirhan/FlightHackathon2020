"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Author(s):
Eitan Navon, Ofir Han, Tal Mendelovits
===================================================
"""
import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib as plt


class FlightPredictor:
    def __init__(self, path_to_weather=''):
        """
        Initialize an object from this class.
        @param path_to_weather: The path to a csv file containing weather data.
        """
        raise NotImplementedError

    def _pre_processing(self, x):
        # x = x.drop("bli bli blah", 0) #TODO what should we write here
        date = pd.to_datetime(x['FlightDate'], errors="coerce")
        x['year'] = date.dt.year
        x['month'] = date.dt.month
        x['day'] = date.dt.day

        x = pd.get_dummies(x, columns=['DayOfWeek'], drop_first=True)
        x = pd.get_dummies(x, columns=['Reporting_Airline'], drop_first=True)
        x = pd.get_dummies(x, columns=['Origin'], drop_first=True)
        x = pd.get_dummies(x, columns=['OriginCityName'], drop_first=True)
        x = pd.get_dummies(x, columns=['Dest'], drop_first=True)
        x = pd.get_dummies(x, columns=['DestCityName'], drop_first=True)
        # x = pd.get_dummies(x, columns=["Tail_Number"], drop_first=True)
        x = pd.drop(x, columns=['Tail_Number', 'Flight_Number',
                                'FlightDate', 'OriginState', 'DestState'])  # 8000 unique?!
        # x = x.dropna()TODO are they going to bring none?


    def __split_y(self, x):
        y_delay = x["ArrDelay"].values
        x = x.drop("ArrDelay", 1)
        y_factor = x["DelayFactor"].values
        x = x.drop("DelayFactor", 1)

    def predict(self, x):
        """
        Recieves a pandas DataFrame of shape (m, 15) with m flight features, and predicts their
        delay at arrival and the main factor for the delay.
        @param x: A pandas DataFrame with shape (m, 15)
        @return: A pandas DataFrame with shape (m, 2) with your prediction
        """
        raise NotImplementedError


