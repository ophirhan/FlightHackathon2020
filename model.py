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


    def __hhmm_to_minutes_from_midnight(self, x):
        """Turns a time column of format hmm or hhmm to minutes from midnight"""
        return ((x // 100) * 60) + (x % 100)

    def pre_processing(self, x):
        """
        Preproccess X from sample space turns it into x_hat edible by our hypothesis
        :param x:
        :return:
        """

        # DayOfWeek:	Day of Week
        x = pd.get_dummies(x, columns=['DayOfWeek'], drop_first=True)

        # FlightDate:	Flight Date (yyyymmdd)
        date = pd.to_datetime(x['FlightDate'], errors="coerce")
        x['year'] = date.dt.year
        x['month'] = date.dt.month
        x['day'] = date.dt.day

        # Reporting_Airline: Unique Carrier Code.
        x = pd.get_dummies(x, columns=['Reporting_Airline'], drop_first=True)

        # Tail_Number: Tail Number
        # Flight_Number_Reporting_Airline: Flight Number
        # DestState: Destination Airport, State Code
        # x = pd.get_dummies(x, columns=["Tail_Number"], drop_first=True)
        x = x.drop(columns=['Tail_Number', 'Flight_Number_Reporting_Airline',
                            'FlightDate', 'OriginState', 'DestState'])  # 8000 unique?!

        # Origin:	Origin Airport
        x = pd.get_dummies(x, columns=['Origin'], drop_first=True)

        # OriginCityName:	Origin City, State
        x = pd.get_dummies(x, columns=['OriginCityName'], drop_first=True)

        # OriginState: Origin Airport, State Code

        # Dest: Destination Airport
        x = pd.get_dummies(x, columns=['Dest'], drop_first=True)

        # DestCityName: Destination City, Dest state
        x = pd.get_dummies(x, columns=['DestCityName'], drop_first=True)


        # CRSDepTime: CRS Departure Time (local time: hhmm)
        x['CRSDepTime'] = self.__hhmm_to_minutes_from_midnight(x['CRSDepTime'])

        # CRSElapsedTime: CRS Elapsed Time of Flight, in Minutes

        # Distance: Distance between airports (miles)

        # CRSArrTime: Planned Arrival Time (local time: hhmm)
        x['CRSArrTime'] = self.__hhmm_to_minutes_from_midnight(x['CRSArrTime'])

        # x = x.dropna()TODO are they going to bring none?

        return x



    def predict(self, x):
        """
        Recieves a pandas DataFrame of shape (m, 15) with m flight features, and predicts their
        delay at arrival and the main factor for the delay.
        @param x: A pandas DataFrame with shape (m, 15)
        @return: A pandas DataFrame with shape (m, 2) with your prediction
        """
        raise NotImplementedError


