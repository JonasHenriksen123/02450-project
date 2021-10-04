import numpy as np
import pandas as pd
import xlrd
from pathlib import Path


class Data:
    # region class members
    __filename = Path(__file__).parent / './ressources/forestfires.csv'

    __months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    __week_days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    # endregion

    def __init__(self):
        # region fetch data from .csv file
        df = pd.read_csv(self.__filename)
        self.__raw_data = df.values
        # endregion

        cols = range(0, 13)
        self.x = np.asarray(self.__raw_data[:, cols])
        self.x_no_label = self.x

        # region create labeling
        labels = np.asarray(self.x[:, 12])
        self.df = np.asarray(self.__raw_data[:, cols])
        self.x = self.df[:,:12]
        
        labels = np.asarray(self.df[:,12])
        
        temp = []
        for label in labels:
            if label == 0:
                temp.append(0)
            else:
                temp.append(1)

        labels = temp
        names = sorted(set(labels))
        self.class_dict = dict(zip(names, (range(len(names)))))

        self.df[:, 12] = labels
        self.y = np.asarray(([self.class_dict[value] for value in labels]))
        # endregion

        # region encoding months and week days
        self.month_dict = dict(zip(self.__months, range(1, len(self.__months) + 1)))
        self.week_days_dict = dict(zip(self.__week_days, range(1, len(self.__week_days) + 1)))

        aplic_cols = range(2, 4)
        vals = np.asarray(self.__raw_data[:, aplic_cols])

        for val in vals:
            val[0] = self.month_dict[val[0]]
            val[1] = self.week_days_dict[val[1]]
        self.x[:, aplic_cols] = vals
        # endregion
        
        self.x = np.asarray(self.x, dtype=float)
        self.x2 = self.x[:, range(0, 12)]
        # self.x2 = self.x[:, range(0, 12)]
        self.attributes = np.asarray(df.columns[range(12)])
        self.attribute_units = ['coordinate',
                                'coordinate',
                                'month',
                                'day',
                                'FFMC index',
                                'DMC index',
                                'DC index',
                                'ISI index',
                                '°C',
                                '%',
                                'km/h',
                                'mm/m^2',
                                'ha']

        self.N, self.M = self.x.shape
        self.C = len(names)

        # region summary statistics
        self.mean = np.mean(self.x, axis=0)
        self.std = np.std(self.x, axis=0)
        self.min = np.min(self.x, axis=0)
        self.q1 = np.quantile(self.x, 0.25, axis=0)
        self.median = np.median(self.x, axis=0)
        self.q3 = np.quantile(self.x, 0.75, axis=0)
        self.max = np.max(self.x, axis=0)
        self.range = np.max(self.x, axis=0) - np.min(self.x, axis=0)
        # endregion
        
        self.x_tilda = self.x - np.ones((self.N, 1)) * self.mean
        self.x_tilda = self.x_tilda * (1 / np.std(self.x_tilda, axis=0)) 
        
        
        # read as pandas dataframe: for plotting
        self.df = pd.read_csv(self.__filename)
        self.df_attributes = list(self.df.columns)
        self.df['month'] = df['month'].replace(self.month_dict)
        self.df['day'] = df['day'].replace(self.week_days_dict)
        self.df_entire = self.df
        
        temp = []
        temp1 = []
        for i in df['area']:
            if i > 0.5:
                temp.append('Small')
                temp1.append(0)
            else:
                temp.append('Big')
                temp1.append(1)
        
        self.df = self.df.drop('area',axis=1)
        self.df_entire = self.df_entire.drop('area',axis=1)
        self.df['Burned area'] = temp
        self.df_entire['area'] = temp1
        
        self.df_tilda = (self.df.iloc[:, :-1] - self.df.iloc[:, :-1].mean(axis=0))/self.df.iloc[:, :-1].std(axis=0)
        self.df_tilda['Burned area'] = temp
        self.df_data = self.df.drop('Burned area',axis=1) # output variable not included
        self.df_data_tilda= self.df_tilda.drop('Burned area',axis=1) #output variable not included
        #endregion
        

    def get_column_range(self, col_range: range):
        return self.x[:, col_range]

    def get_columns_sorted(self, col_array: [int]):
        cols = []
        col_array = sorted(col_array, reverse=False)

        for col in col_array:
            cols.append(self.x[:, col])

        return cols

    def get_columns(self, col_array: [int]):
        cols = []

        for col in col_array:
            cols.append(self.x[:, col])

        return cols
