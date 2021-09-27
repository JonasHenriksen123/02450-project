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
        self.x = self.__raw_data[:, cols]

        # region index months and week days
        self.month_dict = dict(zip(self.__months, range(1, len(self.__months) + 1)))
        self.week_days_dict = dict(zip(self.__week_days, range(1, len(self.__week_days) + 1)))

        aplic_cols = range(2, 4)

        vals = self.__raw_data[:, aplic_cols]

        for val in vals:
            val[0] = self.month_dict[val[0]]
            val[1] = self.week_days_dict[val[1]]

        self.x[:, aplic_cols] = vals
        # endregion

        self.attributes = np.asarray(df.columns[cols])
        self.data_count, self.attribute_count = self.x.shape

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
