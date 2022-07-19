import pandas as pd


class Helper:
    def __init__(self):
        pass

    @classmethod
    def read_food_data(cls, filename):
        """
        Read the dataset

        :param filename: filename with complete/absolute path
        :return: data as pandas dataframe
        """
        df = pd.read_csv(filename, engine='python', encoding='latin-1')
        return df

    @classmethod
    def write_data_to_csv(cls, filename, data):
        """
        Write the data in CSV

        :param filename: name of the output csv file
        :param data: data that need to be written
        :return: None
        """
        data.to_csv(filename, encoding='utf-8', index=False)

# data_2013 = Helper.read_food_data("data/FoodBalanceSheets_2013_PP.csv")
# data = Helper.read_food_data("data/FoodBalanceSheets_2014_PP.csv")
#
# merged_data = pd.merge(data_2013, data, how='left', on=['Area', 'Element', 'Item', 'Unit'])
# print("*************************MERGED DATA**********************")
# print(merged_data.shape)
# print(merged_data.head(10))
# Helper.write_data_to_csv("data/merged_data.csv", merged_data)

