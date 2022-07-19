import pandas as pd
import numpy as np
from sklearn import preprocessing


class Preprocessor:
    """
    This class helps in preprocessing the data. We have three different dataset.
    1. Food Balances
    2. Water, Sanitation and Hygiene
    3. Disease burden by risk factor

    We tried to make the code as general as possible but some functions are specific to dataset.
    """

    def __init__(self, data):
        """
        Initialize the Preprocessing class with the data

        :param data: data (pandas dataframe) to be preprocessed
        """
        self.data = data

    def get_data(self):
        """
        Return the data
        :return: data - pandas dataframe
        """
        return self.data.reset_index(drop=True)

    def drop_feature(self, cols, rows, cols_regex=None, row_regex=None):
        """
        Drop features based on feature. Removes entire column or an entire row
        :param cols: list of columns to be dropped
        :param rows: dict with feature name and row value which needs to be removed
        :param cols_regex: regex of columns to be dropped
        :param row_regex: regex of rows to be dropped
        :return: None
        """

        self.col_drop_feature(cols, cols_regex)
        self.row_drop_feature(rows, row_regex)

    def row_drop_feature(self, rows, regex=None):
        """
        Drop all rows which matches the parameter

        :param rows: dict with feature name and row value which needs to be removed
        :param regex: regex of row value
        :return: None
        """
        if(len(rows)) > 0:
            for key, value in rows.items():
                for val in value:
                    self.data = self.data[self.data[key] != val]

    def col_drop_feature(self, features, regex_features):
        """
        Drop columns feature

        :param features: list of columns to be dropped
        :param regex_features: regex of list of columns to be dropped
        :return: None
        """
        filtered_data = self.data.drop(features, axis=1)
        if regex_features:
            filtered_data = filtered_data[filtered_data.columns.drop(list(filtered_data.filter(regex=regex_features)))]
        self.data = filtered_data

    def feature_subset(self, col_features=None, row_features=None):
        """
        Select the only interesting feature from the dataset

        :param col_features: list of columns to be included
        :param row_features: dict with feature name with row value to be included
        :return: None
        """
        if row_features is None:
            row_features = []
        if col_features is None:
            col_features = []
        self.col_feature_subset(col_features)
        if row_features:
            self.row_feature_subset(row_features)

    def col_feature_subset(self, col_features):
        pass

    def row_feature_subset(self, row_features):
        """
        Select only the rows which contribute to the analysis

        :param row_features: dict with feature name with row value to be included
        :return: None
        """
        for key, value in row_features.items():
            self.data = self.data[self.data[key].isin(value)]

    #Duplicate data
    def remove_duplicate_data(self, duplicate_cols=None, duplicate_cols_regex=None):
        """
        Remove the duplicate data

        :param duplicate_cols: list of columns name
        :param duplicate_cols_regex: regex of list of columns
        :return: None
        """
        if duplicate_cols_regex:
            duplicate_cols = list(self.data.filter(regex=duplicate_cols_regex))
        self.data = self.data.groupby(duplicate_cols).mean().reset_index()
        self.data = self.data.drop_duplicates(subset=duplicate_cols).reset_index(drop=True)

    def remove_missing_values(self, missing_features=None, missing_features_regex=None, threshold=None):
        """
        Removing missing values. As the data is dense, it will not affect the model drastically

        :param missing_features: list of missing fetures/columns
        :param missing_features_regex: regex of list of missing fetures/columns
        :param threshold: threshold of the missing values in a row
        :return: None
        """
        if missing_features_regex:
            missing_features = list(self.data.filter(regex=missing_features_regex))
        data = self.data.dropna(subset=missing_features, axis=0, thresh=threshold)
        data = data.loc[(data[missing_features] != 0).all(axis=1)]
        self.data = data

    def food_data_population(self, col_regex):
        """
        Preprocess food data to get population

        :param col_regex: regex of columns to consider
        :return: None
        """

        df_population = self.data[(self.data['Item'] == 'Population')]
        self.data = self.data[(self.data['Item'] != 'Population')]
        years_list = list(self.data.filter(regex=col_regex))

        for index, row in df_population.iterrows():
            for year in years_list:
                self.data[year] = np.where(self.data['Area'] == row['Area'], self.data[year] * row[year], self.data[year])


    def standardize_data(self):
        """
        Standardize the data
        :return: standardized data = pandas dataframe
        """
        standardized_data = (self.data - self.data.mean()) / self.data.std()
        return standardized_data

    def normalize_data(self, features_list=None, features_regex=None):
        """
        Normalize the data between [0,1]
        :param features_list: list of columns which needs to be normalized
        :param features_regex: regex of columns which needs to be normalized
        :return: None
        """
        if features_regex:
            features_list = list(self.data.filter(regex=features_list))
        x = self.data[features_list]
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        normalized_data = pd.DataFrame(x_scaled)
        self.data[features_list] = normalized_data[features_list]

    def rename_cols(self, list_cols):
        """
        Rename attributes

        :param list_cols: list of columns to be renamed
        :return: None
        """
        list_cols = self.data.columns
        for cols in list_cols:
            cols_arr = cols.split(" - ")
            if len(cols_arr) > 1:
                self.data.rename(columns={cols: cols_arr[1].strip()}, inplace=True)

    def get_population(self):
        """
        Get the population of the food dataset
        :return: None
        """
        self.data = self.data[self.data['Element'] == 'Total Population - Both sexes']

    # def get_per_capital(self, population_data, col_regex=".*[0-9]{4}"):
    #     years_list = list(population_data.filter(regex=col_regex))
    #     for index, row in population_data.iterrows():
    #         for year in years_list:
    #             self.data = self.data.loc[self.data['Year'] == year[1:] & self.data['Entity'] == row['Area'], 3:] / row[year]

