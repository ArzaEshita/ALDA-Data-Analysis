import os
from preprocessor import Preprocessor
from helper import Helper


if __name__ == "__main__":
    col_years_fc_regex = ".*[0-9]{4}F"
    col_years_regex = ".*[0-9]{4}"

    col_drop_features = ['Area Code', 'Item Code', 'Element Code']
    row_drop_features = {
        "Unit": ['1000 tonnes'],
        # "Item": ['Grand Total', 'Infant food', 'Miscellaneous', 'Population'],
        "Item": ['Grand Total', 'Infant food', 'Miscellaneous'],
    }

    row_feature_subset = {
        "Item": [
            'Population',
            'Alcoholic Beverages',
            'Animal fats',
            'Aquatic Products, Other',
            'Cereals - Excluding Beer',
            'Eggs',
            'Fish, Seafood',
            'Fruits - Excluding Wine',
            'Meat',
            'Milk - Excluding Butter',
            'Offals',
            'Oilcrops',
            'Pulses',
            'Spices',
            'Starchy Roots',
            'Stimulants',
            'Sugar & Sweeteners',
            'Sugar Crops',
            'Treenuts',
            'Vegetable Oils',
            'Vegetables'
        ]
    }

    cols_duplicate_features = ['Area', 'Item', 'Element', 'Unit']

    dir_path = os.path.dirname(os.path.realpath(__file__))
    population_file = os.path.join(dir_path, '../data/FoodBalanceSheets_E_All_Data.csv')
    data = Helper.read_food_data(filename=population_file)
    prep_obj = Preprocessor(data)
    prep_obj.get_population()
    prep_obj.drop_feature(cols=col_drop_features, rows=row_drop_features, cols_regex=col_years_fc_regex)
    data = prep_obj.get_data()
    Helper.write_data_to_csv(os.path.join(dir_path, '../data/Population.csv'), data)

    fooddata_filepath_2014 = os.path.join(dir_path, '../data/FoodBalanceSheets_E_All_Data.csv')
    fooddata_filepath_2013 = os.path.join(dir_path, '../data/FoodBalanceSheetsHistoric_E_All_Data.csv')
    data = Helper.read_food_data(filename=fooddata_filepath_2014)
    print(data.head())


    prep_obj = Preprocessor(data)
    prep_obj.drop_feature(cols=col_drop_features, rows=row_drop_features, cols_regex=col_years_fc_regex)
    prep_obj.feature_subset({}, row_feature_subset)
    prep_obj.remove_duplicate_data(duplicate_cols=cols_duplicate_features)
    prep_obj.remove_missing_values(missing_features_regex=col_years_regex)
    # prep_obj.food_data_population(col_regex=col_years_regex)

    data = prep_obj.get_data()
    print(data.head())
    Helper.write_data_to_csv(os.path.join(dir_path, '../data/FoodBalanceSheets_2014_PP.csv'), data)
    #
    # data_2013 = Helper.read_food_data(filename=fooddata_filepath_2013)
    # print(data_2013.head())
    #
    # prep_obj = Preprocessor(data_2013)
    # prep_obj.drop_feature(cols=col_drop_features, rows=row_drop_features, cols_regex=col_years_fc_regex)
    # prep_obj.feature_subset({}, row_feature_subset)
    # prep_obj.remove_duplicate_data(duplicate_cols=cols_duplicate_features)
    # prep_obj.remove_missing_values(missing_features_regex=col_years_regex)
    # # prep_obj.food_data_population(col_regex=col_years_regex)
    #
    # data_2013 = prep_obj.get_data()
    # print(data_2013.head())
    # Helper.write_data_to_csv(os.path.join(dir_path, '../data/FoodBalanceSheets_2013_PP.csv'), data_2013)

    disease_burden = os.path.join(dir_path, '../data/disease-burden-by-risk-factor.csv')
    data = Helper.read_food_data(filename=disease_burden)
    col_drop_features = ['Code',
                         'DALYs (Disability-Adjusted Life Years) - Air pollution - Sex: Both - Age: All Ages (Number)',
                         'DALYs (Disability-Adjusted Life Years) - Child wasting - Sex: Both - Age: All Ages (Number)',
                         'DALYs (Disability-Adjusted Life Years) - Child stunting - Sex: Both - Age: All Ages (Number)',
                         'DALYs (Disability-Adjusted Life Years) - Secondhand smoke - Sex: Both - Age: All Ages (Number)',
                         'DALYs (Disability-Adjusted Life Years) - Low physical activity - Sex: Both - Age: All Ages (Number)',
                         'DALYs (Disability-Adjusted Life Years) - Non-exclusive breastfeeding - Sex: Both - Age: All Ages (Number)',
                         'DALYs (Disability-Adjusted Life Years) - Ambient particulate matter pollution - Sex: Both - Age: All Ages (Number)',
                         'DALYs (Disability-Adjusted Life Years) - Household air pollution from solid fuels - Sex: Both - Age: All Ages (Number)',
                         'DALYs (Disability-Adjusted Life Years) - Drug use - Sex: Both - Age: All Ages (Number)',
                         'DALYs (Disability-Adjusted Life Years) - Smoking - Sex: Both - Age: All Ages (Number)',
                         'DALYs (Disability-Adjusted Life Years) - High total cholesterol - Sex: Both - Age: All Ages (Number)',
                         'DALYs (Disability-Adjusted Life Years) - High fasting plasma glucose - Sex: Both - Age: All Ages (Number)',
                         'DALYs (Disability-Adjusted Life Years) - High body-mass index - Sex: Both - Age: All Ages (Number)']

    prep_obj = Preprocessor(data)
    prep_obj.col_drop_feature(features=col_drop_features, regex_features='')
    prep_obj.rename_cols(col_drop_features)

    # population_file = os.path.join(dir_path, '../data/Population.csv')
    # pop_data = Helper.read_food_data(filename=population_file)

    # prep_obj.get_per_capital(population_data=pop_data)
    daly_data = prep_obj.get_data()
    data = daly_data.loc[(daly_data['Year'] >= 2008) & (daly_data['Year'] <= 2017), :]
    print(data.head())
    Helper.write_data_to_csv(os.path.join(dir_path, '../data/DiseaseBurden_PP.csv'), data)





