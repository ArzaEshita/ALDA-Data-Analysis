import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from clustering import Clustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
sns.set()

YEAR_DATA = "Y2008-Y2017"
kmeans_kwargs = {
  "init": "random",
  "n_init": 10,
  "max_iter": 300,
  "random_state": None,
}
plot_style = "fivethirtyeight"
number_of_clusters = 21  # 1 - 10 clusters will be created and tested
colors = {}
dir_path = os.path.dirname(os.path.realpath(__file__))

years = ['2017', '2016', '2015', '2014']
food_dict = {
    "Iron deficiency":["Animal fats","Aquatic Products, Other","Cereals - Excluding Beer","Eggs","Fish, Seafood","Fruits - Excluding Wine","Meat","Milk - Excluding Butter","Offals","Oilcrops","Pulses","Spices","Starchy Roots","Stimulants","Sugar & Sweeteners","Sugar Crops","Treenuts","Vegetable Oils","Vegetables"],
    "Zinc deficiency":["Animal fats","Aquatic Products, Other","Cereals - Excluding Beer","Eggs","Fish, Seafood","Fruits - Excluding Wine","Meat","Milk - Excluding Butter","Offals","Oilcrops","Pulses","Spices","Starchy Roots","Stimulants","Sugar & Sweeteners","Sugar Crops","Treenuts","Vegetable Oils","Vegetables"],
    "Diet high in sodium":["Animal fats","Aquatic Products, Other","Cereals - Excluding Beer","Eggs","Fish, Seafood","Fruits - Excluding Wine","Meat","Milk - Excluding Butter","Offals","Oilcrops","Pulses","Spices","Starchy Roots","Stimulants","Sugar & Sweeteners","Sugar Crops","Treenuts","Vegetable Oils","Vegetables"],
    "Diet low in vegetables":["Oilcrops", "Starchy Roots","Stimulants","Sugar & Sweeteners","Sugar Crops","Treenuts","Vegetable Oils","Vegetables"],
    "Vitamin A deficiency": ["Animal fats","Aquatic Products, Other","Cereals - Excluding Beer","Eggs","Fish, Seafood","Fruits - Excluding Wine","Meat","Milk - Excluding Butter","Offals","Oilcrops","Pulses","Spices","Starchy Roots","Stimulants","Sugar & Sweeteners","Sugar Crops","Treenuts","Vegetable Oils","Vegetables"],
    "High systolic blood pressure":["Animal fats","Aquatic Products, Other","Cereals - Excluding Beer","Eggs","Fish, Seafood","Fruits - Excluding Wine","Meat","Milk - Excluding Butter","Offals","Oilcrops","Pulses","Spices","Starchy Roots","Stimulants","Sugar & Sweeteners","Sugar Crops","Treenuts","Vegetable Oils","Vegetables"]
}
food_dict = {
    "Diet low in vegetables": ["Vegetables"],
    "Diet low in fruits": ["Fruits - Excluding Wine","Sugar & Sweeteners","Sugar Crops"],
    "High systolic blood pressure": ["Animal fats"]
}
food_elements = ['Fat supply quantity (g/capita/day)', 'Food supply (kcal/capita/day)', 'Food supply quantity (kg/capita/yr)', 'Protein supply quantity (g/capita/day)']

sanitation_dict = {
    "Unsafe sanitation": ["At least basic","Unimproved","Basic service"],
    "Unsafe water source": ["At least basic","Unimproved","Basic service"]
}


class Analyze:
    def __init__(self):
        pass

    @classmethod
    def cluster_sanitation_daly_data(cls, service_type):
        """
        Clusters the sanitation data with respective DALY using KMeans and DBSCAN and plots the graphs
        :param service_type: pass Sanitation for sanitation data, pass Drinking water for drinking water data
        :return:
        """
        cwd_path = Path.cwd()

        for key, value in sanitation_dict.items():
            for val in value:
                dbscan_results = []
                kmeans_results = []
                for year in years:
                    daly_col_name = key
                    target_sanitation_category = [val]

                    # Read the population data
                    population_row_name = "Total Population - Both sexes"
                    yearCol = 'Y' + year

                    food_population_df = pd.read_csv(cwd_path.joinpath("data/FoodBalanceSheets_E_All_Data.csv"), encoding='latin-1')
                    population_df = food_population_df[food_population_df['Element'] == population_row_name]
                    population_attributes_set = [yearCol, 'Area']
                    population_df = pd.DataFrame(population_df, columns=population_attributes_set)

                    # Read the sanitation data
                    original_df_1 = pd.read_csv(cwd_path.joinpath("data/water_sanitation_countries-1.csv"))
                    original_df_2 = pd.read_csv(cwd_path.joinpath("data/water_sanitation_countries-2.csv"))
                    original_df_3 = pd.read_csv(cwd_path.joinpath("data/water_sanitation_countries-3.csv"))

                    df_combined = pd.concat([original_df_1, original_df_2, original_df_3])
                    df_sanitation = df_combined
                    df_sanitation = df_sanitation[df_sanitation['Residence Type'] == 'total']
                    df_sanitation = df_sanitation[df_sanitation['Service Type'] == service_type]
                    print("DF Sanitation")
                    # Select required years data the sanitation data
                    print(df_sanitation.Year.unique())
                    df_sanitation = df_sanitation.loc[(df_sanitation['Year'] >= int(year)) & (df_sanitation['Year'] <= int(year)), :]

                    df_sanitation = df_sanitation.drop('Year', axis=1)

                    # Select only the required attributes data from the sanitation data
                    attributes_set = ['Country', 'Coverage', 'Service level']
                    df_sanitation = pd.DataFrame(df_sanitation, columns=attributes_set)

                    df_sanitation = df_sanitation.rename(columns={'Service level': 'Service_level'})
                    df_sanitation = df_sanitation[df_sanitation.Service_level.isin(target_sanitation_category)]
                    df_sanitation.groupby(['Country'])['Coverage'].sum().reset_index()

                    print("Sanitation data here is ")
                    print(df_sanitation.head())

                    df_sanitation = df_sanitation.drop('Service_level', axis=1)

                    # Read the disease burden data for the DALYs
                    disease_original_data = pd.read_csv(cwd_path.joinpath("data/DiseaseBurden_PP.csv"), encoding='latin-1')
                    dod = disease_original_data
                    data = dod.loc[(dod['Year'] >= int(year)) & (dod['Year'] <= int(year)), :]
                    data = data.drop('Year', axis=1)
                    averaged_data = data.groupby(['Entity']).mean()

                    data = averaged_data.reset_index()

                    disease_attributes = [daly_col_name,'Entity']
                    d_df = pd.DataFrame(data,columns=disease_attributes)
                    d_df[daly_col_name] = d_df[daly_col_name].astype(int)

                    d_df = d_df.rename(columns={'Entity': 'Area'})

                    d_df = pd.merge(d_df, population_df, on='Area')

                    print("Disease data is ")
                    print(d_df.head())
                    d_df[daly_col_name] = d_df[daly_col_name]/d_df[yearCol]

                    d_df = d_df.drop(yearCol, axis=1)

                    d_df = d_df.rename(columns={'Area': 'Country'})

                    df_combined = pd.merge(df_sanitation, d_df, on='Country')
                    df_combined = df_combined.sort_values(by=['Coverage'])
                    df_combined_k = df_combined.drop('Country', axis=1)
                    length, width = df_combined_k.shape
                    if length == 0 or width == 0:
                        continue

                    #Performing cosine similarity on Sanitation Data

                    cosine_sim = Clustering.get_cosine_similarity(df_combined_k,'Coverage',daly_col_name)

                    coverage = plt.plot(df_combined['Country'], df_combined_k['Coverage'], label = "Coverage")
                    daly = plt.scatter(df_combined['Country'], df_combined_k[daly_col_name],  label = "DALY", color='red')
                    plt.xlabel('Countries')
                    plt.ylabel('')
                    plt.xticks('Country', " ")
                    leg = plt.legend(loc='upper center')
                    plt.title('Correlation of DALY-Unsafe Sanitation and' + val + ' Sanitation Coverage \n' + str(year) + '\n Cosine Similarity = ' + str(cosine_sim[0]))
                    image_filepath = os.path.join(dir_path, 'images/results/cosine_similarity/' + val)
                    plt.savefig(image_filepath)
                    plt.close()

                    # Normalising the data
                    df_combined_k=(df_combined_k-df_combined_k.mean())/df_combined_k.std()

                    print("Df_combined_k is ")
                    print(df_combined_k.head())

                    x_label = 'Coverage % for ' + target_sanitation_category[0]
                    y_label = 'DALY - ' + key
                    title = 'Country-wise impact of ' + service_type + ' service level on DALY'
                    # Run DBSCAN on the dataset
                    dbscan_results.append(Clustering.run_dbscan(data=df_combined_k, data_column='Coverage', daly_column=daly_col_name))
                    # Run KMeans on the dataset
                    kmeans_results.append(Clustering.run_kmeans(data=df_combined_k, data_column='Coverage', daly_column=daly_col_name))

                print(len(dbscan_results[0]))
                # plot the respective graphs for DBSCAN results
                fig_dbscan, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 7))
                ax1.scatter(dbscan_results[0][0], dbscan_results[0][1], c=dbscan_results[0][2], cmap='rainbow')
                ax1.set_title("DBSCAN - " + years[0])
                ax2.scatter(dbscan_results[1][0], dbscan_results[1][1], c=dbscan_results[1][2], cmap='rainbow')
                ax2.set_title("DBSCAN - " + years[1])
                ax3.scatter(dbscan_results[2][0], dbscan_results[2][1], c=dbscan_results[2][2], cmap='rainbow')
                ax3.set_title("DBSCAN - " + years[2])
                ax4.scatter(dbscan_results[3][0], dbscan_results[3][1], c=dbscan_results[3][2], cmap='rainbow')
                ax4.set_title("DBSCAN - " + years[3])
                fig_dbscan.suptitle(service_type)
                for ax in fig_dbscan.get_axes():
                    ax.set(xlabel=val, ylabel="DALY" + ' ' + key)

                # Hide x labels and tick labels for top plots and y ticks for right plots.
                for ax in fig_dbscan.get_axes():
                    ax.label_outer()

                filename = service_type.replace(" ", "_") + "_" + key.replace(" ", "_") + "_" + val.replace(" ", "_") + ".png"
                # store the respective graphs in the path below
                image_filepath = os.path.join(dir_path, 'images/results/dbscan/' + filename)
                print(image_filepath)
                plt.savefig(image_filepath)
                plt.close()

                # plot the respective graphs for DBSCAN results
                fig_kmeans, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 7))
                ax1.scatter(kmeans_results[0][0], kmeans_results[0][1], c=kmeans_results[0][2], cmap='rainbow')
                ax1.set_title("KMeans - " + years[0])
                ax2.scatter(kmeans_results[1][0], kmeans_results[1][1], c=kmeans_results[1][2], cmap='rainbow')
                ax2.set_title("KMeans - " + years[1])
                ax3.scatter(kmeans_results[2][0], kmeans_results[2][1], c=kmeans_results[2][2], cmap='rainbow')
                ax3.set_title("KMeans - " + years[2])
                ax4.scatter(kmeans_results[3][0], kmeans_results[3][1], c=kmeans_results[3][2], cmap='rainbow')
                ax4.set_title("KMeans - " + years[3])
                fig_kmeans.suptitle(service_type)
                for ax in fig_kmeans.get_axes():
                    ax.set(xlabel=val, ylabel="DALY" + ' ' + key)

                # Hide x labels and tick labels for top plots and y ticks for right plots.
                for ax in fig_kmeans.get_axes():
                    ax.label_outer()

                filename = service_type.replace(" ", "_") + "_" + key.replace(" ", "_") + "_" + val.replace(" ", "_") + ".png"
                # store the respective graphs in the path below
                image_filepath = os.path.join(dir_path, 'images/results/kmeans/' + filename)
                print(image_filepath)
                plt.savefig(image_filepath)
                plt.close()

    @classmethod
    def cluster_food_daly_data(cls):
        """
        Clusters the nutrition dataset with respective DALY using KMeans and DBSCAN and plots the graphs
        :return:
        """
        for key, value in food_dict.items():
            for val in value:
                for element in food_elements:
                    dbscan_results = []
                    kmeans_results = []
                    for year in years:
                        cwd_path = Path.cwd()

                        daly_col_name = key
                        element_name = element
                        target_food_category = [val]

                        yearCol = 'Y' + year
                        # Read the population data
                        population_df = pd.read_csv(cwd_path.joinpath("data/Population.csv"), usecols=['Area', yearCol])
                        print("Population DF {}".format(population_df.head()))

                        # Read the food balance datasets
                        food_original_df = pd.read_csv(cwd_path.joinpath("data/FoodBalanceSheets_2014_PP.csv"))
                        food_original_df = food_original_df[food_original_df['Area'] != 'World']
                        food_fat_df_1 = food_original_df[food_original_df['Element'] == element_name]
                        food_fat_df_1 = food_fat_df_1[food_fat_df_1.Item.isin(target_food_category)]
                        food_fat_df_1.groupby(['Area'])[yearCol].sum().reset_index()
                        food_fat_df = food_fat_df_1

                        print("Food data here is {}".format(food_fat_df.head()))
                        attributes_set = [yearCol, 'Area']
                        food_fat_2014_df = pd.DataFrame(food_fat_df, columns=attributes_set)

                        ff_df = food_fat_2014_df

                        print("The food data is {}".format(ff_df.head()))

                        # Read the disease burden data for the DALYs
                        disease_original_data = pd.read_csv(cwd_path.joinpath("data/DiseaseBurden_PP.csv"))
                        dod = disease_original_data
                        data = dod.loc[(dod['Year'] >= int(year)) & (dod['Year'] <= int(year)), :]
                        data = data.drop('Year', axis=1)

                        disease_attributes = [daly_col_name, 'Entity']
                        d_df = pd.DataFrame(data, columns=disease_attributes)

                        d_df = d_df.rename(columns={'Entity': 'Area'})

                        print("Disease data is ")
                        print(d_df.head())

                        df_daly_population = pd.merge(d_df, population_df, on='Area')
                        print("DALY + Population : {}".format(df_daly_population.head()))
                        df_daly_population[daly_col_name] = df_daly_population[daly_col_name] / df_daly_population[yearCol]
                        df_daly_population = df_daly_population.drop(yearCol, axis=1)
                        print("DALY Per Capita : {}".format(df_daly_population.head()))

                        df_combined = pd.merge(ff_df, df_daly_population, on='Area')

                        print("DALY Combined : {}".format(df_combined.head()))

                        df_combined_k = df_combined.drop('Area', axis=1)

                        # Normalise the data
                        print("Df_combined_k is ")
                        print(df_combined_k.head())

                        # print("Mean: {} \t Standard Dev {}".format(df_combined_k.mean(), df_combined_k.std()))
                        # df_combined_k = (df_combined_k - df_combined_k.mean()) / df_combined_k.std()

                        print("Df_combined_k is ")
                        # print(df_combined_k.nsmallest(5, daly_col_name))
                        print(df_combined_k.head())

                        x_label = element_name
                        y_label = cls.clean_element(daly_col_name)
                        title = target_food_category[0]
                        # Run DBSCAN on the dataset
                        dbscan_results.append(Clustering.run_dbscan(data=df_combined_k, data_column=yearCol, daly_column=daly_col_name))
                        # Run KMeans on the dataset
                        kmeans_results.append(Clustering.run_kmeans(data=df_combined_k, data_column=yearCol, daly_column=daly_col_name))

                    element_split = element.split("(")
                    element_name = element_split[0].replace(" ", "_")

                    # plot the respective graphs for DBSCAN results
                    fig_dbscan, axs = plt.subplots(2, 2, figsize=(7, 7))
                    axs[0, 0].scatter(dbscan_results[0][0], dbscan_results[0][1], c=dbscan_results[0][2], cmap='rainbow')
                    axs[0, 0].set_title("DBSCAN - " + years[0])
                    axs[0, 1].scatter(dbscan_results[1][0], dbscan_results[1][1], c=dbscan_results[1][2], cmap='rainbow')
                    axs[0, 1].set_title("DBSCAN - " + years[1])
                    axs[1, 0].scatter(dbscan_results[2][0], dbscan_results[2][1], c=dbscan_results[2][2], cmap='rainbow')
                    axs[1, 0].set_title("DBSCAN - " + years[2])
                    axs[1, 1].scatter(dbscan_results[3][0], dbscan_results[3][1], c=dbscan_results[3][2], cmap='rainbow')
                    axs[1, 1].set_title("DBSCAN - " + years[3])
                    fig_dbscan.suptitle(val)
                    for ax in axs.flat:
                        ax.set(xlabel=element, ylabel=key)

                    # Hide x labels and tick labels for top plots and y ticks for right plots.
                    for ax in axs.flat:
                        ax.label_outer()

                    filename = key.replace(" ", "_") + "_" + val.replace(" ", "_") + "_" + element_name + ".png"
                    # Store the plots in the path given below
                    image_filepath = os.path.join(dir_path, 'images/results/dbscan/' + filename)
                    print(image_filepath)
                    plt.savefig(image_filepath)
                    plt.close()

                    # plot the respective graphs for KMeans results
                    fig_kmeans, axs = plt.subplots(2, 2, figsize=(7, 7))
                    axs[0, 0].scatter(kmeans_results[0][0], kmeans_results[0][1], c=kmeans_results[0][2], cmap='rainbow')
                    axs[0, 0].set_title("KMeans - " + years[0])
                    axs[0, 1].scatter(kmeans_results[1][0], kmeans_results[1][1], c=kmeans_results[1][2], cmap='rainbow')
                    axs[0, 1].set_title("KMeans - " + years[1])
                    axs[1, 0].scatter(kmeans_results[2][0], kmeans_results[2][1], c=kmeans_results[2][2], cmap='rainbow')
                    axs[1, 0].set_title("KMeans - " + years[2])
                    axs[1, 1].scatter(kmeans_results[3][0], kmeans_results[3][1], c=kmeans_results[3][2], cmap='rainbow')
                    axs[1, 1].set_title("KMeans - " + years[3])
                    fig_kmeans.suptitle(val)
                    for ax in axs.flat:
                        ax.set(xlabel=element, ylabel=key)

                    # Hide x labels and tick labels for top plots and y ticks for right plots.
                    for ax in axs.flat:
                        ax.label_outer()

                    filename = key.replace(" ", "_") + "_" + val.replace(" ", "_") + "_" + element_name + ".png"
                    # Store the plots in the path given below
                    image_filepath = os.path.join(dir_path, 'images/results/kmeans/' + filename)
                    print(image_filepath)
                    plt.savefig(image_filepath)
                    plt.close()

    @classmethod
    def clean_element(cls, data):
        """
        Remove special characters from a string
        :param data: String containing an attribute value
        :return: cleaned string
        """
        data = data.replace("(Disability-Adjusted Life Years) ", "")
        data_arr = data.split("(")
        data = data_arr[0].replace("-", "")
        # data = data.replace(" ", "_")
        # data = data.replace("__", "_")
        return data