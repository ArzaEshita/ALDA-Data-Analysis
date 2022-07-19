import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt


def visualise_yearWise(country,service,service_level,df):
    df_country = df[df['Country'] == country]
    df_country = df_country[df_country['Service Type'] == service]
    df_country = df_country[df_country['Service level'] == service_level]
    df_country = pd.DataFrame(df_country,columns=['Year','Coverage'])
    df_country.plot(x='Year', y='Coverage', figsize=(10,5), grid=True)


def visualise(country,service,year,df):
    df_country = df[df['Country'] == country]
    df_country = df_country[df_country['Service Type'] == service]
    df_country = df_country[df_country['Year'] == year]
    df_country.groupby(['Service level']).sum().plot(kind='pie', y='Coverage')

cwd_path = Path.cwd()
original_df_1 = pd.read_csv(cwd_path.joinpath("data/water_sanitation_countries-1.csv"))
original_df_2 = pd.read_csv(cwd_path.joinpath("data/water_sanitation_countries-2.csv"))
original_df_3 = pd.read_csv(cwd_path.joinpath("data/water_sanitation_countries-3.csv"))
original_attributes = ['ISO3','Country','Residence Type','Service Type','Year','Coverage','Population','Service level']
df_combined = pd.concat([original_df_1, original_df_2,original_df_3])
attributes_set = ['Country','Residence Type','Service Type','Year','Coverage','Service level']
df_selected_attributes = pd.DataFrame(df_combined,columns=attributes_set)
df_total_rows = df_selected_attributes[df_selected_attributes['Residence Type'] == 'total']
df_total_rows.Coverage = df_total_rows.Coverage.astype(float)
df_total_rows['Year']= df_total_rows['Year'].map(str)
df_total_rows.Coverage = (df_total_rows.Coverage)/100

df_sanitation = df_total_rows[df_total_rows['Service Type'] == 'Sanitation']
df_hygiene = df_total_rows[df_total_rows['Service Type'] == 'Hygiene']
df_water = df_total_rows[df_total_rows['Service Type'] == 'Drinking Water']

visualise('India','Sanitation','2020',df_total_rows)
visualise_yearWise('India','Sanitation','Limited service',df_total_rows)