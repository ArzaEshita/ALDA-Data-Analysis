import os
from analyze import Analyze

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # data_filepath = os.path.join(dir_path, '../data/DiseaseBurden_PP.csv')
    # data = Helper.read_food_data(filename=data_filepath)

    Analyze.cluster_food_daly_data()
    Analyze.cluster_sanitation_daly_data(service_type="Sanitation")
    Analyze.cluster_sanitation_daly_data(service_type="Drinking water")



