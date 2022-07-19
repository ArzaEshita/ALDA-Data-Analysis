import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from random import randint
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

kmeans_kwargs = {
  "init": "random",
  "n_init": 10,
  "max_iter": 300,
  "random_state": None,
}
plot_style = "fivethirtyeight"
number_of_clusters = 11  # 1 - 10 clusters will be created and tested
colors = {}
dir_path = os.path.dirname(os.path.realpath(__file__))


class Clustering:
    """
    This class is the main clustering module. This class helps in running DBScan and KMeans.
    The class is also responsible to plot the clustered data along with finding performance metrics like SSE.
    """

    def __init__(self):
        pass

    @classmethod
    def run_dbscan(cls, data, data_column, daly_column):
        """
        Runs the DBSCAN algorithm on the given dataset

        :param data: Dataframe containing the preprocessed data
        :param data_column: Data column name containing the X axis data
        :param daly_column: Data column name containing the Y axis data
        :return: a combination of data and cluster it belongs to
        """
        dbscan = DBSCAN()
        dbscan.fit(data)
        identified_clusters = dbscan.fit_predict(data)
        return ([data[data_column], data[daly_column], identified_clusters])

    @classmethod
    def run_kmeans(cls, data, data_column, daly_column):
        """
        Runs the KMeans algorithm on the given dataset

        :param data: Dataframe containing the preprocessed data
        :param data_column: Data column name containing the X axis data
        :param daly_column: Data column name containing the Y axis data

        :return: a combination of data and cluster it belongs to
        """
        optimal_k = cls.find_optimum_k(data, method="sse", filename_suffix="testing")
        if optimal_k:
            kmeans = KMeans(optimal_k)
            kmeans.fit(data)
            identified_clusters = kmeans.fit_predict(data)
            return ([data[data_column], data[daly_column], identified_clusters])

    @classmethod
    def find_optimum_k(cls, data, method, filename_suffix):
        """
        Calculates the optimum K value, where K stands for the number of clusters

        :param data: Dataframe containing the preprocessed data
        :param method: String containing the method to be used for estimating the optimal number of clusters
        :param filename_suffix: String containing the suffix to be used for the graph image
        :return: the optimal k value if any, else None
        """
        # Using sum of squared error to evaluate clustering technique
        if method == "sse":
            sse = []
            exceptions_rasied = 0
            for k in range(1, number_of_clusters):
                try:
                    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
                    kmeans.fit(data)
                    sse.append(kmeans.inertia_)
                except Exception as e:
                    exceptions_rasied += 1
                    print("An exception occurred, passing iteration with {} clusters for {}. \nException : {}".format(k, filename_suffix, e))
                    pass

            final_number_of_clusters = number_of_clusters - exceptions_rasied
            title = "Scree plot for " + filename_suffix + " data"
            filename = "{}_{}_cluster_{}.png".format(method, k, filename_suffix)
            # cls.plot_scree_graphs(sse, final_number_of_clusters, "Number of Clusters", "SSE", title, filename)

            try:
                kl = KneeLocator(
                    range(1, final_number_of_clusters), sse, curve="convex", direction="decreasing"
                )
                return kl.elbow
            except Exception as e:
                print("An exception occurred, when finding elbow : {}".format(e))
                return None


    @classmethod
    def generate_colors(cls):
        """
        Generates random colours to be used with the scatter plots

        :return: List containing random color codes
        """
        for i in range(number_of_clusters):
            colors[i] = '#%06X' % randint(0, 0xFFFFFF)
        return colors

    @classmethod
    def plot_scree_graphs(cls, sse, final_number_of_clusters, x_label, y_label, title, filename):
        """
        Plots scree graphs for the number of clusters

        :param final_number_of_clusters: Integer containing the number of clusters
        :param x_label: String containing the x label for the graph
        :param y_label: String containing the y label for the graph
        :param title: String containing the title for the graph
        :param filename: String containing the filename to be used for the graph image
        :return: None
        """
        plt.style.use(plot_style)
        plt.plot(range(1, final_number_of_clusters), sse)
        plt.xticks(range(1, final_number_of_clusters))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        # plt.show()
        image_filepath = os.path.join(dir_path, 'images/kmeans/sse/' + filename)
        plt.savefig(image_filepath)

    @classmethod
    def plot_graphs(cls, kmeans, x_axis_data, y_axis_data, x_label, y_label, title, filename):
        """
        Plot graphs that help with visualizing the different cluster groups generated

        :param kmeans: Object for kmeans
        :param x_axis_data: List containing the x axis data
        :param y_axis_data: List containing the y axis data
        :param x_label: String containing the x label for the graph
        :param y_label: String containing the y label for the graph
        :param title: String containing the title for the graph
        :param filename: String containing the filename to be used for the graph image
        :return: None
        """
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        km_colors = [colors[label] for label in kmeans.labels_]
        ax.scatter(x_axis_data, y_axis_data, c=km_colors)
        for i in range(len(x_axis_data)):
            plt.annotate("{} - {}".format(x_axis_data[i], round(y_axis_data[i], 2)),
                         (x_axis_data[i], y_axis_data[i]), rotation=45, fontsize=8)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("{} clustered data".format(title))
        image_filepath = os.path.join(dir_path, 'images/kmeans/clustered_data/' + filename)
        plt.savefig(image_filepath)
        plt.close()

    @ classmethod
    def get_cosine_similarity(cls, data, attribute1, attribute2):
        """
        Finds out the cosine similarity for two attributes of a given dataset

        :param data: Given data set
        :param attribute1: First attribute for comparison using cosine similarity
        :param attribute2: Second attribute for comparison using cosine similarity
        :return: The cosine similarity value between the target attributes
        """
        vec1 = np.array([data[attribute1]])
        vec2 = np.array([data[attribute2]])
        print(cosine_similarity(vec1, vec2))
        return cosine_similarity(vec1, vec2)
