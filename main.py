import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from fcmeans import FCM
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from seaborn import scatterplot as scatter
from yellowbrick.cluster import KElbowVisualizer

from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.birch import birch

from math import fabs
import scipy

# ---------------- Метод локтя ------------------------------------------------------------------
def ElbowMethod(X):
    model = KMeans()
    visualizer1 = KElbowVisualizer(model, k=(1, 15), timings=False)

    visualizer1.fit(X)  # Fit the data to the visualizer
    visualizer1.show()  # Finalize and render the figure


# метод силуэта
def silhouette_method(X):
    poss_n = range(2, 10)
    results = []
    for n in poss_n:
        clusterer = KMeans(n_clusters=n)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels, metric='euclidean')
        results.append(silhouette_avg)
    plt.plot(poss_n, results)
    plt.show()


# Метод Дависа-Болдуина
def dbi(X):
    poss_n = range(2, 10)
    results = []
    for n in poss_n:
        clusterer = KMeans(n_clusters=n)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = davies_bouldin_score(X, cluster_labels)
        results.append(silhouette_avg)
    plt.plot(poss_n, results)
    plt.xlabel('K')
    plt.show()


def load_data(file_name):
    birds = pd.read_csv(file_name).dropna(axis=1, how='all')
    print(birds)
    birds = birds.dropna()
    data = birds.drop('target', axis='columns')
    #data = data.drop('id', axis='columns')

    return birds, data


# кластеризация kmeans
def clustering(data, n_clusters):
    scalar = StandardScaler()
    X = scalar.fit_transform(data)
    print(any(map(lambda row: any(map(lambda el: el is None or el is float("-inf") or el is float("inf"), row)), X)))
    km = KMeans(n_clusters=n_clusters)

    clusters = km.fit_predict(data)
    print("k-means")
    print(clusters)
    return clusters


# кластеризация
def clustering2(data, n_clusters):
    data2 = data.to_numpy().tolist()

    birch_instance = birch(data2, n_clusters)

    birch_instance.process()

    objects_in_clusters = birch_instance.get_clusters()
    print("birch")
    print(objects_in_clusters)
    clusters = []
    for i in range(len(data2)):
        indexes = [i2 for i2, v in enumerate(objects_in_clusters) if v.count(i) > 0]
        clusters.append(indexes[0])

    print(clusters)
    return clusters


def str_to_indexes(values):
    indexes = {}
    for name in values:
        if name not in indexes:
            indexes[name] = len(indexes)

    return list(map(lambda v: indexes[v], values))


# нечеткая кластеризация
def fuzzy(data, n_clusters):
    fcm = FCM(n_clusters)
    fcm.fit(data[['oldpeak']])
    print("fuzzy")
    print(fcm.centers)
    print(fcm.u)

    res = np.transpose(fcm.u)

    for cl_num in range(n_clusters):
        vals = res[cl_num]
        plt.plot(range(len(vals)), vals, label=cl_num + 1)
    plt.show()
    return res


def calc_clusters_cnt(data):
    ElbowMethod(data)
    silhouette_method(data)
    dbi(data)


def main():
    file_name = "heart.txt"
    birds, data = load_data(file_name)
    calc_clusters_cnt(data)

    n_clusters = 2
    clusters1 = clustering(data, n_clusters)
    clusters2 = clustering2(data, n_clusters)
    fuzzy_props = fuzzy(data, n_clusters)

    types_indexes = str_to_indexes(birds['target'])
    corr_c1_c2 = scipy.stats.spearmanr(clusters1, clusters2)[0]
    corr_res_c2 = scipy.stats.spearmanr(types_indexes, clusters2)[0]
    corr_c1_res = scipy.stats.spearmanr(clusters1, types_indexes)[0]
    print("Корреляция между kmeans и birth = " + str(fabs(corr_c1_c2)))
    print("Корреляция между birth и исходными = " + str(fabs(corr_res_c2)))
    print("Корреляция между kmeans и исхоными = " + str(fabs(corr_c1_res)))

    data['test_types'] = birds['target']
    data['clusters1'] = clusters1
    data['clusters2'] = clusters2

    plt.scatter(data.sex, data.oldpeak, c=data.clusters1)
    plt.show()
    plt.scatter(data.sex, data.oldpeak, c=data.clusters2)
    plt.show()
    for i in range(n_clusters):
        data['fuzzy_prop' + str(i + 1)] = fuzzy_props[i]

    data = data.sort_values(by=['clusters1'])

    for i in range(n_clusters):
        vals = data['fuzzy_prop' + str(i + 1)]
        plt.plot(range(len(vals)), vals, label=i + 1)
    plt.show()

    data.to_csv('result.csv')


main()
