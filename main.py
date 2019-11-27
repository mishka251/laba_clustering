import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from fcmeans import FCM
from seaborn import scatterplot as scatter


#---------------- Метод локтя ------------------------------------------------------------------
def ElbowMethod(X):
    distortions = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, #задаем число кластеров
                    init='k-means++', # Метод инициализации, {'k-means++', 'random' или ndarray}
                    n_init=10, #Количество раз, когда алгоритм k-means будет выполняться с разными значениями центров кластеров. Окончательные результаты будут самым лучшим выходом n_init последовательно работает в условиях инерции.
                    max_iter=300, # Максимальное число итераций алгоритма k-средних для a одиночная пробежка.
                    random_state=0) # Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.
        km.fit(X)
        distortions.append(km.inertia_)
    print('Искажение : %.2f ' % km.inertia_)
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Чиcлo кластеров')
    plt.ylabel('Искажение')
    plt.show()
    print("ElbowMethod")

def load_data(file_name):
    birds = pd.read_csv(file_name).dropna(axis=1, how='all')
    print(birds)
    birds = birds.dropna()
    data = birds.drop('type', axis='columns')
    data = data.drop('id', axis='columns')

    return birds, data


def clustering(data, n_clusters):
    scalar = StandardScaler()
    X = scalar.fit_transform(data)
    print(any(map(lambda row: any(map(lambda el: el is None or el is float("-inf") or el is float("inf"), row)), X)))
    km = KMeans(n_clusters=n_clusters)

    print(X)
    print(np.any(np.isnan(X)))
    print(np.any(np.isinf(X)))

    clusters = km.fit_predict(data)
    data['cluster'] = clusters


def add_indexes(data, values):
    indexes = {}
    for name in values:
        if name not in indexes:
            indexes[name] = len(indexes)

    data['res_indexes'] = list(map(lambda type: indexes[type], data['test_result']))


def fuzzy(data, n_clusters):
    fcm = FCM(n_clusters)
    fcm.fit(data)

    print(fcm.centers)
    print(fcm.u)
    # plot result
    #% matplotlib
    #inline
    #f, axes = plt.subplots(1, 2, figsize=(11, 5))
    #scatter(data[:, 0], data1[:, 1], ax=axes[0])
    #scatter(X[:, 0], X[:, 1], ax=axes[1], hue=fcm_labels)
    #scatter(fcm_centers[:, 0], fcm_centers[:, 1], ax=axes[1], marker="s", s=200)
    #plt.show()

def main():
    file_name = "bird.csv"
    birds, data = load_data(file_name)
    ElbowMethod(data)
    clustering(data, n_clusters=5)
    fuzzy(data, n_clusters=5)
    data['test_result'] = birds['type']
    add_indexes(data, birds['type'])

    print(data)

    print(np.corrcoef(data['cluster'], data['res_indexes']))

    #fuzzy(data, n_clusters=5)


main()
