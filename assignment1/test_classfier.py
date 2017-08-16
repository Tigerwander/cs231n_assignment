from cs231n.classifiers import KNearestNeighbor
from cs231n.data_utils import get_CIFAR10_data
import numpy as np

if __name__ == '__main__':
    data = get_CIFAR10_data(num_training=1500)
    knn = KNearestNeighbor()
    knn.train(data['X_train'], data['y_train'])
    predict = knn.predict(data['X_test'])
