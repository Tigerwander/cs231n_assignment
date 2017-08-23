from cs231n.classifiers import KNearestNeighbor
from cs231n.data_utils import get_CIFAR10_data
import numpy as np

if __name__ == '__main__':
    data = get_CIFAR10_data(num_training=5000)
    knn = KNearestNeighbor()
    num_training = data['X_train'].shape[0]
    train = data['X_train'].reshape(num_training, 32 * 32 * 3)
    knn.train(train, data['y_train'])

    num_test = data['X_test'].shape[0]
    test = data['X_test'].reshape(num_test, 32 * 32 * 3)
    predict = knn.predict(test)
    accuracy = np.sum(predict == data['y_test'])
    print('accuracy {}'.format(accuracy/ float(num_test)))
