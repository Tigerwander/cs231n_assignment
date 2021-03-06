from cs231n.classifiers import KNearestNeighbor
from cs231n.classifiers import LinearSVM
from cs231n.classifiers import Softmax
from cs231n.classifiers.neural_net import TwoLayerNet
from cs231n.data_utils import get_CIFAR10_data
import numpy as np

def test_knn(data):
    knn = KNearestNeighbor()
    num_training = data['X_train'].shape[0]
    train = data['X_train'].reshape(num_training, 32 * 32 * 3)
    print('training...')
    knn.train(train, data['y_train'])

    num_test = data['X_test'].shape[0]
    test = data['X_test'].reshape(num_test, 32 * 32 * 3)
    predict = knn.predict(test)
    accuracy = np.sum(predict == data['y_test'])
    print('knn accuracy {}'.format(accuracy/ float(num_test)))

def test_svm(data):
    svm = LinearSVM()
    num_training = data['X_train'].shape[0]
    train = data['X_train'].reshape(num_training, 32 * 32 * 3)
    print('training...')
    svm.train(train, data['y_train'], num_iters=20000, verbose=True, reg=1e-3)

    num_test = data['X_test'].shape[0]
    test = data['X_test'].reshape(num_test, 32 * 32 * 3)
    predict = svm.predict(test)
    accuracy = np.mean(predict == data['y_test'])
    print('svm predict accuracy {}'.format(accuracy))

def test_softmax(data):
    softmax = Softmax()
    num_training = data['X_train'].shape[0]
    train = data['X_train'].reshape(num_training, 32 * 32 * 3)
    print('training...')
    softmax.train(train, data['y_train'], num_iters=1000, verbose=True, reg=1e-3)

    num_test = data['X_test'].shape[0]
    test = data['X_test'].reshape(num_test, 32 * 32 * 3)
    predict = softmax.predict(test)
    accuracy = np.mean(predict == data['y_test'])
    print('softmax predict accuracy {}'.format(accuracy))

def test_nn(data):
    nn = TwoLayerNet(32 * 32 * 3, 1000, 10)
    num_training = data['X_train'].shape[0]
    X_train = data['X_train'].reshape(num_training, 32 * 32 * 3)
    y_train = data['y_train']

    num_val = data['X_val'].shape[0]
    X_val = data['X_val'].reshape(num_val, 32 * 32 * 3)
    y_val = data['y_val']

    print('training...')
    nn.train(X_train, y_train, X_val, y_val, verbose=True, num_iters=1000)

    num_test = data['X_test'].shape[0]
    X_test = data['X_test'].reshape(num_test, 32 * 32 * 3)
    y_test = data['y_test']
    predict = nn.predict(X_test)
    accuracy = np.mean(predict == y_test)
    print('softmax predict accuracy {}'.format(accuracy))


if __name__ == '__main__':
    data = get_CIFAR10_data(num_training=5000)

    #test_knn(data)
    #test_svm(data)
    #test_softmax(data)
    test_nn(data)
