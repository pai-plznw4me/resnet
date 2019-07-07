import random
from keras.datasets import cifar10


class DataProvider(object):
    def __init__(self, images, labels):
        self.n_sample = len(labels)
        self.queue = list(range(self.n_sample))
        random.shuffle(self.queue)

        self.images = images
        self.labels = labels
        self.epoch_count = 0

    def next_batch(self, batch_size):
        if len(self.queue) < batch_size:
            self.queue = list(range(self.n_sample))
            self.epoch_count += 1
        target_indices = self.queue[:batch_size]
        del self.queue[:batch_size]
        return self.images[target_indices], self.labels[target_indices]


class Cifar10Provider(DataProvider):

    def __init__(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Data reshape
        self.y_train, self.y_test = [y.reshape([-1]) for y in [y_train, y_test]]

        # Data Normalization
        self.x_train, self.x_test = [x / 255. for x in [x_train, x_test]]

        super(Cifar10Provider, self).__init__(self.x_train, self.y_train)

        print('image shape : {}, label shape : {} '.format(x_train.shape, y_train.shape))
        print('image shape : {}, label shape : {} '.format(x_test.shape, y_test.shape))
        print('train minimun : {}, train_maximum : {} '.format(x_train.min(), x_train.max()))
        print('tests minimun : {}, test_maximum : {} '.format(x_test.min(), x_test.max()))
