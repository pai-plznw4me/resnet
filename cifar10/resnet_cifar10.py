import tensorflow as tf
from resnet import Resnet
from augmentator import images_augmentation
#from dataprovider import Cifar10Provider


class ResnetCifar10(Resnet):

    def __init__(self):
        # define input placeholder
        self.n_classes = 10
        self.xs = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='xs')
        self.ys = tf.placeholder(shape=[None], dtype=tf.int32, name='ys')
        self.lr = tf.placeholder(shape=[], dtype=tf.float32, name='lr')
        self.phase_train = tf.placeholder(shape=[], dtype=tf.bool, name='phase_train')
        self.phase_aug = tf.placeholder(shape=[], dtype=tf.bool, name='phase_aug')

        self.global_step = tf.train.create_global_step(graph=None)
        # Data Provider
        # self.cifar10_provider = Cifar10Provider()
        super(ResnetCifar10, self).__init__('resnet', self.phase_train)

        # Augmentation
        self.xs_aug = images_augmentation(self.xs, self.phase_aug)

        # Model Implement
        self.logits = self.model_a()

        # cost function
        self.loss = self.get_loss(self.ys, self.logits)

        # metric
        self.acc = self.get_accuracy(self.ys, self.logits)

        # add tensor to tensorboard
        self.merged = self.scalar_to_tensorboard(acc=self.acc, loss=self.loss)

        # train_optimizer
        self.train_op = tf.train.MomentumOptimizer(self.lr, 0.9).minimize(self.loss, global_step=self.global_step)

        # tensorboard writer
        self.train_writer, self.test_writer = self.generate_tensorwriter()

        # generate saver
        self.saver = self.generator_saver()

    def model_a(self):
        # Stem
        layer = tf.layers.Conv2D(16, 3, 1, 'same', activation=tf.nn.relu, name='stem')(self.xs_aug)
        # Residual
        layer = self.residual_block(layer, 2, 16, 'block_1')
        layer = tf.layers.MaxPooling2D(2, 2)(layer)
        layer = self.residual_block(layer, 2, 32, 'block_2')
        layer = tf.layers.MaxPooling2D(2, 2)(layer)
        layer = self.residual_block(layer, 2, 64, 'block_3')
        layer = tf.layers.MaxPooling2D(2, 2)(layer)
        layer = self.residual_block(layer, 2, 128, 'block_4')
        layer = tf.layers.AveragePooling2D(4, 1)(layer)
        layer = tf.layers.Flatten()(layer)

        logits = tf.layers.Dense(self.n_classes)(layer)
        return tf.identity(logits, 'logits')

    def training(self, batch_size, lr):
        for i in range(batch_size):
            batch_xs, batch_ys = self.cifar10_provider.next_batch(batch_size)
            train_feed = {self.xs: batch_xs, self.ys: batch_ys, self.lr: lr, self.phase_train: True}
            self.sess.run(self.train_op, feed_dict=train_feed)

    def eval(self, global_step):
        """
        1. eval에서 나온 acc, loss 을 tensorboard 에 추가한다
        :return:
        """
        n_sample = len(self.cifar10_provider.y_test)
        train_xs, train_ys = self.cifar10_provider.next_batch(n_sample)
        eval_xs, eval_ys = self.cifar10_provider.x_test, self.cifar10_provider.y_test

        eval_fetch = [self.merged, self.loss, self.acc]
        eval_feed = {self.xs: train_xs, self.ys: train_ys, self.phase_train: False}
        train_feed = {self.xs: eval_xs, self.ys: eval_ys, self.phase_train: False}
        train_merged, train_cost, train_acc = self.sess.run(eval_fetch, feed_dict=eval_feed)
        test_merged, test_cost, test_acc = self.sess.run(eval_fetch, feed_dict=train_feed)

        self.train_writer.add_summary(train_merged, global_step=global_step)
        self.test_writer.add_summary(test_merged, global_step=global_step)


if __name__ == '__main__':
    resnet_cifar10 = ResnetCifar10()