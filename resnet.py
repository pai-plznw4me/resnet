import tensorflow as tf
import os


class Resnet(object):
    def __init__(self, root_folder, phase_train):
        self.root_folder = root_folder
        self.phase_train = phase_train
        self.sess = None
        self.train_writer = None
        self.test_writer = None
        self.saver = None

    def batch_norm_(self, x, scope='bn'):
        """
        Batch normalization on convolutional maps.
        Args:
            x:           Tensor, 4D BHWD input maps
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """

        n_out = x.get_shape()[-1]
        with tf.variable_scope(scope):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                name='gamma', trainable=True)
            if len(x.get_shape()) == 4:
                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            elif len(x.get_shape()) == 2:
                batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')

            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(self.phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean),
                                         ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def residual_block(self, input_xs, n_skip, filters, block_name):
        """
        residual network 의 residual block을 구현합니다.
        Args:
            input_xs:    tensor, 4D dimension BHWC
            n_skip:      interger, 몇 개의 block 을 건너뛸지.
            filters: 각 레이어 당 filter 갯수
            block_name : redisual block 의 이름

        Return:
            layer:      Tensor, 4D Dimension BHWC
        """
        with tf.variable_scope(block_name):
            layer = input_xs
            for index in range(n_skip):
                layer_name = 'layer_{}'.format(index)
                with tf.variable_scope(layer_name):
                    layer = tf.layers.Conv2D(filters, 3, 1, 'same', activation=None,
                                             use_bias=False)(layer)
                    layer = self.batch_norm_(layer)
                    layer = tf.nn.relu(layer)

            # Projection Layer
            if input_xs.shape[-1] != filters:
                input_xs = tf.layers.Conv2D(filters, 1, 1, 'same',
                                            activation=tf.nn.relu)(input_xs)

            return layer + input_xs

    @staticmethod
    def get_loss(ys, logits):
        l2_reg = tf.reduce_sum([tf.reduce_sum(var**2, axis=None) for var in tf.get_collection('ws')])
        l2_beta = 0.0005

        # L2 reularization
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ys, logits=logits)
        loss = tf.reduce_mean(loss, name='loss') + l2_reg*l2_beta
        loss = tf.identity(loss, 'loss')
        print('loss tensor was added to Graph')
        return loss

    @staticmethod
    def get_accuracy(ys, logits):
        logits_cls = tf.argmax(logits, axis=1)
        logits_cls = tf.cast(logits_cls, dtype=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(ys, logits_cls), tf.float32), name='accuracy')
        print('accuracy tensor was added to Graph')
        return acc

    @staticmethod
    def scalar_to_tensorboard(**kwargs):
        for key, value in kwargs.items():
            tf.summary.scalar(name=key, tensor=value)

        merged = tf.summary.merge_all()
        print('summary tensor was added to Graph')
        return merged

    def generate_session(self, fraction_ratio=0.5):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = fraction_ratio
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        self.sess.run(init)
        return self.sess
        print('Session was Created')

    def generate_tensorwriter(self):
        # tensorboard
        self.train_writer = tf.summary.FileWriter(logdir='{}/log/train'.format(self.root_folder))
        self.test_writer = tf.summary.FileWriter(logdir='{}/log/test'.format(self.root_folder))
        print('train and test writer was generated')
        return self.train_writer, self.test_writer

    def generator_saver(self):
        # saver
        self.saver = tf.train.Saver(max_to_keep=10)
        save_root_folder = '{}/model'.format(self.root_folder)
        os.makedirs(save_root_folder, exist_ok=True)
        print('Saver was generated')
        return self.saver
