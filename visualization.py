import tensorflow as tf
import cv2
import numpy as np


class ClassActivationMap(object):

    def __init__(self, saved_model):
        # restore tensorflow graph
        saver = tf.train.import_meta_graph('{}.meta'.format(saved_model))

        # restore Session
        self.sess = tf.Session()
        saver.restore(self.sess, saved_model)

        # reconstruct tensor
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.xs = tf.get_default_graph().get_tensor_by_name('xs:0')
        self.phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        self.phase_aug = tf.get_default_graph().get_tensor_by_name('phase_aug:0')
        self.top_conv = tf.get_default_graph().get_tensor_by_name('block_4/output:0')
        self.global_average_pool = tf.get_default_graph().get_tensor_by_name('global_average_pool/AvgPool:0')
        self.dense_weights = tf.get_default_graph().get_tensor_by_name('dense/kernel:0')

        h, w = self.xs.get_shape()[1:3]
        n_filters = self.dense_weights.get_shape()[0]

        # linear interpolate latent vector
        resize_images = tf.image.resize_bilinear(self.top_conv, size=[h, w])
        reshaped_images = tf.reshape(resize_images, [-1, n_filters])
        classmap = tf.matmul(reshaped_images, self.dense_weights)
        n_classes = classmap.get_shape()[-1]
        self.classmap = tf.reshape(classmap, [-1, h, w, n_classes])

        # Activation map Normalization
        min_value = tf.reshape(tf.reduce_min(self.classmap, axis=[1, 2]), [-1, 1, 1, n_classes])
        max_value = tf.reshape(tf.reduce_max(self.classmap, axis=[1, 2]), [-1, 1, 1, n_classes])
        self.classmap = (self.classmap - min_value) / (max_value - min_value)

    def get_cam(self, sample_images, label_indices):

        assert len(sample_images) == len(label_indices)
        cam_samples = self.sess.run(self.classmap,
                                    feed_dict={self.xs: sample_images, self.phase_train: False, self.phase_aug: False})

        overlay_images = []

        # overlay two images
        for ind, label_num in enumerate(label_indices):
            cmap = cv2.applyColorMap(np.uint8(cam_samples[ind][:, :, label_num] * 255), cv2.COLORMAP_JET)
            cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
            ori = np.uint8(sample_images[ind] * 255)

            overlay_img = cv2.addWeighted(ori, 0.5, cmap, 0.3, 0)
            overlay_images.append(overlay_img)

        return np.stack(overlay_images)


def generate_stitch_images(sample_images, stitch_h, stitch_w):
    assert len(sample_images) == stitch_h * stitch_w

    img_h, img_w = list(sample_images.shape)[1:3]
    shape = [stitch_h, stitch_w] + list(sample_images.shape)[1:]

    # color images
    if np.ndim(sample_images) == 4:
        sample_images = sample_images.reshape(shape).transpose(0, 2, 1, 3, 4)
        sample_images = sample_images.reshape(img_h * stitch_h, stitch_w * img_w, -1)
    # gray images
    elif np.ndim(sample_images) == 3:
        sample_images = sample_images.reshape(shape).transpose(0, 2, 1, 3)
        sample_images = sample_images.reshape(img_h * stitch_h, stitch_w * img_w)

    return sample_images

if __name__ == '__main__':
    # load cifar dataset
    import pickle
    import matplotlib.pyplot as plt
    f = open('test_batch', 'rb')
    cifar10_dict = pickle.load(f, encoding='bytes')
    images = cifar10_dict[b'data'].reshape([10000, 3, 32, 32]).transpose([0, 2, 3, 1]) / 255.
    labels = cifar10_dict[b'labels']

    # class activation map
    cam = ClassActivationMap('./Models/0.4841_0.8327-25000')
    act_images = cam.get_cam(images[:100], labels[:100])
    stitch_images = generate_stitch_images(act_images, 10, 10)
    plt.imshow(stitch_images)
    plt.show()

