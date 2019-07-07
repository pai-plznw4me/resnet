import tensorflow as tf


def image_augmentation(image, is_training, crop_h, crop_w):

    def _aug_with_train(input_x, crop_height, crop_width):
        img_h, img_w, ch = list(map(int, input_x.get_shape()[:]))

        pad_w = int(img_h * 0.2)
        pad_h = int(img_w * 0.2)

        input_x = tf.image.resize_image_with_crop_or_pad(input_x, img_h+pad_h, img_w+pad_w)
        input_x = tf.random_crop(input_x, [crop_height, crop_width, ch])
        input_x = tf.image.random_flip_left_right(input_x)
        input_x = tf.image.random_flip_up_down(input_x)

        input_x = tf.image.random_contrast(input_x, lower=0.2, upper=2.0)
        input_x = tf.image.random_brightness(input_x, max_delta=63. / 255.)
        input_x = tf.image.random_saturation(input_x, lower=0.5, upper=1.8)
        input_x = tf.image.per_image_standardization(input_x)
        return input_x

    def _aug_with_test(input_x, crop_height, crop_width):

        input_x = tf.image.resize_image_with_crop_or_pad(input_x, crop_height, crop_width)
        input_x = tf.image.per_image_standardization(input_x)
        return input_x

    image = tf.cond(is_training,
                    lambda: _aug_with_train(image, crop_h, crop_w),
                    lambda: _aug_with_test(image, crop_h, crop_w))
    return image


def images_augmentation(images, phase_train):
    with tf.name_scope('augmentation'):
        crop_h, crop_w = list(map(int, images.get_shape()[1:3]))
        images = tf.map_fn(lambda image: image_augmentation(image, phase_train, crop_h, crop_w),
                           images)
        return images
