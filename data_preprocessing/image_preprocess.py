import tensorflow as tf
import utils.data_helper
import utils.global_var
import numpy as np
import cv2


def preprocess_for_train(image,
                         output_height,
                         output_width):
    resize_side = tf.random_uniform(
        [],
        minval=utils.global_var._RESIZE_SIDE_MIN,
        maxval=utils.global_var._RESIZE_SIDE_MIN+100,
        dtype=tf.int32
    )
    image = utils.data_helper._aspect_preserving_resize(image, resize_side)
    image = utils.data_helper._random_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    image = tf.image.random_flip_left_right(image)
    return utils.data_helper._mean_image_subtraction(image)


# def preprocess_for_test(image,
#                          output_height,
#                          output_width):
#     # 这个函数的作用就是按照原有比例对图片进行双线性插值缩放
#     # 这里的resize有个bug，如果说图片的size小于RESIZE_SIDE_MIN的话，就会变成图片的最大值为RESIZE_SIDE_MIN
#     image = utils.data_helper._aspect_preserving_resize(image, utils.global_var._RESIZE_SIDE_MIN)
#     #
#     image = utils.data_helper._central_crop([image], output_height, output_width)[0]
#     image.set_shape([output_height, output_width, 3])
#     image = tf.to_float(image)
#     return utils.data_helper._mean_image_subtraction(image)


def preprocess_for_test(image, output_height, output_width):
    min_side = output_height if output_height>output_width else output_width
    image = resize(image, min_side)
    image = central_crop(image, output_height, output_width)
    return image


def resize(image, min_side = utils.global_var._RESIZE_SIDE_MIN):
    '''
    对图片按照原比例进行resize
    :param image:
    :type image:
    :param min_side:
    :return:
    '''
    shape = image.shape
    height = float(shape[0])
    width = float(shape[1])
    min_side = float(min_side)
    if height >= min_side and width >= min_side:
        scale = min_side/height if height>width else min_side/width
    else:
        scale = min_side/height if height<width else min_side/width
    new_height = int(height*scale)
    new_width = int(width*scale)
    image = cv2.resize(image, (new_height, new_width))
    return image


def central_crop(image, output_height, output_width):
    '''
    中心裁剪
    :param image:
    :param output_height:
    :param output_width:
    :return:
    '''
    shape = image.shape
    origin_height = shape[0]
    origin_width = shape[1]
    offset_height = int((origin_height - output_height)/2)
    offset_width = int((origin_width - output_width)/2)
    crop_img = image[offset_height:(origin_height-offset_height), offset_width:(origin_width-offset_width)]
    return crop_img