import tensorflow as tf
import config
import data_preprocessing
import net
import accuracy.default_accuracy as default_accuracy
import accuracy.multi_accuracy as multi_accuracy
import data_scraping.picDevider as picDevider
import data_convertor.process_img as process_img
import utils
import utils.data_helper
import sys
import config.chamo
import config.chamo_full_run
import data_preprocessing.test_preprocess
import net.vgg16
import net.mobilenet_v2
import matplotlib
import matplotlib.pyplot as plt
import data_scraping.materil_name_73
import numpy as np
import time
import os
from utils.data_helper import *
import shutil
import tf_cnnvis.tf_cnnvis as tf_cnnvis
import random
import cv2
import utils.plot_show as plot_show


excel_path = '/home/leo/Documents/chamo/food_material/V1.1.0.0525.xlsx'

pic_src_path = '/home/leo/Downloads/chamo/alltest/alltest_devider/chamo_00000/'

#
pic_des_path_devider = 'E:/test_data/divide_path/similarPic/cp/'

pic_merge_path_train = 'E:/material_merge/'

label_dim = len(picDevider.materials)

RIGHT = 0
PART = 1
ERROR = 2
PIC_SAVE_ROOT_PATH = 'E:/test_data/divide_path_20180621_1000/'

def get_config(config_name):
    print('choose config: '+config_name)
    config_obj=None
    if config_name=='chamo':
        config_obj=config.chamo.get_config()
    elif config_name=='chamo_full_run':
        config_obj = config.chamo_full_run.get_config()
    return config_obj


def eval_smooth_show(config_obj, one_pic_path=None, isDeconv=False):
    test_preprocess_obj=data_preprocessing.test_preprocess.test_preprocess(
        config_obj.tfrecord_test_addr, config_obj.class_num)
    net_name=config_obj.net_type
    test_net_obj=None
    if net_name=='vgg16':
        test_net_obj=net.vgg16.vgg16(False, 'vgg16', config_obj.class_num)
    elif net_name=='mobilenet_v2':
        test_net_obj=net.mobilenet_v2.mobilenet_v2(False, 'mobilenet_v2', config_obj.class_num)

    accu_name=config_obj.accuracy_type
    accu_obj=None
    if accu_name=='default':
        accu_obj=default_accuracy.default_accuracy()
    elif accu_name=='multi':
        accu_obj=multi_accuracy.multi_accuracy()

    images_test, labels_test = test_preprocess_obj.def_preposess(batch_size=1)
    net_test = test_net_obj.def_net(images_test)
    inputs = tf.sigmoid(net_test)
    predict = tf.cast(inputs > 0.1, tf.float32)
    accuracy_TOTAL = accu_obj.def_accuracy(net_test, labels_test)

    saver = tf.train.Saver()
    img_mean = utils.global_var.means
    sess = tf.Session()

    writer = tf.summary.FileWriter("Log/", tf.get_default_graph())
    with sess.as_default():
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver.restore(sess, config_obj.result_addr)

        [accu_TOTAL, predict_v, labels_test_v, images_test_v] \
            = sess.run([accuracy_TOTAL, predict, labels_test, images_test])
        print('len(predict_V):', len(predict_v))
        for k in range(len(predict_v)):
            #print(labels_test_v[k])
            mat_show = []
            for i in range(len(predict_v[k])):
                if predict_v[k][i] == 1:
                    mat_show.append(data_scraping.materil_name_73.material_list[i])
                    print(str(data_scraping.materil_name_73.material_list[i]))
            print(predict_v[k])
            show_img = images_test_v[k]
            show_img = show_img+img_mean
            show_img = abs(show_img) / 256.0
            plt.imshow(show_img)
            # zhfont = matplotlib.font_manager.FontProperties(
            # fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
            for i in range(len(mat_show)):
                plt.text(150, 25*(i+1), str(mat_show[i]), fontproperties='Simhei', fontsize=15, color='red')
            plt.show()

        coord.request_stop()
        coord.join(threads)
    writer.close()
    if isDeconv:
        #show_cnnvis(sess, feed_dict={X: image_list, Y: tempY}, input_tensor=images_test)
        show_cnnvis(sess, feed_dict={}, input_tensor=images_test)


def eval_smooth_tensorboard(config_obj, one_pic_path):
    net_name=config_obj.net_type
    test_net_obj=None
    if net_name=='vgg16':
        test_net_obj=net.vgg16.vgg16(False, 'vgg16', config_obj.class_num)
    elif net_name=='mobilenet_v2':
        test_net_obj=net.mobilenet_v2.mobilenet_v2(False, 'mobilenet_v2', config_obj.class_num)

    accu_name=config_obj.accuracy_type
    accu_obj=None
    if accu_name=='default':
        accu_obj=default_accuracy.default_accuracy()
    elif accu_name=='multi':
        accu_obj=multi_accuracy.multi_accuracy()

    images_test_p = tf.placeholder(shape=[config_obj.batchsize, 224, 224, 3], dtype=tf.float32)
    labels_test_p = tf.placeholder(shape=[config_obj.batchsize, config_obj.class_num], dtype=tf.float32)
    images_test, labels_test = read_a_pic(one_pic_path, config_obj.class_num)
    net_test = test_net_obj.def_net(images_test_p)
    inputs = tf.sigmoid(net_test)
    predict = tf.cast(inputs > 0.1, tf.float32)
    accuracy_TOTAL = accu_obj.def_accuracy(net_test, labels_test_p, 0.5)

    saver = tf.train.Saver()
    img_mean = utils.global_var.means
    sess = tf.Session()

    writer = tf.summary.FileWriter("Log/", tf.get_default_graph())
    with sess.as_default():
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver.restore(sess, config_obj.result_addr)
        feed_dict = {images_test_p: images_test, labels_test_p: labels_test}

        [accu_TOTAL, predict_v] \
            = sess.run([accuracy_TOTAL, predict], feed_dict=feed_dict)
        print('len(predict_V):', len(predict_v))
        for k in range(len(predict_v)):
            # print(labels_test_v[k])
            mat_show = []
            for i in range(len(predict_v[k])):
                if predict_v[k][i] == 1:
                    mat_show.append(data_scraping.materil_name_73.material_list[i])
                    print(str(data_scraping.materil_name_73.material_list[i]))
            print(predict_v[k])
            show_img = images_test[k]
            show_img = show_img+img_mean
            show_img = abs(show_img) / 256.0
            plt.imshow(show_img)
            # zhfont = matplotlib.font_manager.FontProperties(
            # fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
            for i in range(len(mat_show)):
                plt.text(150, 25*(i+1), str(mat_show[i]), fontproperties='Simhei', fontsize=15, color='red')
            plt.show()

        coord.request_stop()
        coord.join(threads)
    writer.close()
    show_cnnvis(sess, feed_dict=feed_dict, input_tensor=images_test_p)


def eval_smooth(config_obj, repeat_time, threshold=0.5):
    test_preprocess_obj = data_preprocessing.test_preprocess.test_preprocess(config_obj.tfrecord_test_addr,
                                                                             config_obj.class_num)
    net_name = config_obj.net_type
    test_net_obj = None
    if net_name == 'vgg16':
        test_net_obj = net.vgg16.vgg16(False, 'vgg16', config_obj.class_num)
    elif net_name == 'mobilenet_v2':
        test_net_obj = net.mobilenet_v2.mobilenet_v2(False, 'mobilenet_v2', config_obj.class_num)

    accu_name = config_obj.accuracy_type
    accu_obj = None
    if accu_name == 'default':
        accu_obj = default_accuracy.default_accuracy()
    elif accu_name == 'multi':
        accu_obj = multi_accuracy.multi_accuracy()

    images_test, labels_test = test_preprocess_obj.def_preposess()
    net_test = test_net_obj.def_net(images_test)
    inputs = tf.sigmoid(net_test)
    predict = tf.cast(inputs > threshold, tf.float32)
    accuracy_perfect, accuracy, precision, recall, f1, acc_list, \
        pre_list, pre_list_nume, pre_list_deno, rec_list, \
        rec_list_nume, rec_list_deno = accu_obj.def_accuracy(net_test, labels_test, threshold)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver.restore(sess, config_obj.result_addr)
        sess.graph.finalize()
        # len_all = len(labels_test)
        each_size = labels_test.get_shape().as_list()[0]
        len_all = labels_test.get_shape().as_list()[1]

        acc_perfect_all = 0.0
        acc_all = 0.0
        precision_all = 0.0
        recall_all = 0.0
        f1_all = 0.0

        acc_list_all = np.zeros(shape=[len_all], dtype=np.float32)
        precision_all_nume = np.zeros(shape=[len_all], dtype=np.float32)
        precision_all_deno = np.zeros(shape=[len_all], dtype=np.float32)
        recall_all_nume = np.zeros(shape=[len_all], dtype=np.float32)
        recall_all_deno = np.zeros(shape=[len_all], dtype=np.float32)

        for repeat_i in range(1, repeat_time+1):
            accuracy_perfect_v, accuracy_v, precision_v, recall_v, f1_v, acc_list_v, pre_list_nume_v, pre_list_deno_v, \
            rec_list_nume_v, rec_list_deno_v, predict_v, labels_test_v, images_test_v = sess.run(
                [accuracy_perfect, accuracy, precision, recall, f1, acc_list,
                 pre_list_nume, pre_list_deno, rec_list_nume, rec_list_deno, predict, labels_test,
                 images_test])

            acc_perfect_all = acc_perfect_all + accuracy_perfect_v
            acc_all = acc_all + accuracy_v
            precision_all = precision_all + precision_v
            recall_all = recall_all + recall_v
            f1_all = f1_all + f1_v

            acc_list_all = np.nan_to_num(acc_list_all) + acc_list_v
            precision_all_nume = precision_all_nume + pre_list_nume_v
            precision_all_deno = precision_all_deno + pre_list_deno_v
            recall_all_nume = recall_all_nume + rec_list_nume_v
            recall_all_deno = recall_all_deno + rec_list_deno_v

            repeat_i = float(repeat_i)
            print('step: %d total pictures: %d' % (repeat_i, each_size*repeat_i))
            print('accuracy_prefect_v:', acc_perfect_all/repeat_i)
            print('accuracy_v: ', acc_all/repeat_i)
            print('precision:', precision_all/repeat_i)
            print('recall:', recall_all/repeat_i)
            print('f1:', f1_all/repeat_i)
            print('acc_list_v:', acc_list_all/repeat_i)
            print('pre_list_v:', precision_all_nume/precision_all_deno)
            print('rec_list_v:', recall_all_nume/recall_all_deno)

        coord.request_stop()
        coord.join(threads)
    repeat_time = float(repeat_time)
    acc_perfect_all = acc_perfect_all/repeat_time
    acc_all = acc_all/repeat_time
    precision_all = precision_all/repeat_time
    recall_all = recall_all/repeat_time
    f1_all = f1_all/repeat_time
    acc_list_all = acc_list_all/repeat_time
    precision_list_all = precision_all_nume/precision_all_deno
    recall_list_all = recall_all_nume/recall_all_deno
    return acc_perfect_all, acc_all, precision_all, recall_all, f1_all, acc_list_all, precision_list_all, recall_list_all


def eval_smooth_divide(config_obj, des_path, repeat_time):
    print('read tfrecord from:', config_obj.tfrecord_test_addr)
    RIGHT_PATH = des_path + 'right/'
    PART_PATH = des_path + 'part/'
    ERROR_PATH = des_path + 'error/'
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    if not os.path.exists(RIGHT_PATH):
        os.makedirs(RIGHT_PATH)
    if not os.path.exists(PART_PATH):
        os.makedirs(PART_PATH)
    if not os.path.exists(ERROR_PATH):
        os.makedirs(ERROR_PATH)

    test_preprocess_obj = data_preprocessing.test_preprocess.test_preprocess(config_obj.tfrecord_test_addr,
                                                                             config_obj.class_num)
    net_name = config_obj.net_type
    test_net_obj = None
    if net_name == 'vgg16':
        test_net_obj = net.vgg16.vgg16(False, 'vgg16', config_obj.class_num)
    elif net_name == 'mobilenet_v2':
        test_net_obj = net.mobilenet_v2.mobilenet_v2(False, 'mobilenet_v2', config_obj.class_num)

    accu_name = config_obj.accuracy_type
    accu_obj = None
    if accu_name == 'default':
        accu_obj = default_accuracy.default_accuracy()
    elif accu_name == 'multi':
        accu_obj = multi_accuracy.multi_accuracy()

    images_test, labels_test = test_preprocess_obj.def_preposess()
    labels_right = labels_test
    net_test = test_net_obj.def_net(images_test)
    inputs = tf.sigmoid(net_test)
    predict = tf.cast(inputs > 0.5, tf.float32)
    accuracy_TOTAL = accu_obj.def_accuracy(net_test, labels_test)

    saver = tf.train.Saver()
    img_mean = utils.global_var.means
    sess = tf.Session()
    with sess.as_default():
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver.restore(sess, config_obj.result_addr)
        batch_size = 0
        for repeat_time_i in range(repeat_time):
            [accu_TOTAL, predict_v, labels_test_v, images_test_v] = sess.run(
                [accuracy_TOTAL, predict, labels_right, images_test])
            print('len(predict_V):', len(predict_v))
            batch_size = len(predict_v)
            for k in range(len(predict_v)):
                # print(labels_test_v[k])
                mat_predict = []
                mat_right = []
                for i in range(len(predict_v[k])):
                    if predict_v[k][i] == 1:
                        mat_predict.append(data_scraping.materil_name_73.material_list[i])
                        print(str(data_scraping.materil_name_73.material_list[i]))
                print('===========')
                for m in range(len(labels_test_v[k])):
                    if labels_test_v[k][m] == 1:
                        mat_right.append(data_scraping.materil_name_73.material_list[m])
                        print(str(data_scraping.materil_name_73.material_list[m]))
                print(predict_v[k])
                show_img = images_test_v[k]
                show_img = show_img + img_mean
                show_img = abs(show_img) / 256.0
                ret_num = is_right(predict_v[k], labels_test_v[k])
                if ret_num == RIGHT:
                    tag_pic_and_save(show_img, mat_predict, RIGHT_PATH, mat_right)
                elif ret_num == PART:
                    tag_pic_and_save(show_img, mat_predict, PART_PATH, mat_right)
                else:
                    tag_pic_and_save(show_img, mat_predict, ERROR_PATH, mat_right)
                #plt.show()
        coord.request_stop()
        coord.join(threads)
        print('eval done, batch_size: %d; repeat_time: %d; total eval: %d'
              % (batch_size, repeat_time, batch_size*repeat_time))


def is_right(pre_label, label):
    '''
    判断一个图片输出结果(向量)是完全正确/部分正确/完全错误
    :param pre_label:
    :param label:
    :return:
    '''
    pre_label = pre_label.astype(dtype=np.int32)
    label = label.astype(dtype=np.int32)
    pre_label_sum = np.sum(pre_label)
    label_sum = np.sum(label)
    merge_sum = np.sum(pre_label & label)
    if (pre_label_sum == label_sum) and (pre_label_sum == merge_sum):
        return RIGHT
    elif merge_sum > 0:
        return PART
    else:
        return ERROR


def tag_pic_and_save(show_img, tag_list, root_path, correct_tag_list):
    temp_path = os.path.basename(root_path)+'/temp/'
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    temp_path = temp_path + 'temp.jpg'
    plt.imshow(show_img)
    pic_path = root_path
    for i in range(len(tag_list)):
        plt.text(150, 25 * (i + 1), str(tag_list[i]), fontproperties='Simhei', fontsize=16, color='red')
        pic_path = pic_path + str(tag_list[i]) + '_'
    pic_path = pic_path + str(time.time()) + '_'
    for i in correct_tag_list:
        pic_path = pic_path + str(i) + '_'
    pic_path = pic_path + '.jpg'
    plt.savefig(temp_path)
    print('save to:', pic_path)
    plt.close('all')
    exists, _ = is_file_exist(temp_path, root_path)
    if not exists:
        shutil.copy(temp_path, pic_path)
    os.remove(temp_path)


def show_cnnvis(sess, feed_dict, input_tensor):
    # deconv visualization
    layers = ["r", "p", "c"]
    total_time = 0

    start = time.time()
    is_success = tf_cnnvis.deconv_visualization(
        sess_graph_path=sess, value_feed_dict=feed_dict,
        input_tensor=input_tensor, layers=layers,
        path_logdir=os.path.join("Log", "MobilenetV2"),
        path_outdir=os.path.join("Output", "MobilenetV2"))
    start = time.time() - start
    print("Total Time = %f" % (start))


def eval_one_pic_path(pic_root_path, label_dim):
    father = os.path.abspath(os.path.join(os.path.dirname(pic_root_path), os.path.pardir))
    temp = father + '/temp/'
    tfrecord_src = temp + 'temp/'
    num = 0
    if not os.path.exists(temp):
        os.makedirs(temp)
    if not os.path.exists(tfrecord_src):
        os.makedirs(tfrecord_src)
    for root, dirs, files in os.walk(pic_root_path):
        for file in files:
            file = os.path.join(root, file)
            pic_rename = rand_tag_pic(file, num)
            num = num + 1
            shutil.copy(pic_rename, tfrecord_src)
    tfrecord_temp = father + '/temp_tfrecord/'
    if not os.path.exists(tfrecord_temp):
        os.makedirs(tfrecord_temp)
    process_img.check_and_convert(temp, tfrecord_temp, label_dim, 6)
    config_obj = get_config('chamo_full_run')
    config_obj.tfrecord_test_addr = tfrecord_temp
    eval_smooth_show(config_obj)


def eval_one_pic(pic_path, label_dim):
    father = os.path.abspath(os.path.join(os.path.dirname(pic_path), os.path.pardir))
    temp = father + '/temp/'
    tfrecord_src = temp + 'temp/'
    num = 0
    if not os.path.exists(temp):
        os.makedirs(temp)
    if os.path.exists(tfrecord_src):
        shutil.rmtree(tfrecord_src)
    if not os.path.exists(tfrecord_src):
        os.makedirs(tfrecord_src)
    shutil.copy(pic_path, tfrecord_src)
    new_pic_path = tfrecord_src + os.path.basename(pic_path)
    new_pic_path = rand_tag_pic(new_pic_path, 0)
    tfrecord_temp = father + '/temp_tfrecord/'
    if not os.path.exists(tfrecord_temp):
        os.makedirs(tfrecord_temp)
    process_img.check_and_convert(temp, tfrecord_temp, label_dim, 6)
    config_obj = get_config('chamo_full_run')
    config_obj.tfrecord_test_addr = tfrecord_temp
    eval_smooth_show(config_obj, new_pic_path, True)



def eval_one_pic_tensorboard(pic_path, label_dim):
    config_obj = get_config('chamo_full_run')
    config_obj.batchsize = 1
    eval_smooth_tensorboard(config_obj, pic_path)

def read_a_pic(pic_path, dim):
    img = cv2.imread(pic_path)
    img = cv2.resize(img, dsize=(224, 224))
    img_list = []
    img_list.append(img)
    pic_base = os.path.basename(pic_path)
    items = pic_base.split('_')
    if len(items) > 2:
        label = utils.data_helper.num2label(int(items[1]), dim)
        labels = []
        labels.append(label)
    else:
        labels = np.zeros([1, dim], dtype=np.float32)
    return img_list, labels


def rand_tag_pic(pic_path, num):
    randnamebase = 'test' + str(time.time()) + '_' + str(num) + '_' + 'test.jpg'
    randname = os.path.dirname(pic_path) + '/' + randnamebase
    os.rename(pic_path, randname)
    return randname


def threshold_show():
    x_list = []
    acc_list = []
    pre_list = []
    rec_list = []
    f1_list = []
    y_list = []
    for i in range(1, 100, 1):
        x = i / 100.0
        print('threshold:', x)
        acc_perfect_all, acc_all, precision_all, recall_all, f1_all, acc_list_all, precision_list_all, recall_list_all \
            = eval_smooth(config_obj, 100, x)
        x_list.append(x)
        acc_list.append(acc_all)
        pre_list.append(precision_all)
        rec_list.append(recall_all)
        f1_list.append(f1_all)
    y_list.append(acc_list)
    y_list.append(pre_list)
    y_list.append(rec_list)
    y_list.append(f1_list)
    plot_show.show_plot(x_list, y_list, y_name_list=['acc', 'pre', 'rec', 'f1'], color_list=plot_show.COLOR_LIST)


if __name__ == '__main__':
    config_obj = get_config('chamo_full_run')
    # 自己要设置临时路径就打开这个
    # pic_des_path_tfrecord = 'E:/test_data/divide_path/similarPic/tfrecord/'
    # config_obj.tfrecord_test_addr = pic_des_path_tfrecord
    # PIC_SAVE_ROOT_PATH = 'E:/test_data/divide_path_20180621/'


    # picDevider.devide_pic(pic_src_path, pic_des_path_devider, excel_path, picDevider.materials, 1000)
    # process_img.check_and_convert(pic_des_path_devider, pic_des_path_tfrecord, label_dim, 6)
    # threshold_show()
    # eval_smooth_divide(config_obj, PIC_SAVE_ROOT_PATH, 5)
    eval_pic_path = 'E:/test_data/myevalpics/pics1/2.jpg'
    eval_one_pic_tensorboard(eval_pic_path, label_dim)


