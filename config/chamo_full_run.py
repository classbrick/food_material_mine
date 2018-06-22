from config.config_st import config_st
def get_config():
    config = config_st()
    config.tfrecord_addr = 'E:/test_data/73_materials/test_tfrecord/'
    config.tfrecord_test_addr = 'E:/test_data/73_materials/test_tfrecord/'
    config.debug_step_len = 100
    config.batchsize = 1
    config.loss_type = 'entropy_loss'
    config.net_type = 'mobilenet_v2'
    config.class_num = 73
    config.result_addr = './output/chamo_560000.000000_0.014772_0.650000/chamo.ckpt'
    config.ckpt_name = 'chamo_560000.000000_0.014772_0.650000'
    config.loading_his = True
    config.accuracy_type = 'multi'
    return config
