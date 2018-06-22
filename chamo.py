from config.config_st import config_st
def get_config():
    config=config_st()
    config.tfrecord_addr='/home/leo/Downloads/chamo/v2_material/train_tfrecord/'
    config.tfrecord_test_addr = '/home/leo/Downloads/chamo/v2_material/train_tfrecord/'
    config.debug_step_len = 100
    config.batchsize=32
    config.loss_type='entropy_loss'
    config.net_type = 'mobilenet_v2'
    config.class_num = 73
    config.result_addr = './output/'
    config.ckpt_name='chamo_22000.000000_0.039581_0.500000'
    config.loading_his = True
    config.accuracy_type = 'multi'
    return config
