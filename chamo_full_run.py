from config.config_st import config_st
def get_config():
    config = config_st()
    config.tfrecord_addr = 'D:/Users/zhengnan.luo/PycharmProjects/food_material_brick/tfrecord/'
    config.tfrecord_test_addr = 'D:/Users/zhengnan.luo/PycharmProjects/food_material_brick/tfrecord/'
    config.debug_step_len = 100
    config.batchsize = 32
    config.loss_type = 'entropy_loss'
    config.net_type = 'mobilenet_v2'
    config.class_num = 73
    config.result_addr = './output/'
    config.ckpt_name = 'chamo_532000.000000_0.013970_0.800000'
    config.loading_his = True
    config.accuracy_type = 'multi'
    return config
