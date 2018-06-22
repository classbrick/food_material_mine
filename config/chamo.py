from config.config_st import config_st
def get_config():
    config=config_st()
    config.tfrecord_addr='D:/Users/zhengnan.luo/PycharmProjects/food_material_brick/tfrecord/'
    config.tfrecord_test_addr = 'D:/Users/zhengnan.luo/PycharmProjects/food_material_brick/tfrecord/'
    config.debug_step_len = 100
    config.batchsize=32
    config.loss_type='entropy_loss'
    config.net_type = 'mobilenet_v2'
    config.class_num = 102
    config.result_addr = \
        'D:/Users/zhengnan.luo/PycharmProjects/food_material_brick/output/chamo_676000.000000_0.012152_0.650000/chamo.ckpt'
    config.ckpt_name = \
        'D:/Users/zhengnan.luo/PycharmProjects/food_material_brick/output/chamo_676000.000000_0.012152_0.650000/chamo.ckpt'
    config.loading_his = True
    config.accuracy_type = 'multi'
    return config
