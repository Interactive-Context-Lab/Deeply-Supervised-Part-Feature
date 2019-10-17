from net.PartNetwork import Partbased
from net.se_resnext_backbone import SE_ResNeXt

import tensorflow as tf

class DSPF(object):
    def __init__(self, x, class_num, training_flag, num_dims=256, drop_rate=0.2, atten=True, ds=True, part=True, all=False):
        self.part_list = []
        self.ds_list = []
        self.local_x_list = []
        #############   get model   ###############################
        self.end_points = SE_ResNeXt(x, None, is_training=training_flag, data_format='channels_last', trainable=True, atten=atten)

        if ds:
            block1_x_list, block1_logits_list = Partbased(self.end_points["block1"], num_classes=class_num,
                                                          is_training=training_flag,
                                                          num_dims=num_dims, layer_name="block1/", all=True)
            block2_x_list, block2_logits_list = Partbased(self.end_points["block2"], num_classes=class_num,
                                                          is_training=training_flag,
                                                          num_dims=num_dims, layer_name="block2/", all=True)
            block3_x_list, block3_logits_list = Partbased(self.end_points["block3"], num_classes=class_num,
                                                          is_training=training_flag,
                                                          num_dims=num_dims, layer_name="block3/", all=True)

            self.ds_list.extend(block1_logits_list)
            self.ds_list.extend(block2_logits_list)
            self.ds_list.extend(block3_logits_list)

            self.local_x_list.extend(block1_x_list)
            self.local_x_list.extend(block2_x_list)
            self.local_x_list.extend(block3_x_list)

        ### Part-level features
        local_x_part, part_logits_list = Partbased(self.end_points["block4"],
                                                   num_classes=class_num, is_training=training_flag,
                                                   num_dims=num_dims, drop_rate=drop_rate,
                                                   part=part, all=all)

        self.part_list.extend(part_logits_list)

        self.local_x_list.extend(local_x_part)

    def get_endpoints(self):
        return self.end_points, self.ds_list, self.part_list, self.local_x_list