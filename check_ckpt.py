import tensorflow as tf
# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#
# print_tensors_in_checkpoint_file("senetmodel/Inception_resnet_v2.ckpt", all_tensors=False,
#                                  all_tensor_names=True, tensor_name="")
# #
# print_tensors_in_checkpoint_file("pretrain/all_baseline/model.ckpt-100", all_tensors=False,
#                                  all_tensor_names=True, tensor_name="")

# print_tensors_in_checkpoint_file("pretrain/model.ckpt-60", all_tensors=True,
#                                  all_tensor_names=True, tensor_name="stem")


from tensorflow.python import pywrap_tensorflow
import os
#
# # checkpoint_path = os.path.join(model_dir, "model.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader('model/190608_164544/model.ckpt-77')
var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:
    # if "stem" in key and "Adam" not in key:
    print("tensor_name: ", key)
    # print(reader.get_tensor(key)) # Remove this is you want to print only variable names