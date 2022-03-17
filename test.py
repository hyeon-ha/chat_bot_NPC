
import tensorflow as tf


print(tf.test.is_built_with_cuda())

print(tf.test.is_built_with_gpu_support())

print(tf.test.gpu_device_name())


import torch

print(torch.cuda.is_available())
print(torch.cuda.current_device())