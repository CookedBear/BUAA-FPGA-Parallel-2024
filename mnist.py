"""
　　# -*- coding: utf-8 -*-
　　# @Time    : 2024/4/3 20:45
　　# @Author  : CookedBear
　　# @File    : mnist.py
"""

import tvm
from tvm import relay
import onnx
import numpy as np
import struct


test_images_idx3_ubyte_file = 't10k-images.idx3-ubyte'   # 测试图片数据
test_labels_idx1_ubyte_file = 't10k-labels.idx1-ubyte'   # 测试标签数据


def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


test_images = load_test_images()
test_labels = load_test_labels()
print("data installed!")

# 加载ONNX模型
onnx_model = onnx.load("mnist_model.onnx")

# 设置target为CPU，如果要用GPU可以设置为"cuda"
target = "llvm"

# 设置输入信息，这里 "1, 1, 28, 28" 对应于(batch_size, num_channels, H, W)
input_shape = (1, 1, 28, 28)
input_name = "input.1"  # 根据实际模型可能需要修改

# 使用relay模块将ONNX模型转换为TVM的表示
mod, params = relay.frontend.from_onnx(onnx_model, shape={input_name: input_shape})

# 使用TVM进行模型优化，target指定优化的硬件
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# 创建TVM运行时
ctx = tvm.runtime.device(target, 0)
module = tvm.contrib.graph_executor.GraphModule(lib["default"](ctx))

print("model loaded!")
print("tranvse " + str(test_images[0].dtype) + " to ", end="")

test_images = np.expand_dims(test_images, axis=1) / 255.0
test_images = test_images.astype(np.float32)
test_labels = test_labels.astype('int')
print(test_images[0].dtype)

correct_predictions = 0
total_predictions = len(test_images)

print("evaluation start!")
# 对于每一张图片，我们都进行预测并与真实标签比较
for i in range(total_predictions):
    # 设置输入并运行模型
    module.set_input(input_name, tvm.nd.array(test_images[i:i+1]))
    module.run()
    # 获取输出
    output_data = module.get_output(0).asnumpy()
    # 得到预测结果
    predicted_label = np.argmax(output_data[0])
    # 检查预测是否正确
    correct_predictions += (predicted_label == test_labels[i])

# 计算准确率
accuracy = correct_predictions / total_predictions
print(f"Model accuracy on test set: {accuracy * 100:.2f}%")
