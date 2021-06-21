# coding:utf-8
import tensorflow as tf
import numpy as np
import os
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import PIL.Image
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions)


def readimg(image_path, max_dim=None):
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)


# Normalize an image
def deprocess(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8).numpy()


# 用于调试时查看图像，测试时不用
def show(img):
    plt.imshow(img)
    plt.show()


###-----TODO 1 BEGIN: 梯度上升法损失函数设计------------
###注意，该loss将被tf.function调用，应尽量用tf算子完成计算
def calc_loss(img, model):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    # 对每个选定特征层，取其前1/8通道[0:1/8 num_channel]，
    # 用reduce_mean对前1/8通道上的所有结点求均值，作为该层的loss分量
    for act in layer_activations:
        num_channel = int(1/8 * act.shape[-1])
        loss = tf.math.reduce_mean(act[:, :, :, 0:num_channel])
        losses.append(loss)

    # 返回各层loss分量之和
    return tf.reduce_sum(losses)


###-----TODO 1 END


###-----TODO 2 BEGIN:  梯度上升法修改输入图像实现deepdream------------
###本实验中，梯度修改的范围被限制在一个矩形区域
class DeepPartDream(tf.Module):
    def __init__(self, model):
        self.model = model

    # 2.1 根据输入参数类型，完成tf.function的函数签名，
    # 函数签名可以限制参数类型，防止因参数类型变化引起的计算图重绘，提高计算效率
    # 限定求导区域为矩形，用x,y,w,h四个整数参数表示
    # 坐标系统原点在图像左上角，横轴为x,纵轴y
    # img_tile的左上顶点坐标（x,y),宽和高为w,h
    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.int32),)
    )
    def __call__(self, img, steps, step_size, x, y, w, h):
        # 绘制计算图时，才会调用print("Tracing")
        # 计算图不变时，后续对tf.function的调用不会调用print,只会调用tf.print
        # 如果后面计算图发生重绘，则会再次看见打印的Tracing
        print("Tracing")
        loss = tf.constant(0.0)
        gradients = tf.zeros_like(img)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # 2.2 完成指定的img_tile区域前向记录过程，用于对指定区域img_tile求导
                # `GradientTape` 默认只观察记录变量，对于输入img应手动watch
                tape.watch(img)
                img_tile = img[y:y + h, x:x + w, :]
                loss = calc_loss(img_tile, self.model)

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)

            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img

###-----TODO 2 END


base_model = MobileNetV2(weights=None, include_top=False)
base_model.load_weights('./model/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_192_no_top.h5')

# Maximize the activations of these layers
names = ['block_13_project', 'block_16_project']
layers = [base_model.get_layer(name).output for name in names]

# Create the feature extraction model
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
deepdream = DeepPartDream(dream_model)

# 对固定尺寸图像应用梯度上升进行多次迭代修改图像
def run_deep_dream_in_part(img, x, y, w, h, steps=100, step_size=0.01):
    # Convert from uint8 to the range expected by the model.
    img = preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    w = tf.convert_to_tensor(w)
    h = tf.convert_to_tensor(h)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining > 100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps

        loss, img = deepdream(img, run_steps, tf.constant(step_size), x, y, w, h)

        # show(deprocess(img))
        print("Step {}, loss {}".format(step, loss))

    result = deprocess(img)
    # show(result)

    return result

###-----TODO 3 BEGIN:  多尺度融合------------
def run_deep_dream_with_parts_octaves(img, part_position, steps_per_octave=100, step_size=0.01,
                                      octaves=range(-2, 3), octave_scale=1.3):
    start = time.time()
    base_shape = tf.shape(img)
    base_part_position = tf.convert_to_tensor(part_position)
    print(base_part_position)
    img = tf.keras.preprocessing.image.img_to_array(img)

    initial_shape = img.shape[:-1]
    img = tf.image.resize(img, initial_shape)
    # 在多个图像尺度上迭代修改输入图像，
    # 注意，需修改的限定区域part_position也随输入图像尺度同步缩放
    # 每个迭代的新尺度相比原图像尺度比例为  octave_scale**octave
    # 每次迭代在上次迭代的中间结果基础上进行，需要在迭代前将上次中间结果img以及part_position进行resize
    for octave in octaves:
        new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * (octave_scale ** octave)
        img = tf.image.resize(img, tf.cast(new_size, tf.int32))
        part_position = tf.cast(tf.cast(base_part_position, tf.float32) * (octave_scale ** octave), tf.int32)
        print(part_position)
        print(type(part_position))
        print(part_position.shape)
        y = part_position[0]
        x = part_position[1]
        h = part_position[2]
        w = part_position[3]
        print(y, x, w, h)
        img = run_deep_dream_in_part(img, x, y, w, h, steps=steps_per_octave, step_size=0.01)
        # show(img)
        print("Octave {}".format(octave))

    img = tf.image.resize(img, base_shape[:-1])
    img = tf.image.convert_image_dtype(img / 255.0, dtype=tf.uint8)
    # show(img)
    end = time.time()
    print("Time: {}".format(end - start))
    return img

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

max_dim = 224
IMG_PATH = './img/cat_dog.png'

original_img = readimg(IMG_PATH, max_dim)
part_position = [10, 70, 70, 70]  # 注意，顺序是：y,x,h,w
dream_img = run_deep_dream_with_parts_octaves(original_img, part_position)
ims = PIL.Image.fromarray(dream_img.numpy())
ims.save("./myfig/cat_dog_dream.png")

#用图片对比的方法测试网络结构是否正确
test_img = mpimg.imread('./myfig/cat_dog_dream.png')
answer_img = mpimg.imread('./answer/cat_dog_dream_touge.png')
assert((answer_img == test_img).all())
print('Success!')
###-----TODO 3 END

###---------------测试代码----------------
'''
import numpy as np
import PIL.Image
from ChannelDeepDream_work import run_deep_dream_with_parts_octaves, readimg
import matplotlib.image as mpimg
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

max_dim=224
IMG_PATH='./img/cat_dog.png'

original_img  = readimg(IMG_PATH,max_dim) 
part_position=[10,70,70,70] # 注意，顺序是：y,x,h,w
dream_img = run_deep_dream_with_parts_octaves(original_img, part_position)
ims = PIL.Image.fromarray(dream_img.numpy())
ims.save("./myfig/cat_dog_dream.png")

#用图片对比的方法测试网络结构是否正确
test_img = mpimg.imread('./myfig/cat_dog_dream.png') 
answer_img= mpimg.imread('./answer/cat_dog_dream_touge.png') 
assert((answer_img == test_img).all())
print('Success!')
'''