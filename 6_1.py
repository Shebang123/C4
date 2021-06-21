# coding=utf-8
import numpy as np
import os
import tensorflow as tf
# 自己看需要啥就import啥

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

NUM_CLASSES = 10


###-----TODO 1 BEGIN: 构建迁移上层网络结构------------
# 可以用任何结构，任何定义方式(Sequential，Class Model都可以)，只要能和迁移的base_model连接就行
# 后端不要太复杂，注意评测时间
def build_top():
    top_model = tf.keras.Sequential([
        Flatten(),
        Dense(NUM_CLASSES, activation='softmax'),
    ])
    return top_model


###-----TODO 1 END------------------------------------

###-----TODO 2 BEGIN: 设计训练策略------------
# 需要指定optimizer,batch_size,是否加载预训练模型参数load_weight(不想用预训练参数重头训也行)
# lr_scheduler，lr_reducer等callbacks函数可选，不用也行
def training_options():
    def lr_schedule(epoch):
        lr = 0.01
        print("learning rate: ", lr)
        return lr

    lr_scheduler = LearningRateScheduler(schedule=lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=0,
                                   cooldown=0,
                                   patience=0,
                                   min_lr=0,
                                   verbose=0)

    callbacks = []  # 如果不用lr_scheduler，lr_reducer等，可以传一个空表：callbacks = []
    optimizer = tf.keras.optimizers.Adam(0.01)
    batch_size = 64
    load_weight = True # 是否选加载预训练模型参数，True or False
    return callbacks, optimizer, batch_size, load_weight


###-----TODO 2 END------------------------------------

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

IMG_SIZE = 32 # All images will be resized to IMG_SIZE x IMG_SIZE
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

callbacks,optimizer,batch_size,load_weight=training_options()

def make_data(batch_size):
    path = 'D:\Study\C4\mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()

    x_train = x_train[:, :, :, np.newaxis].repeat([3],axis=-1)
    x_test = x_test[:, :, :, np.newaxis].repeat([3],axis=-1)

    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)
    x_train = (x_train/127.5) - 1
    x_test = (x_test/127.5) - 1
    x_train = tf.image.resize(x_train, [IMG_SIZE, IMG_SIZE])
    x_test = tf.image.resize(x_test, [IMG_SIZE, IMG_SIZE])
    train_batches = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(batch_size)
    test_batches = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train_batches,test_batches

train_batches,test_batches=make_data(batch_size)


# Create the base model from the pre-trained model MobileNet V2
mobilenet = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,alpha=1,
                                               include_top=False,weights=None)
if load_weight:
    mobilenet.load_weights('D:\Study\C4\model\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5')

base_model=tf.keras.Model(inputs=mobilenet.inputs,outputs=mobilenet.get_layer('expanded_conv_project_BN').output)

top_model=build_top()
model = tf.keras.Sequential([
    base_model,
    top_model,
])

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

base_model.summary()
top_model.summary()
history = model.fit(train_batches,
                    epochs=5,
                    validation_data=test_batches,
                    callbacks=callbacks,verbose=2)

if history.history['val_accuracy'][-1]>0.7:
    print("Success!")
# 以下为测试代码
'''
import numpy as np
import tensorflow as tf
from transfer_mnist_fit_work import build_top,training_options

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

IMG_SIZE = 32 # All images will be resized to IMG_SIZE x IMG_SIZE
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

callbacks,optimizer,batch_size,load_weight=training_options()

def make_data(batch_size):
    path = './dataset/mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()

    x_train = x_train[:, :, :, np.newaxis].repeat([3],axis=-1)
    x_test = x_test[:, :, :, np.newaxis].repeat([3],axis=-1)

    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)
    x_train = (x_train/127.5) - 1
    x_test = (x_test/127.5) - 1
    x_train = tf.image.resize(x_train, [IMG_SIZE, IMG_SIZE])
    x_test = tf.image.resize(x_test, [IMG_SIZE, IMG_SIZE])
    train_batches = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(batch_size)
    test_batches = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train_batches,test_batches

train_batches,test_batches=make_data(batch_size)


# Create the base model from the pre-trained model MobileNet V2
mobilenet = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,alpha=1,
                                               include_top=False,weights=None)
if load_weight:
    mobilenet.load_weights('./model/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5')

base_model=tf.keras.Model(inputs=mobilenet.inputs,outputs=mobilenet.get_layer('expanded_conv_project_BN').output)

top_model=build_top()
model = tf.keras.Sequential([
    base_model,
    top_model,
])

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_batches,
                    epochs=5,
                    validation_data=test_batches,
                    callbacks=callbacks,verbose=2)

if history.history['val_accuracy'][-1]>0.7:
    print("Success!")
'''
