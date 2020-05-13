'''
卷积神经网络
使用resnet50 残差网络训练
resnet50 处理的图片为224*224
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文使用
plt.rcParams['axes.unicode_minus'] = False # 用于显示负号

# 数据集的地址
train_dir = "Resources/Kaggle_10_monkey/training/training"
valid_dir = "Resources/Kaggle_10_monkey/validation/validation"
label_file = "Resources/Kaggle_10_monkey/monkey_labels.txt"
# print(os.path.exists(train_dir))
# print(os.path.exists(valid_dir))
# print(os.path.exists(label_file))

# 观察标签
labels = pd.read_csv(label_file, header=0)
# print(labels)


# 做卷积的时候所有图片的尺寸应该是一样的
height = 224
width = 224
channels = 3
batch_size = 24
num_classes = 10

# 读取训练数据并作数据增强
# 确定一些读取格式要求
train_datagen = ImageDataGenerator(
    preprocessing_function = keras.applications.resnet50.preprocess_input,
    # 图片旋转的角度范围，用来数据增强
    rotation_range=40,
    # 水平平移
    width_shift_range=0.2,
    # 高度平移
    height_shift_range=0.2,
    # 剪切强度
    shear_range=0.2,
    # 缩放强度
    zoom_range=0.2,
    # 水平翻转
    horizontal_flip=True,
    # 对图片做处理时需要填充图片，用最近的像素点填充
    fill_mode="nearest")
# 测试集数据无需使用数据增强
test_datagen = ImageDataGenerator(rescale=1. / 255)

# 读取训练数据
train_generator = train_datagen.flow_from_directory(
    train_dir,
    # 读取后将图片存什么大小
    target_size = (height, width),
    batch_size = batch_size,
    seed = 7,
    shuffle = True,
    # label的编码格式：这里为one-hot编码
    class_mode = 'categorical')


# 读取验证数据
valid_datagen = ImageDataGenerator(
                preprocessing_function = keras.applications.resnet50.preprocess_input)
valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    # 读取后将图片存什么大小
    target_size = (height, width),
    batch_size = batch_size,
    seed = 7,
    shuffle = False,
    # label的编码格式：这里为one-hot编码
    class_mode = 'categorical')


train_num = train_generator.samples
valid_num = valid_generator.samples
print(train_num, valid_num)


# for i in range(2):
#     x, y = train_generator.next()
#     print(x.shape, y.shape)
#     print(y)

# 构建cnn模型
'''
卷积层：
conv2d 参数说明：
    filters 代表有多少个卷积核
    kernel_size 卷积核的大小
    padding 输出图像是否与原来等大小
    activation 激活函数
    input_shape 输入图像的大小[width, height, channels] 长 宽 通道数
池化层：
MaxPool2D：

'''

resnet50_fine_tune = keras.models.Sequential()
# include_top 最后一层去掉（resnet最后一层输出的为1000类）
# weights= 'imagenet' 使用现有的数据库 进行模型的初始化
resnet50_fine_tune.add(keras.applications.ResNet50(include_top=False,
                                                   pooling='avg',
                                                   weights='imagenet'))
resnet50_fine_tune.add(keras.layers.Dense(num_classes, activation='softmax'))
resnet50_fine_tune.layers[0].trainable = False
# 编译模型
resnet50_fine_tune.compile(loss="categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
resnet50_fine_tune.summary()

# 训练模型
epochs = 1
# 数据是generator出来的，所以不能直接用fit
history = resnet50_fine_tune.fit_generator(train_generator,
                               steps_per_epoch = train_num // batch_size,
                               epochs=epochs,
                              validation_data = valid_generator,
                              validation_steps = valid_num // batch_size)

print(history.history.keys())
# 以上print输出内容：dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])


# def plot_learning_curves(history, label, epochs, min_value, max_value):
#     data = {}
#     data[label] = history.history[label]
#     data['val_' + label] = history.history['val_' + label]
#     pd.DataFrame(data).plot(figsize=(8, 5))
#     plt.grid(True)
#     plt.axis([0, epochs, min_value, max_value])
#     plt.show()
#
#
# plot_learning_curves(history, 'accuracy', epochs, 0, 1)
# plot_learning_curves(history, 'loss', epochs, 1.5, 2.5)


# 查看增强后的图像效果
def showGenImage():
    fnames = [os.path.join('Resources/Kaggle_10_monkey/training/training/n0', fname) for
              fname in os.listdir('Resources/Kaggle_10_monkey/training/training/n0')]

    # 载入第3张图像
    img_path = fnames[3]
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(32, 32))
    x = image.img_to_array(img)
    plt.figure(1, figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(image.array_to_img(x))
    plt.title('original image')
    # 数据增强后的图像
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in train_datagen.flow(x, batch_size=1):
        plt.subplot(2, 2, i + 2)
        plt.imshow(image.array_to_img(batch[0]))
        plt.title('after augumentation %d' % (i + 1))
        i = i + 1
        if i % 3 == 0:
            break
    plt.show()
