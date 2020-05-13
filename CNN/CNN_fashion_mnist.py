'''
    实现卷积神经网络
'''
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd

#取消AVX2的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 加载数据
(train_img, train_lab), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
#将数据分为： 训练集 验证集 测试集
#可以将训练数据据拆分为 训练集和验证集
valid_image, train_image = train_img[:5000], train_img[5000:]
valid_label, train_label = train_lab[:5000], train_lab[5000:]


# 标准化
scaler = StandardScaler()
train_image_scaler = scaler.fit_transform(train_image.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28,1)
valid_image_sacler = scaler.transform(valid_image.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28,1)
test_image_scaler = scaler.transform(test_image.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28,1)

# 模型创建
# 1 模型选择
def Creat_CNN_Model():
    model = tf.keras.Sequential()
    # 2 模型搭建
    # 搭建神经网络 构建三层 卷积层和池化层
    # 卷积层 filters 输出多少维数据
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,
                                     padding='same',
                                     input_shape=(28,28,1)))
    # 池化层
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))
    # 卷积层
    model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,
                                     padding='same'))
    # 池化层
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))
    # 卷积层
    model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,
                                     padding='same'))
    # 池化层
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))
    # 分类模型使用Flatten进行二维图像的一维化
    model.add(tf.keras.layers.Flatten())
    # 输出层 分类模型最后输出层使用sofetmax确认类型
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


model = Creat_CNN_Model()

# 3 编译模型
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['acc'])


# 4 训练模型
history = model.fit(train_image_scaler, train_label, epochs=2, validation_data=(valid_image_sacler, valid_label))


# 5 评价模型
test_loss, test_acc = model.evaluate(test_image_scaler, test_label)
print('Test accuracy:', test_acc)

# 6 打印学习曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True) #是否带有网格
    plt.gca().set_ylim(0, 1)
    plt.show()
plot_learning_curves(history)