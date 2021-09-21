#导入模块
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

#加载数据
fashion_mnist = keras.datasets.fashion_mnist
#训练集和测试集的划分
(train_images, train_lables), (test_images, test_lables) = fashion_mnist.load_data()

#进行数据预处理，将每个像素点都压缩在0和1之间。
train_images = train_images/255.0
test_images = test_images/255.0

#搭建简单的神经网络
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128,activation="relu"),
        keras.layers.Dense(18)
    ])

#编译模型
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()]
    )

#返回模型
    return model

#构建模型
new_model = create_model()

#训练模型
new_model.fit(train_images,train_lables,epochs=50)

#保存模型
new_model.save("model/my_model3.h5")

# 打印图像
# plt.figure()
# plt.xticks()
# plt.yticks()
# plt.imshow(train_images[0])
# plt.show()

